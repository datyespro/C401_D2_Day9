"""
workers/synthesis.py — Synthesis Worker
Sprint 2: Tổng hợp câu trả lời từ retrieved_chunks và policy_result.

Input (từ AgentState):
    - task: câu hỏi
    - retrieved_chunks: evidence từ retrieval_worker
    - policy_result: kết quả từ policy_tool_worker

Output (vào AgentState):
    - final_answer: câu trả lời cuối với citation [source_name]
    - sources: danh sách nguồn tài liệu được cite
    - confidence: mức độ tin cậy (0.0 - 1.0) — tính từ chunk scores, không hard-code

Gọi độc lập để test:
    python workers/synthesis.py

Author: M6 - Nguyễn Anh Đức - Documentation & Synthesis Owner
"""

import os
from dotenv import load_dotenv

load_dotenv()

WORKER_NAME = "synthesis_worker"

# Ngưỡng confidence để trigger HITL
HITL_CONFIDENCE_THRESHOLD = 0.4

# Keyword nhận biết câu abstain
ABSTAIN_KEYWORDS = [
    "không đủ thông tin",
    "không có trong tài liệu",
    "không tìm thấy",
    "insufficient information",
    "not found in",
    "no information",
]

SYSTEM_PROMPT = """Bạn là trợ lý IT Helpdesk và CS nội bộ. Nhiệm vụ của bạn là tổng hợp câu trả lời CHÍNH XÁC từ tài liệu được cung cấp.

Quy tắc BẮT BUỘC:
1. CHỈ trả lời dựa vào [TÀI LIỆU THAM KHẢO] bên dưới. TUYỆT ĐỐI không dùng kiến thức bên ngoài.
2. Sau mỗi thông tin quan trọng, đánh số nguồn: [tên_file.txt] hoặc [1], [2] theo thứ tự tài liệu.
3. Nếu [TÀI LIỆU THAM KHẢO] không đủ để trả lời → nói rõ:
   "Không đủ thông tin trong tài liệu nội bộ để trả lời câu hỏi này."
4. Nếu có [POLICY EXCEPTIONS] → nêu rõ exception TRƯỚC KHI kết luận.
5. Trả lời súc tích, có cấu trúc (dùng gạch đầu dòng nếu cần).
6. KHÔNG bịa số liệu, tên người, hoặc quy trình không có trong tài liệu."""


# ─────────────────────────────────────────────
# LLM Caller
# ─────────────────────────────────────────────

def _call_llm(messages: list) -> str:
    """
    Gọi LLM để tổng hợp câu trả lời.
    Thử OpenAI trước, fallback sang Gemini, cuối cùng là rule-based fallback.
    """
    # Option A: OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key.startswith("sk-"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,   # Low temperature để grounded, ít hallucinate
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  OpenAI call failed: {e}")

    # Option B: Google Gemini
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config={"temperature": 0.1, "max_output_tokens": 600},
            )
            # Ghép system + user messages thành 1 prompt
            combined = "\n\n".join([m["content"] for m in messages])
            response = model.generate_content(combined)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️  Gemini call failed: {e}")

    # Fallback: không gọi được LLM → abstain, không hallucinate
    return "Không đủ thông tin trong tài liệu nội bộ để trả lời câu hỏi này. (LLM unavailable — kiểm tra API key trong .env)"


# ─────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────

def _build_context(chunks: list, policy_result: dict) -> tuple[str, list]:
    """
    Xây dựng context string từ chunks và policy result.

    Returns:
        (context_str, numbered_sources_list)
    """
    parts = []
    numbered_sources = []   # Danh sách nguồn theo thứ tự số [1], [2], ...

    if chunks:
        parts.append("=== TÀI LIỆU THAM KHẢO ===")
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "").strip()
            score = chunk.get("score", 0.0)

            # Track sources theo số thứ tự
            if source not in numbered_sources:
                numbered_sources.append(source)

            source_idx = numbered_sources.index(source) + 1
            parts.append(
                f"[{source_idx}] Nguồn: {source} (relevance: {score:.2f})\n{text}"
            )

    # Policy exceptions — nếu có
    if policy_result:
        exceptions = policy_result.get("exceptions_found", [])
        version_note = policy_result.get("policy_version_note", "")

        if exceptions:
            parts.append("\n=== POLICY EXCEPTIONS (ƯU TIÊN HIỂN THỊ) ===")
            for ex in exceptions:
                rule = ex.get("rule", "")
                src = ex.get("source", "")
                parts.append(f"⚠️  {rule} [Nguồn: {src}]")

        if version_note:
            parts.append(f"\n⚠️  GHI CHÚ VERSION: {version_note}")

        # Policy applies = False nhưng không có exception cụ thể
        if not policy_result.get("policy_applies", True) and not exceptions:
            parts.append("\n⚠️  Policy check: yêu cầu này KHÔNG được phép theo quy định hiện hành.")

    if not parts:
        return "(Không có context — không có tài liệu nào được retrieve)", []

    return "\n\n".join(parts), numbered_sources


# ─────────────────────────────────────────────
# Confidence Estimator
# ─────────────────────────────────────────────

def _estimate_confidence(chunks: list, answer: str, policy_result: dict) -> float:
    """
    Tính confidence thực tế dựa vào nhiều tín hiệu — KHÔNG hard-code.

    Tín hiệu:
    1. Chunk quality: weighted average của cosine score từ ChromaDB
    2. Abstain signal: nếu answer tự nhận không đủ thông tin → low confidence
    3. Answer length: câu trả lời quá ngắn → ít detail → lower confidence
    4. Exception penalty: có exception phức tạp → harder case → reduced confidence
    5. No chunks penalty: không có evidence → rất low confidence
    """
    # Trường hợp không có chunks
    if not chunks:
        return 0.1

    # Abstain signal — model tự nhận không đủ thông tin
    answer_lower = answer.lower()
    if any(kw in answer_lower for kw in ABSTAIN_KEYWORDS):
        return 0.25

    # --- Chunk quality signal (40% weight) ---
    valid_scores = [c.get("score", 0.5) for c in chunks if isinstance(c.get("score"), (int, float))]
    if valid_scores:
        # Top chunk score có trọng số cao hơn (top-1 quan trọng nhất)
        sorted_scores = sorted(valid_scores, reverse=True)
        if len(sorted_scores) >= 2:
            chunk_quality = sorted_scores[0] * 0.6 + sum(sorted_scores[1:]) / len(sorted_scores[1:]) * 0.4
        else:
            chunk_quality = sorted_scores[0]
    else:
        chunk_quality = 0.4  # default khi không có score

    # --- Answer length signal (20% weight) ---
    # Câu trả lời ngắn hơn 80 ký tự → chưa đủ detail
    length_factor = min(1.0, len(answer.strip()) / 200)

    # --- Citation signal (20% weight) ---
    # Có citation [x] trong câu trả lời → tốt hơn
    has_citation = (
        "[" in answer and "]" in answer
        and any(src_name.split(".")[0] in answer for src_name in [c.get("source", "") for c in chunks])
    ) or any(f"[{i}]" in answer for i in range(1, len(chunks) + 1))

    citation_factor = 1.0 if has_citation else 0.7

    # --- Exception complexity penalty ---
    exceptions = policy_result.get("exceptions_found", []) if policy_result else []
    exception_penalty = min(0.15, 0.05 * len(exceptions))  # Tối đa −15%

    # Tổng hợp
    raw_confidence = (
        chunk_quality * 0.5
        + length_factor * 0.25
        + citation_factor * 0.25
    ) - exception_penalty

    return round(max(0.1, min(0.95, raw_confidence)), 2)


# ─────────────────────────────────────────────
# Core Synthesize Function
# ─────────────────────────────────────────────

def synthesize(task: str, chunks: list, policy_result: dict) -> dict:
    """
    Tổng hợp câu trả lời từ chunks và policy context.

    Returns:
        {
            "answer": str,           # câu trả lời với citation
            "sources": list,         # danh sách sources được cite
            "confidence": float,     # 0.0-1.0
        }
    """
    context, numbered_sources = _build_context(chunks, policy_result)

    # Nếu hoàn toàn không có context → abstain ngay, không gọi LLM
    if not chunks and not policy_result:
        return {
            "answer": "Không đủ thông tin trong tài liệu nội bộ để trả lời câu hỏi này.",
            "sources": [],
            "confidence": 0.1,
        }

    # Build prompt messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Câu hỏi: {task}\n\n"
                f"{context}\n\n"
                "Hãy trả lời câu hỏi dựa ĐÚNG vào tài liệu trên. "
                "Nhớ đánh số nguồn [1], [2] hoặc [tên_file] sau thông tin trích dẫn."
            ),
        },
    ]

    answer = _call_llm(messages)
    sources = list({c.get("source", "unknown") for c in chunks if c.get("source")})
    confidence = _estimate_confidence(chunks, answer, policy_result)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────
# Worker Entry Point
# ─────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    Worker entry point — gọi từ graph.py.

    Đọc từ state:  task, retrieved_chunks, policy_result
    Ghi vào state: final_answer, sources, confidence, hitl_triggered (nếu cần)
    """
    task = state.get("task", "")
    chunks = state.get("retrieved_chunks", [])
    policy_result = state.get("policy_result", {})

    state.setdefault("workers_called", [])
    state.setdefault("history", [])
    state.setdefault("worker_io_logs", [])

    state["workers_called"].append(WORKER_NAME)

    # Log worker IO (theo contract)
    worker_io = {
        "worker": WORKER_NAME,
        "input": {
            "task": task,
            "chunks_count": len(chunks),
            "has_policy_result": bool(policy_result),
            "has_exceptions": bool(
                policy_result.get("exceptions_found") if policy_result else False
            ),
        },
        "output": None,
        "error": None,
    }

    try:
        result = synthesize(task, chunks, policy_result)

        state["final_answer"] = result["answer"]
        state["sources"] = result["sources"]
        state["confidence"] = result["confidence"]

        # HITL trigger: confidence quá thấp → cần human review
        if result["confidence"] < HITL_CONFIDENCE_THRESHOLD:
            state["hitl_triggered"] = True
            state["history"].append(
                f"[{WORKER_NAME}] ⚠️  HITL triggered: confidence={result['confidence']} < {HITL_CONFIDENCE_THRESHOLD}"
            )

        worker_io["output"] = {
            "answer_length": len(result["answer"]),
            "sources": result["sources"],
            "confidence": result["confidence"],
            "hitl_triggered": state.get("hitl_triggered", False),
        }
        state["history"].append(
            f"[{WORKER_NAME}] answer generated | "
            f"confidence={result['confidence']} | "
            f"sources={result['sources']} | "
            f"hitl={state.get('hitl_triggered', False)}"
        )

    except Exception as e:
        error_msg = f"SYNTHESIS_ERROR: {e}"
        worker_io["error"] = {"code": "SYNTHESIS_FAILED", "reason": str(e)}
        state["final_answer"] = error_msg
        state["confidence"] = 0.0
        state["sources"] = []
        state["hitl_triggered"] = True   # Safety: escalate nếu synthesis fail
        state["history"].append(f"[{WORKER_NAME}] ERROR: {e}")

    state["worker_io_logs"].append(worker_io)
    return state


# ─────────────────────────────────────────────
# Standalone Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Synthesis Worker — Standalone Test")
    print("=" * 60)

    # Test 1: SLA P1 — normal retrieval case
    print("\n--- Test 1: SLA P1 retrieval ---")
    state1 = {
        "task": "SLA ticket P1 là bao lâu? Ai nhận thông báo đầu tiên?",
        "retrieved_chunks": [
            {
                "text": (
                    "Ticket P1: Phản hồi ban đầu trong 15 phút kể từ khi ticket được tạo. "
                    "Xử lý và khắc phục hoàn toàn trong 4 giờ. "
                    "Escalation: tự động escalate lên Senior Engineer nếu không có phản hồi sau 10 phút. "
                    "Thông báo qua: Slack #incidents và email on-call engineer."
                ),
                "source": "sla_p1_2026.txt",
                "score": 0.92,
            }
        ],
        "policy_result": {},
    }
    result1 = run(state1.copy())
    print(f"Answer:\n{result1['final_answer']}")
    print(f"Sources: {result1['sources']}")
    print(f"Confidence: {result1['confidence']}")
    print(f"HITL triggered: {result1.get('hitl_triggered', False)}")

    # Test 2: Flash Sale exception case
    print("\n--- Test 2: Flash Sale exception ---")
    state2 = {
        "task": "Khách hàng Flash Sale yêu cầu hoàn tiền vì sản phẩm lỗi nhà sản xuất — được không?",
        "retrieved_chunks": [
            {
                "text": (
                    "Chính sách hoàn tiền v4: Sản phẩm lỗi do nhà sản xuất được hoàn tiền trong 7 ngày làm việc. "
                    "Ngoại lệ (Điều 3): Đơn hàng Flash Sale không được hoàn tiền dưới bất kỳ hình thức nào. "
                    "Trong trường hợp lỗi nhà sản xuất, Flash Sale customers được đề nghị đổi hàng thay thế."
                ),
                "source": "policy_refund_v4.txt",
                "score": 0.88,
            }
        ],
        "policy_result": {
            "policy_applies": False,
            "policy_name": "refund_policy_v4",
            "exceptions_found": [
                {
                    "type": "flash_sale_exception",
                    "rule": "Đơn hàng Flash Sale không được hoàn tiền (Điều 3, chính sách v4).",
                    "source": "policy_refund_v4.txt",
                }
            ],
            "policy_version_note": "",
        },
    }
    result2 = run(state2.copy())
    print(f"Answer:\n{result2['final_answer']}")
    print(f"Confidence: {result2['confidence']}")
    print(f"HITL triggered: {result2.get('hitl_triggered', False)}")

    # Test 3: Abstain case — không có thông tin
    print("\n--- Test 3: Abstain (không có context) ---")
    state3 = {
        "task": "Mức phạt tài chính khi vi phạm SLA P1 là bao nhiêu?",
        "retrieved_chunks": [],   # Không retrieve được gì → phải abstain
        "policy_result": {},
    }
    result3 = run(state3.copy())
    print(f"Answer:\n{result3['final_answer']}")
    print(f"Confidence: {result3['confidence']}")
    print(f"HITL triggered: {result3.get('hitl_triggered', False)}")

    # Test 4: Multi-hop — cần cross-reference 2 tài liệu (gq09 hardest case)
    print("\n--- Test 4: Multi-hop cross-reference ---")
    state4 = {
        "task": "P1 lúc 2am + cần cấp quyền Level 2 tạm thời cho contractor — cả hai quy trình là gì?",
        "retrieved_chunks": [
            {
                "text": (
                    "SLA P1 lúc ngoài giờ hành chính: On-call engineer nhận thông báo qua PagerDuty trong 5 phút. "
                    "Escalation tự động lên Manager nếu không phản hồi trong 10 phút. "
                    "Thời hạn xử lý P1: 4 giờ kể từ khi tạo ticket."
                ),
                "source": "sla_p1_2026.txt",
                "score": 0.89,
            },
            {
                "text": (
                    "Cấp quyền Level 2 tạm thời: Yêu cầu phê duyệt từ IT Manager và Security Lead. "
                    "Trong tình huống khẩn cấp (P1), IT Manager có thể phê duyệt đơn lẻ. "
                    "Quyền tạm thời có hiệu lực tối đa 24 giờ, phải được thu hồi sau khi xử lý xong."
                ),
                "source": "access_control_sop.txt",
                "score": 0.85,
            },
        ],
        "policy_result": {
            "policy_applies": True,
            "policy_name": "access_control_sop",
            "exceptions_found": [],
            "policy_version_note": "",
        },
    }
    result4 = run(state4.copy())
    print(f"Answer:\n{result4['final_answer']}")
    print(f"Sources: {result4['sources']}")
    print(f"Confidence: {result4['confidence']}")
    print(f"Workers called: {result4.get('workers_called', [])}")

    print("\n✅ synthesis_worker standalone test done.")
