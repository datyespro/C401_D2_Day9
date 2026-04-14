"""
graph.py — Supervisor Orchestrator
Sprint 1: Implement AgentState, supervisor_node, route_decision và kết nối graph.

Kiến trúc:
    Input → Supervisor → [retrieval_worker | policy_tool_worker | human_review] → synthesis → Output

    Với policy_tool_worker route, pipeline luôn chạy retrieval trước để lấy context,
    sau đó mới chạy policy analysis.

Chạy thử:
    python graph.py

Author: Supervisor Owner (Nguyễn Thành Đạt - 2A202600203)
"""

import json
import os
import re
from datetime import datetime
from typing import TypedDict, Literal, Optional

# ─────────────────────────────────────────────
# 1. Shared State — dữ liệu đi xuyên toàn graph
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    task: str                           # Câu hỏi đầu vào từ user

    # Supervisor decisions
    route_reason: str                   # Lý do route sang worker nào
    risk_high: bool                     # True → cần HITL hoặc human_review
    needs_tool: bool                    # True → cần gọi external tool qua MCP
    hitl_triggered: bool                # True → đã pause cho human review

    # Worker outputs
    retrieved_chunks: list              # Output từ retrieval_worker
    retrieved_sources: list             # Danh sách nguồn tài liệu
    policy_result: dict                 # Output từ policy_tool_worker
    mcp_tools_used: list                # Danh sách MCP tools đã gọi

    # Final output
    final_answer: str                   # Câu trả lời tổng hợp
    sources: list                       # Sources được cite
    confidence: float                   # Mức độ tin cậy (0.0 - 1.0)

    # Trace & history
    history: list                       # Lịch sử các bước đã qua
    workers_called: list                # Danh sách workers đã được gọi
    supervisor_route: str               # Worker được chọn bởi supervisor
    latency_ms: Optional[int]           # Thời gian xử lý (ms)
    run_id: str                         # ID của run này


def make_initial_state(task: str) -> AgentState:
    """Khởi tạo state cho một run mới."""
    return {
        "task": task,
        "route_reason": "",
        "risk_high": False,
        "needs_tool": False,
        "hitl_triggered": False,
        "retrieved_chunks": [],
        "retrieved_sources": [],
        "policy_result": {},
        "mcp_tools_used": [],
        "final_answer": "",
        "sources": [],
        "confidence": 0.0,
        "history": [],
        "workers_called": [],
        "supervisor_route": "",
        "latency_ms": None,
        "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }


# ─────────────────────────────────────────────
# 2. Supervisor Node — quyết định route
# ─────────────────────────────────────────────

# --- Routing keyword configuration ---
# Mỗi nhóm keyword map tới 1 router và 1 category giải thích.
# Thứ tự ưu tiên: policy_exception > access_control > multi_hop > sla_ticket > hr_it > default

POLICY_EXCEPTION_KEYWORDS = [
    "flash sale", "license key", "license", "subscription",
    "kỹ thuật số", "digital product", "store credit",
    "ngoại lệ", "exception", "đã kích hoạt", "đã sử dụng",
    "hoàn tiền được không", "có được hoàn", "được hoàn tiền không",
    "được không",  # pattern: "hoàn tiền ... được không"
]

ACCESS_CONTROL_KEYWORDS = [
    "cấp quyền", "access level", "level 3", "level 2",
    "phê duyệt", "admin access", "elevated access",
    "access control", "permission",
]

REFUND_POLICY_KEYWORDS = [
    "hoàn tiền", "refund", "policy", "chính sách",
    "31/01", "30/01", "trước 01/02",  # temporal scoping dates
]

SLA_TICKET_KEYWORDS = [
    "p1", "p2", "sla", "ticket", "escalation", "escalate",
    "sự cố", "incident", "on-call", "pagerduty",
]

HR_KEYWORDS = [
    "remote", "nghỉ phép", "leave", "thử việc", "probation",
    "nhân viên", "annual leave", "sick leave",
]

IT_HELPDESK_KEYWORDS = [
    "đăng nhập", "mật khẩu", "password", "tài khoản bị khóa",
    "vpn", "wifi", "reset", "phần mềm", "cài đặt",
]

RISK_KEYWORDS = [
    "emergency", "khẩn cấp", "2am", "3am", "nửa đêm",
    "không rõ", "urgent", "critical",
]

# Regex pattern cho mã lỗi không rõ
UNKNOWN_ERROR_PATTERN = re.compile(r"err[-_]\d{3}", re.IGNORECASE)


def _detect_category(task_lower: str) -> tuple:
    """
    Phân tích task để xác định category và route.

    Returns:
        (route, route_reason, needs_tool, risk_high, is_multi_hop)
    """
    matched_signals = []
    route = "retrieval_worker"
    needs_tool = False
    risk_high = False
    is_multi_hop = False

    # --- Check từng nhóm keyword ---

    # 1. Policy exception detection (strongest signal for policy_tool)
    policy_ex_matches = [kw for kw in POLICY_EXCEPTION_KEYWORDS if kw in task_lower]
    if policy_ex_matches:
        matched_signals.append(f"policy_exception[{','.join(policy_ex_matches[:3])}]")
        route = "policy_tool_worker"
        needs_tool = True

    # 2. Access control detection
    access_matches = [kw for kw in ACCESS_CONTROL_KEYWORDS if kw in task_lower]
    if access_matches:
        matched_signals.append(f"access_control[{','.join(access_matches[:2])}]")
        route = "policy_tool_worker"
        needs_tool = True

    # 3. Refund policy detection
    # Key design decision: "hoàn tiền" alone is NOT enough to route to policy_tool.
    # Simple factual questions ("bao nhiêu ngày?") → retrieval_worker
    # Exception/decision questions ("Flash Sale được hoàn không?") → policy_tool_worker
    refund_matches = [kw for kw in REFUND_POLICY_KEYWORDS if kw in task_lower]
    if refund_matches:
        matched_signals.append(f"refund_policy[{','.join(refund_matches[:2])}]")
        if route != "policy_tool_worker":
            # Check if this is a policy DECISION question (not just factual)
            decision_signals = [
                "được không", "có được", "được hoàn", "ngoại lệ", "exception",
                "áp dụng", "điều kiện", "store credit", "giá trị",
            ]
            temporal_signals = ["31/01", "30/01", "trước 01/02"]

            if any(kw in task_lower for kw in decision_signals):
                route = "policy_tool_worker"
                needs_tool = True
            elif any(kw in task_lower for kw in temporal_signals):
                # Temporal scoping → policy
                route = "policy_tool_worker"
                needs_tool = True
            else:
                # Simple factual refund question → retrieval is sufficient
                # e.g. "hoàn tiền trong bao nhiêu ngày?" → just retrieve the fact
                route = "retrieval_worker"

    # 4. SLA/Ticket detection
    sla_matches = [kw for kw in SLA_TICKET_KEYWORDS if kw in task_lower]
    if sla_matches:
        matched_signals.append(f"sla_ticket[{','.join(sla_matches[:2])}]")
        # SLA alone → retrieval; SLA + access/policy → multi-hop (keep policy route)
        if route == "retrieval_worker":
            route = "retrieval_worker"

    # 5. HR keywords
    hr_matches = [kw for kw in HR_KEYWORDS if kw in task_lower]
    if hr_matches:
        matched_signals.append(f"hr_policy[{','.join(hr_matches[:2])}]")
        if route == "retrieval_worker":
            route = "retrieval_worker"

    # 6. IT helpdesk keywords
    it_matches = [kw for kw in IT_HELPDESK_KEYWORDS if kw in task_lower]
    if it_matches:
        matched_signals.append(f"it_helpdesk[{','.join(it_matches[:2])}]")
        if route == "retrieval_worker":
            route = "retrieval_worker"

    # 7. Multi-hop detection: query spans multiple document domains
    domain_count = sum([
        bool(sla_matches),
        bool(access_matches),
        bool(refund_matches),
        bool(hr_matches),
    ])
    if domain_count >= 2:
        is_multi_hop = True
        matched_signals.append(f"multi_hop[{domain_count}_domains]")
        # Multi-hop: prefer policy_tool since it handles cross-doc reasoning
        route = "policy_tool_worker"
        needs_tool = True

    # 8. Risk assessment
    risk_matches = [kw for kw in RISK_KEYWORDS if kw in task_lower]
    if risk_matches:
        risk_high = True
        matched_signals.append(f"risk_high[{','.join(risk_matches[:2])}]")

    # 9. Unknown error code → human_review
    if UNKNOWN_ERROR_PATTERN.search(task_lower):
        matched_signals.append("unknown_error_code")
        # Only route to human_review if there's genuinely no context to work with
        if not sla_matches and not refund_matches and not access_matches:
            route = "human_review"
            risk_high = True

    # 10. Default fallback
    if not matched_signals:
        matched_signals.append("no_keyword_match→default_retrieval")
        route = "retrieval_worker"

    # Build route_reason
    route_reason = f"route={route} | signals: {'; '.join(matched_signals)}"
    if is_multi_hop:
        route_reason += " | multi_hop=True"

    return route, route_reason, needs_tool, risk_high, is_multi_hop


def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor phân tích task và quyết định:
    1. Route sang worker nào (retrieval_worker / policy_tool_worker / human_review)
    2. Có cần MCP tool không (needs_tool)
    3. Có risk cao cần HITL không (risk_high)
    4. Có phải multi-hop query không

    Routing logic dựa vào keyword signal detection, không dùng LLM
    (nhanh, deterministic, dễ debug).
    """
    task = state["task"]
    task_lower = task.lower()
    state["history"].append(f"[supervisor] received task: {task[:100]}")

    # Detect route category
    route, route_reason, needs_tool, risk_high, is_multi_hop = _detect_category(task_lower)

    # Log quyết định MCP
    if needs_tool:
        route_reason += " | mcp_tools=available"
    else:
        route_reason += " | mcp_tools=not_needed"

    # Ghi vào state
    state["supervisor_route"] = route
    state["route_reason"] = route_reason
    state["needs_tool"] = needs_tool
    state["risk_high"] = risk_high
    state["history"].append(
        f"[supervisor] route={route} | needs_tool={needs_tool} | "
        f"risk_high={risk_high} | multi_hop={is_multi_hop}"
    )
    state["history"].append(f"[supervisor] route_reason: {route_reason}")

    return state


# ─────────────────────────────────────────────
# 3. Route Decision — conditional edge
# ─────────────────────────────────────────────

def route_decision(state: AgentState) -> Literal["retrieval_worker", "policy_tool_worker", "human_review"]:
    """
    Trả về tên worker tiếp theo dựa vào supervisor_route trong state.
    Đây là conditional edge của graph.
    """
    route = state.get("supervisor_route", "retrieval_worker")
    # Validate route to prevent invalid values
    valid_routes = {"retrieval_worker", "policy_tool_worker", "human_review"}
    if route not in valid_routes:
        state["history"].append(f"[route_decision] invalid route '{route}', fallback to retrieval_worker")
        return "retrieval_worker"
    return route  # type: ignore


# ─────────────────────────────────────────────
# 4. Human Review Node — HITL placeholder
# ─────────────────────────────────────────────

def human_review_node(state: AgentState) -> AgentState:
    """
    HITL node: pause và chờ human approval.
    Trong lab này, implement dưới dạng placeholder (in ra warning).
    """
    state["hitl_triggered"] = True
    state["history"].append("[human_review] HITL triggered — awaiting human input")
    state["workers_called"].append("human_review")

    # Placeholder: tự động approve để pipeline tiếp tục
    print(f"\n⚠️  HITL TRIGGERED")
    print(f"   Task: {state['task']}")
    print(f"   Reason: {state['route_reason']}")
    print(f"   Action: Auto-approving in lab mode (set hitl_triggered=True)\n")

    # Sau khi human approve, route về retrieval để lấy evidence
    state["supervisor_route"] = "retrieval_worker"
    state["route_reason"] += " | human approved → retrieval"
    state["history"].append("[human_review] auto-approved → continuing to retrieval_worker")

    return state


# ─────────────────────────────────────────────
# 5. Worker Nodes — gọi workers thật hoặc fallback placeholder
# ─────────────────────────────────────────────

def retrieval_worker_node(state: AgentState) -> AgentState:
    """Wrapper gọi retrieval worker. Thử import worker thật, fallback placeholder."""
    try:
        from workers.retrieval import run as retrieval_run
        state = retrieval_run(state)
        return state
    except Exception as e:
        # Fallback: placeholder output để test graph flow
        state["workers_called"].append("retrieval_worker")
        state["history"].append(f"[retrieval_worker] fallback mode (import error: {e})")
        state["retrieved_chunks"] = [
            {"text": "SLA P1: phản hồi 15 phút, xử lý 4 giờ. Escalation tự động sau 10 phút.",
             "source": "sla_p1_2026.txt", "score": 0.92}
        ]
        state["retrieved_sources"] = ["sla_p1_2026.txt"]
        state["history"].append(f"[retrieval_worker] retrieved {len(state['retrieved_chunks'])} chunks (placeholder)")
        return state


def policy_tool_worker_node(state: AgentState) -> AgentState:
    """Wrapper gọi policy/tool worker. Thử import worker thật, fallback placeholder."""
    try:
        from workers.policy_tool import run as policy_tool_run
        state = policy_tool_run(state)
        return state
    except Exception as e:
        # Fallback: placeholder
        state["workers_called"].append("policy_tool_worker")
        state["history"].append(f"[policy_tool_worker] fallback mode (import error: {e})")
        state["policy_result"] = {
            "policy_applies": True,
            "policy_name": "refund_policy_v4",
            "exceptions_found": [],
            "source": "policy_refund_v4.txt",
        }
        state["history"].append("[policy_tool_worker] policy check complete (placeholder)")
        return state


def synthesis_worker_node(state: AgentState) -> AgentState:
    """Wrapper gọi synthesis worker. Thử import worker thật, fallback placeholder."""
    try:
        from workers.synthesis import run as synthesis_run
        state = synthesis_run(state)
        return state
    except Exception as e:
        # Fallback: placeholder
        state["workers_called"].append("synthesis_worker")
        state["history"].append(f"[synthesis_worker] fallback mode (import error: {e})")
        chunks = state.get("retrieved_chunks", [])
        sources = state.get("retrieved_sources", [])
        state["final_answer"] = f"[PLACEHOLDER] Câu trả lời tổng hợp từ {len(chunks)} chunks."
        state["sources"] = sources
        state["confidence"] = 0.75
        state["history"].append(f"[synthesis_worker] answer generated, confidence={state['confidence']} (placeholder)")
        return state


# ─────────────────────────────────────────────
# 6. Build Graph — orchestration logic
# ─────────────────────────────────────────────

def build_graph():
    """
    Xây dựng graph với supervisor-worker pattern.

    Pipeline flow:
        1. Supervisor → quyết định route
        2. Route tới worker(s) phù hợp:
           - retrieval_worker: lấy evidence → synthesis
           - policy_tool_worker: lấy evidence (retrieval) → phân tích policy → synthesis
           - human_review: HITL → lấy evidence (retrieval) → synthesis
        3. Synthesis → tổng hợp câu trả lời cuối cùng

    Key design decision: Policy route LUÔN chạy retrieval trước
    để đảm bảo policy worker có context chunks để phân tích.
    """

    def run(state: AgentState) -> AgentState:
        import time
        start = time.time()

        # ── Step 1: Supervisor decides route ──
        state = supervisor_node(state)

        # ── Step 2: Route to appropriate worker(s) ──
        route = route_decision(state)

        if route == "human_review":
            # HITL → auto-approve → retrieval → synthesis
            state = human_review_node(state)
            state = retrieval_worker_node(state)

        elif route == "policy_tool_worker":
            # Policy questions: ALWAYS retrieve evidence first,
            # then run policy analysis on the retrieved context.
            # This ensures policy worker has chunks to analyze.
            state = retrieval_worker_node(state)
            state = policy_tool_worker_node(state)

        else:
            # Default: retrieval_worker → synthesis
            state = retrieval_worker_node(state)

        # ── Step 3: Always synthesize ──
        state = synthesis_worker_node(state)

        # ── Record latency ──
        state["latency_ms"] = int((time.time() - start) * 1000)
        state["history"].append(f"[graph] completed in {state['latency_ms']}ms")
        return state

    return run


# ─────────────────────────────────────────────
# 7. Public API
# ─────────────────────────────────────────────

_graph = build_graph()


def run_graph(task: str) -> AgentState:
    """
    Entry point: nhận câu hỏi, trả về AgentState với full trace.

    Args:
        task: Câu hỏi từ user

    Returns:
        AgentState với final_answer, trace, routing info, v.v.
    """
    state = make_initial_state(task)
    result = _graph(state)
    return result


def save_trace(state: AgentState, output_dir: str = "./artifacts/traces") -> str:
    """Lưu trace ra file JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{state['run_id']}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return filename


# ─────────────────────────────────────────────
# 8. Manual Test — Sprint 1 validation
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Day 09 Lab -- Supervisor-Worker Graph (Sprint 1)")
    print("  Author: Supervisor Owner (Thanh vien 1)")
    print("=" * 70)

    # Test queries covering all routing categories
    test_queries = [
        # 1. SLA/Ticket -> retrieval_worker
        "SLA xu ly ticket P1 la bao lau?",
        # 2. Refund policy -> policy_tool_worker
        "Khach hang Flash Sale yeu cau hoan tien vi san pham loi -- duoc khong?",
        # 3. Access control + emergency -> policy_tool_worker (multi-hop)
        "Can cap quyen Level 3 de khac phuc P1 khan cap. Quy trinh la gi?",
        # 4. IT Helpdesk -> retrieval_worker
        "Tai khoan bi khoa sau bao nhieu lan dang nhap sai?",
        # 5. HR -> retrieval_worker
        "Nhan vien thu viec muon lam remote -- dieu kien la gi?",
        # 6. Unknown error -> human_review
        "ERR-403-AUTH la loi gi va cach xu ly?",
        # 7. Multi-hop (SLA + access control) -> policy_tool_worker
        "Ticket P1 luc 2am. Can cap Level 2 access tam thoi cho contractor. Neu ca hai quy trinh.",
        # 8. Temporal policy scoping -> policy_tool_worker
        "Don hang 31/01/2026, khach yeu cau hoan tien 07/02/2026. Duoc khong?",
    ]

    # Verify routing results
    route_summary = {}
    all_passed = True

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-' * 60}")
        print(f">> [{i}] Query: {query}")
        result = run_graph(query)

        route = result["supervisor_route"]
        route_summary[route] = route_summary.get(route, 0) + 1

        print(f"  Route   : {route}")
        print(f"  Reason  : {result['route_reason']}")
        print(f"  Workers : {result['workers_called']}")
        print(f"  Risk    : {'!! HIGH' if result['risk_high'] else 'OK normal'}")
        print(f"  HITL    : {'!! TRIGGERED' if result['hitl_triggered'] else 'OK no'}")
        print(f"  Answer  : {result['final_answer'][:120]}...")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Latency : {result['latency_ms']}ms")

        # Validate route_reason is not empty or "unknown"
        if not result["route_reason"] or result["route_reason"] == "unknown":
            print(f"  FAIL: route_reason is empty or 'unknown'!")
            all_passed = False

        # Save trace
        trace_file = save_trace(result)
        print(f"  Trace   -> {trace_file}")

    # Summary
    print(f"\n{'=' * 70}")
    print("Routing Summary:")
    for route, count in sorted(route_summary.items()):
        print(f"  {route}: {count} queries ({count * 100 // len(test_queries)}%)")

    print(f"\nSprint 1 Checklist:")
    print(f"  [{'x' if all_passed else ' '}] python graph.py runs without errors")
    print(f"  [{'x' if len(route_summary) >= 2 else ' '}] Supervisor routes >= 2 task types ({len(route_summary)} types)")
    print(f"  [{'x' if all_passed else ' '}] route_reason is clear (not 'unknown')")
    print(f"  [x] State has: task, route_reason, history, risk_high")

    if all_passed and len(route_summary) >= 2:
        print(f"\n>>> Sprint 1 PASSED! All criteria met.")
    else:
        print(f"\n>>> Sprint 1 needs attention.")
