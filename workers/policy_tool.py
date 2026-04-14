"""
workers/policy_tool.py — Policy & Tool Worker
Sprint 2+3: Kiểm tra policy dựa vào context, gọi MCP tools khi cần.

Input (từ AgentState):
    - task: câu hỏi
    - retrieved_chunks: context từ retrieval_worker
    - needs_tool: True nếu supervisor quyết định cần tool call

Output (vào AgentState):
    - policy_result: {"policy_applies", "policy_name", "exceptions_found", "source", "rule"}
    - mcp_tools_used: list of tool calls đã thực hiện
    - worker_io_log: log

Gọi độc lập để test:
    python workers/policy_tool.py
"""

import json
import os
import re
import sys
from datetime import datetime
from typing import Optional
from urllib import error, request

WORKER_NAME = "policy_tool_worker"
MCP_SERVER_MODE = os.getenv("MCP_SERVER_MODE", "mock").strip().lower()
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080").strip()


# ─────────────────────────────────────────────
# MCP Client — Sprint 3: Thay bằng real MCP call
# ─────────────────────────────────────────────

def _call_mcp_tool(tool_name: str, tool_input: dict) -> dict:
    """
    Gọi MCP tool.

    Sprint 3 TODO: Implement bằng cách import mcp_server hoặc gọi HTTP.

    Hiện tại: Import trực tiếp từ mcp_server.py (trong-process mock).
    """
    try:
        result = None

        if MCP_SERVER_MODE == "http" and MCP_SERVER_URL:
            result = _call_mcp_tool_http(tool_name, tool_input)

        if result is None:
            from mcp_server import dispatch_tool
            result = dispatch_tool(tool_name, tool_input)

        return {
            "tool": tool_name,
            "input": tool_input,
            "output": result,
            "error": None,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "tool": tool_name,
            "input": tool_input,
            "output": None,
            "error": {"code": "MCP_CALL_FAILED", "reason": str(e)},
            "timestamp": datetime.now().isoformat(),
        }


def _call_mcp_tool_http(tool_name: str, tool_input: dict) -> Optional[dict]:
    """
    Gọi MCP tool qua HTTP nếu server được cấu hình.

    Thử một vài endpoint phổ biến để phù hợp với mock server hoặc HTTP wrapper.
    Nếu request thất bại thì trả về None để fallback sang in-process dispatch.
    """
    payload = {
        "tool": tool_name,
        "input": tool_input,
    }

    base_url = MCP_SERVER_URL.rstrip("/")
    candidate_paths = ["/tools/call", "/dispatch", "/mcp/call", "/"]

    for path in candidate_paths:
        url = base_url if path == "/" else f"{base_url}{path}"
        try:
            req = request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=5) as resp:
                raw_body = resp.read().decode("utf-8").strip()
                if not raw_body:
                    return {}
                parsed = json.loads(raw_body)
                if isinstance(parsed, dict):
                    if "output" in parsed and isinstance(parsed["output"], dict):
                        return parsed["output"]
                    if "result" in parsed and isinstance(parsed["result"], dict):
                        return parsed["result"]
                    if "data" in parsed and isinstance(parsed["data"], dict):
                        return parsed["data"]
                    return parsed
                return {"result": parsed}
        except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            continue

    return None


# ─────────────────────────────────────────────
# Policy Analysis Logic
# ─────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _chunk_text(chunks: list) -> str:
    return " ".join(c.get("text", "") for c in chunks if isinstance(c, dict))


def _chunk_sources(chunks: list) -> list:
    return sorted({c.get("source", "unknown") for c in chunks if isinstance(c, dict) and c.get("source")})


def _extract_access_level(text: str) -> Optional[int]:
    match = re.search(r"level\s*([1-4])", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if "admin access" in text.lower():
        return 3
    return None


def _extract_requester_role(text: str) -> str:
    lowered = text.lower()
    if "contractor" in lowered:
        return "contractor"
    if "vendor" in lowered or "third-party" in lowered:
        return "vendor"
    if "team lead" in lowered:
        return "team lead"
    if "senior engineer" in lowered:
        return "senior engineer"
    if "manager" in lowered:
        return "manager"
    return "employee"


def _detect_temporal_scope(task_text: str) -> Optional[str]:
    lowered = task_text.lower()
    if any(marker in lowered for marker in ["31/01/2026", "30/01/2026", "trước 01/02/2026", "trước ngày 01/02/2026"]):
        return "Đơn hàng đặt trước 01/02/2026 áp dụng chính sách v3, không phải v4."
    return None


def analyze_policy(task: str, chunks: list, mcp_context: Optional[dict] = None) -> dict:
    """
    Phân tích policy dựa trên context chunks.

    TODO Sprint 2: Implement logic này với LLM call hoặc rule-based check.

    Cần xử lý các exceptions:
    - Flash Sale → không được hoàn tiền
    - Digital product / license key / subscription → không được hoàn tiền
    - Sản phẩm đã kích hoạt → không được hoàn tiền
    - Đơn hàng trước 01/02/2026 → áp dụng policy v3 (không có trong docs)

    Returns:
        dict with: policy_applies, policy_name, exceptions_found, source, rule, explanation
    """
    task_text = _normalize_text(task)
    context_text = _normalize_text(_chunk_text(chunks))
    combined_text = f"{task_text} {context_text}".strip()
    sources = _chunk_sources(chunks)
    if not sources:
        sources = []

    # --- Rule-based exception detection ---
    exceptions_found = []
    policy_name = "generic_policy"
    policy_applies = True
    policy_version_note = ""
    extra_result = {}

    # Refund policy
    refund_keywords = ["hoàn tiền", "refund", "store credit", "credit nội bộ", "refund request"]
    access_keywords = ["cấp quyền", "access", "level", "emergency", "admin access"]
    hr_keywords = ["remote", "probation", "nghỉ phép", "leave", "sick leave", "annual leave"]

    is_refund_task = any(keyword in combined_text for keyword in refund_keywords)
    is_access_task = any(keyword in combined_text for keyword in access_keywords)
    is_hr_task = any(keyword in combined_text for keyword in hr_keywords)

    if is_refund_task:
        policy_name = "refund_policy_v4"
        sources = sorted(set(sources + ["policy_refund_v4.txt"]))

        if any(keyword in combined_text for keyword in ["flash sale", "chương trình khuyến mãi flash sale"]):
            exceptions_found.append({
                "type": "flash_sale_exception",
                "rule": "Đơn hàng Flash Sale không được hoàn tiền (Điều 3, chính sách v4).",
                "source": "policy_refund_v4.txt",
            })

        if any(keyword in combined_text for keyword in ["license key", "license", "subscription", "kỹ thuật số"]):
            exceptions_found.append({
                "type": "digital_product_exception",
                "rule": "Sản phẩm kỹ thuật số (license key, subscription) không được hoàn tiền (Điều 3).",
                "source": "policy_refund_v4.txt",
            })

        if any(keyword in combined_text for keyword in ["đã kích hoạt", "đã đăng ký", "đã sử dụng", "activated"]):
            exceptions_found.append({
                "type": "activated_exception",
                "rule": "Sản phẩm đã được kích hoạt hoặc đăng ký tài khoản không được hoàn tiền (Điều 3).",
                "source": "policy_refund_v4.txt",
            })

        temporal_note = _detect_temporal_scope(task_text)
        if temporal_note:
            policy_version_note = temporal_note
            exceptions_found.append({
                "type": "temporal_scoping_exception",
                "rule": temporal_note,
                "source": "policy_refund_v4.txt",
            })

        policy_applies = len(exceptions_found) == 0
        if policy_applies:
            extra_result["rule"] = "Điều 2: Sản phẩm lỗi do nhà sản xuất, yêu cầu trong 7 ngày làm việc, chưa sử dụng/chưa mở seal."
        else:
            extra_result["rule"] = exceptions_found[0]["rule"]

    elif is_access_task:
        policy_name = "access_control_sop"
        sources = sorted(set(sources + ["access_control_sop.txt"]))
        access_level = _extract_access_level(combined_text)
        requester_role = _extract_requester_role(combined_text)
        is_emergency = any(keyword in combined_text for keyword in ["emergency", "khẩn cấp", "p1", "incident"])

        if access_level is None:
            policy_applies = False
            policy_version_note = "Không xác định được access level từ câu hỏi."
            exceptions_found.append({
                "type": "missing_access_level",
                "rule": "Cần xác định access level trước khi kết luận policy.",
                "source": "access_control_sop.txt",
            })
        else:
            mcp_decision = mcp_context or {}
            mcp_output = mcp_decision.get("output") if isinstance(mcp_decision, dict) else None
            if not isinstance(mcp_output, dict):
                mcp_output = None

            if mcp_output and "can_grant" in mcp_output:
                extra_result["access_decision"] = mcp_output
                policy_applies = bool(mcp_output.get("can_grant", False))
                if not policy_applies:
                    exceptions_found.append({
                        "type": "access_request_denied",
                        "rule": f"Level {access_level} không thỏa điều kiện cấp quyền theo SOP.",
                        "source": "access_control_sop.txt",
                    })
            else:
                if access_level == 1:
                    extra_result["access_decision"] = {
                        "access_level": 1,
                        "can_grant": True,
                        "required_approvers": ["Line Manager"],
                        "emergency_override": False,
                        "source": "access_control_sop.txt",
                    }
                elif access_level == 2:
                    extra_result["access_decision"] = {
                        "access_level": 2,
                        "can_grant": True,
                        "required_approvers": ["Line Manager", "IT Admin"],
                        "emergency_override": is_emergency,
                        "source": "access_control_sop.txt",
                    }
                elif access_level == 3:
                    extra_result["access_decision"] = {
                        "access_level": 3,
                        "can_grant": True,
                        "required_approvers": ["Line Manager", "IT Admin", "IT Security"],
                        "emergency_override": False,
                        "source": "access_control_sop.txt",
                    }
                else:
                    extra_result["access_decision"] = {
                        "access_level": 4,
                        "can_grant": True,
                        "required_approvers": ["IT Manager", "CISO"],
                        "emergency_override": False,
                        "source": "access_control_sop.txt",
                    }

                policy_applies = bool(extra_result["access_decision"].get("can_grant", False))

            if is_emergency and access_level == 3:
                exceptions_found.append({
                    "type": "no_emergency_bypass",
                    "rule": "Level 3 không có emergency bypass; phải follow quy trình chuẩn.",
                    "source": "access_control_sop.txt",
                })
                policy_applies = False
                if "access_decision" in extra_result:
                    extra_result["access_decision"]["can_grant"] = False
                    extra_result["access_decision"]["emergency_override"] = False

            extra_result["requester_role"] = requester_role
            extra_result["is_emergency"] = is_emergency
            extra_result["access_level"] = access_level

    elif is_hr_task:
        policy_name = "hr_leave_policy"
        sources = sorted(set(sources + ["hr_leave_policy.txt"]))
        if any(keyword in combined_text for keyword in ["remote", "làm remote"]):
            if "probation" in combined_text or "thử việc" in combined_text:
                if any(keyword in combined_text for keyword in ["không", "not", "no"]):
                    policy_applies = True
                    extra_result["rule"] = "Nhân viên trong probation period không được làm remote."
                else:
                    policy_applies = False
                    exceptions_found.append({
                        "type": "probation_remote_restriction",
                        "rule": "Nhân viên trong probation period không được phép làm remote.",
                        "source": "hr_leave_policy.txt",
                    })
                    extra_result["rule"] = "Chỉ nhân viên đã qua probation period mới được remote tối đa 2 ngày/tuần với phê duyệt của Team Lead."
            else:
                extra_result["rule"] = "Nhân viên sau probation period có thể làm remote tối đa 2 ngày/tuần với phê duyệt của Team Lead."
        else:
            extra_result["rule"] = "Policy HR được áp dụng theo tài liệu nội bộ phù hợp với câu hỏi."
        policy_applies = len(exceptions_found) == 0

    else:
        policy_name = "retrieval_or_helpdesk_policy"
        policy_applies = len(exceptions_found) == 0

    if policy_version_note and not is_refund_task:
        exceptions_found.append({
            "type": "policy_version_note",
            "rule": policy_version_note,
            "source": "policy_refund_v4.txt",
        })

    result = {
        "policy_applies": policy_applies,
        "policy_name": policy_name,
        "exceptions_found": exceptions_found,
        "source": sources,
        "policy_version_note": policy_version_note,
        "explanation": "Analyzed via rule-based policy check with MCP enrichment when available.",
    }

    result.update(extra_result)

    if not result.get("rule") and exceptions_found:
        result["rule"] = exceptions_found[0].get("rule", "")

    return result


# ─────────────────────────────────────────────
# Worker Entry Point
# ─────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    Worker entry point — gọi từ graph.py.

    Args:
        state: AgentState dict

    Returns:
        Updated AgentState với policy_result và mcp_tools_used
    """
    task = state.get("task", "")
    chunks = state.get("retrieved_chunks", [])
    needs_tool = state.get("needs_tool", False)
    task_text = _normalize_text(task)

    state.setdefault("workers_called", [])
    state.setdefault("history", [])
    state.setdefault("mcp_tools_used", [])

    state["workers_called"].append(WORKER_NAME)

    worker_io = {
        "worker": WORKER_NAME,
        "input": {
            "task": task,
            "chunks_count": len(chunks),
            "needs_tool": needs_tool,
        },
        "output": None,
        "error": None,
    }

    try:
        # Step 1: Nếu chưa có chunks, gọi MCP search_kb
        if not chunks and needs_tool:
            mcp_result = _call_mcp_tool("search_kb", {"query": task, "top_k": 3})
            state["mcp_tools_used"].append(mcp_result)
            state["history"].append(f"[{WORKER_NAME}] called MCP search_kb")

            if mcp_result.get("output") and mcp_result["output"].get("chunks"):
                chunks = mcp_result["output"]["chunks"]
                state["retrieved_chunks"] = chunks

        # Step 1b: Đối với access requests, gọi MCP permission checker khi có thể.
        access_mcp_result = None
        if needs_tool and any(keyword in task_text for keyword in ["access", "cấp quyền", "emergency", "level", "contractor"]):
            access_level = _extract_access_level(task_text)
            if access_level is not None:
                access_mcp_result = _call_mcp_tool(
                    "check_access_permission",
                    {
                        "access_level": access_level,
                        "requester_role": _extract_requester_role(task_text),
                        "is_emergency": any(keyword in task_text for keyword in ["emergency", "khẩn cấp", "p1", "incident"]),
                    },
                )
                state["mcp_tools_used"].append(access_mcp_result)
                state["history"].append(f"[{WORKER_NAME}] called MCP check_access_permission")

        # Step 2: Phân tích policy
        policy_result = analyze_policy(task, chunks, access_mcp_result)
        state["policy_result"] = policy_result

        # Step 3: Nếu cần thêm info từ MCP (e.g., ticket status), gọi get_ticket_info
        if needs_tool and any(kw in task_text for kw in ["ticket", "p1", "jira"]):
            mcp_result = _call_mcp_tool("get_ticket_info", {"ticket_id": "P1-LATEST"})
            state["mcp_tools_used"].append(mcp_result)
            state["history"].append(f"[{WORKER_NAME}] called MCP get_ticket_info")

        worker_io["output"] = {
            "policy_applies": policy_result["policy_applies"],
            "exceptions_count": len(policy_result.get("exceptions_found", [])),
            "mcp_calls": len(state["mcp_tools_used"]),
        }
        state["history"].append(
            f"[{WORKER_NAME}] policy_applies={policy_result['policy_applies']}, "
            f"exceptions={len(policy_result.get('exceptions_found', []))}"
        )

    except Exception as e:
        worker_io["error"] = {"code": "POLICY_CHECK_FAILED", "reason": str(e)}
        state["policy_result"] = {"error": str(e)}
        state["history"].append(f"[{WORKER_NAME}] ERROR: {e}")

    state.setdefault("worker_io_logs", []).append(worker_io)
    return state


# ─────────────────────────────────────────────
# Test độc lập
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Policy Tool Worker — Standalone Test")
    print("=" * 50)

    test_cases = [
        {
            "task": "Khách hàng Flash Sale yêu cầu hoàn tiền vì sản phẩm lỗi — được không?",
            "retrieved_chunks": [
                {"text": "Ngoại lệ: Đơn hàng Flash Sale không được hoàn tiền.", "source": "policy_refund_v4.txt", "score": 0.9}
            ],
        },
        {
            "task": "Khách hàng muốn hoàn tiền license key đã kích hoạt.",
            "retrieved_chunks": [
                {"text": "Sản phẩm kỹ thuật số (license key, subscription) không được hoàn tiền.", "source": "policy_refund_v4.txt", "score": 0.88}
            ],
        },
        {
            "task": "Khách hàng yêu cầu hoàn tiền trong 5 ngày, sản phẩm lỗi, chưa kích hoạt.",
            "retrieved_chunks": [
                {"text": "Yêu cầu trong 7 ngày làm việc, sản phẩm lỗi nhà sản xuất, chưa dùng.", "source": "policy_refund_v4.txt", "score": 0.85}
            ],
        },
        {
            "task": "Contractor cần Level 3 access khẩn cấp để fix incident P1.",
            "retrieved_chunks": [
                {"text": "Level 3 cần Line Manager, IT Admin, IT Security. Không có emergency bypass.", "source": "access_control_sop.txt", "score": 0.91}
            ],
            "needs_tool": True,
        },
        {
            "task": "Nhân viên trong probation period muốn làm remote.",
            "retrieved_chunks": [
                {"text": "Chỉ nhân viên sau probation period mới được remote tối đa 2 ngày/tuần với Team Lead phê duyệt.", "source": "hr_leave_policy.txt", "score": 0.87}
            ],
        },
    ]

    for tc in test_cases:
        print(f"\n▶ Task: {tc['task'][:70]}...")
        result = run(tc.copy())
        pr = result.get("policy_result", {})
        print(f"  policy_applies: {pr.get('policy_applies')}")
        print(f"  policy_name: {pr.get('policy_name')}")
        if pr.get("exceptions_found"):
            for ex in pr["exceptions_found"]:
                print(f"  exception: {ex['type']} — {ex['rule'][:60]}...")
        if pr.get("policy_version_note"):
            print(f"  note: {pr.get('policy_version_note')}")
        print(f"  MCP calls: {len(result.get('mcp_tools_used', []))}")

    print("\n✅ policy_tool_worker test done.")
