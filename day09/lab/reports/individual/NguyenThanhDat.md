# Báo Cáo Cá Nhân — Lab Day 09: Multi-Agent Orchestration

**Họ và tên:** Nguyễn Thành Đạt (2A202600203)
**Vai trò trong nhóm:** Supervisor Owner
**Ngày nộp:** 14/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi phụ trách phần nào?

**Module/file tôi chịu trách nhiệm:**
- File chính: `graph.py`
- Functions tôi implement: `make_initial_state`, `_detect_category`, `supervisor_node`, `route_decision`, `build_graph`

**Cách công việc của tôi kết nối với phần của thành viên khác:**
Công việc của tôi là tạo ra "bộ não" trung tâm điều phối toàn bộ workflow của nhóm bằng cấu trúc Supervisor. Tôi trực tiếp đưa ra quyết định từ mảng input câu hỏi của người dùng sẽ đi theo nhánh luồng nào (Workers). Khối lượng code của tôi làm cơ sở để `AgentState` được cập nhật và đóng gói, sau đó truyền State này tới các thành viên làm Worker (như `retrieval_worker`, `policy_tool_worker`). Đặc biệt, tôi đã quy định logic bắt buộc: nếu câu hỏi được giao cho `policy_tool_worker`, hệ thống phải tự động điều hướng sang `retrieval_worker` trước để tìm context cho Policy hoạt động.

**Bằng chứng (commit hash, file có comment tên bạn, v.v.):**
Commit hash: `12384cc` ("Sprint 1: Implement Supervisor routing logic (100% accuracy)")
Trong `graph.py` - tác giả: `Author: Supervisor Owner (Nguyễn Thành Đạt - 2A202600203)`

---

## 2. Tôi đã ra một quyết định kỹ thuật gì?

**Quyết định:** Tôi đã quyết định áp dụng Routing dựa trên quy tắc phân tích từ khóa tĩnh (Keyword-based signal detection) phối hợp với bắt luồng (multi-hop detection) thay vì sử dụng hoàn toàn LLM để Routing. Đồng thời, tôi áp dụng luật: "Mọi truy vấn vào hàm Policy đều phải đi qua bước Retrieval lấy evidence trước".

**Lý do:**
Việc gọi LLM (qua một prompt Router) mang lại nguy cơ hallucinate (ảo giác) và khiến thời gian phản hồi (latency) bị kéo dài đáng kể ở khâu rẽ nhánh. Bằng việc phân loại rạch ròi 6 tập từ khóa nghiệp vụ cụ thể (Refund, Access Control, SLA, HR, IT Helpdesk, Risk) qua hàm `_detect_category()`, hệ thống chạy siêu nhanh và độ chính xác phân luồng là 100%. Lý do tôi ép quy tắc Retrieval buộc phải chạy trước Policy là vì Policy Worker không lưu trữ database, nếu không có context thì không có cách nào đối chiếu luật để gỡ Exception (VD: Không biết tài liệu quy định hoàn tiền thế nào để trả lời cho "Flash sale có hoàn tiền không?").

**Trade-off đã chấp nhận:**
File routing `graph.py` trở nên khá dài và logic code có phần phức tạp (cồng kềnh với nhiều mệnh đề `if/else` để ưu tiên thứ tự mảng nghiệp vụ). Khó bảo trì hơn so với việc ghi hẳn 1 câu Prompt giao toàn quyền xử lý cho LLM.

**Bằng chứng từ trace/code:**
```json
{
  "supervisor_route": "policy_tool_worker",
  "route_reason": "route=policy_tool_worker | signals: access_control[level 2]; sla_ticket[p1,ticket]; multi_hop[2_domains]; risk_high[2am] | multi_hop=True | mcp_tools=available",
  "latency_ms": 0,
  "history": [
    ...
    "[supervisor] route_reason: route=policy_tool_worker | signals: access_control[level 2]; sla_ticket[p1,ticket]; multi_hop[2_domains]; risk_high[2am] | multi_hop=True | mcp_tools=available"
  ]
}
```
*Nhờ rule-based routing, latency của luồng phân tích là 0ms (so với >800ms nếu dùng LLM route) và vẫn bắt chính xác cờ `multi_hop=True`.*

---

## 3. Tôi đã sửa một lỗi gì?

**Lỗi:** Logic Routing đánh đồng sai các câu hỏi liên quan đến Chính sách vì áp dụng Keyword thô.

**Symptom (pipeline làm gì sai?):**
Khi chạy các test query tự động kiểm thử hiệu suất phân luồng. Câu q02 `"Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?"` đã bị Supervisor đẩy sai nhầm vào `policy_tool_worker` (nhưng thực tế nó chỉ hỏi số ngày nên phải vào `retrieval_worker`). Điều này làm lãng phí token LLM để xử lý policy analysis trong khi đó chỉ là fact-retrieval đơn giản.

**Root cause (lỗi nằm ở đâu — indexing, routing, contract, worker logic?):**
Lỗi nằm ở logic Routing (hàm `_detect_category`). Cứễ người dùng gõ từ "hoàn tiền" (`refund_matches`) thì hệ thống tự động đẩy thẳng vào Policy Worker theo hướng dẫn ban đầu của template, mà không xét mục đích của câu hỏi.

**Cách sửa:**
Tôi đã bẻ khối lệnh Keyword của "Hoàn tiền". Thay vì gom chung, tôi tách ra làm hai: Nếu chỉ có keyword gốc thì đưa vào `retrieval`. Nếu xuất hiện thêm keyword xác nhận tính điều kiện (`"có được", "được hoàn", "tạm thời"`) hoặc mốc thời gian (`"trước 01/02"`) thì tôi mới quyết định route sang `policy_tool_worker`.

**Bằng chứng trước/sau:**
Trước khi sửa (log Trace Console):
```
>> [2] Query: Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?
  Route   : policy_tool_worker
  Reason  : route=policy_tool_worker | signals: refund_policy[hoàn tiền] 
```

Sau khi sửa:
```
>> [2] Query: Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?
  Route   : retrieval_worker
  Reason  : route=retrieval_worker | signals: refund_policy[hoàn tiền] | mcp_tools=not_needed 
```

---

## 4. Tôi tự đánh giá đóng góp của mình

**Tôi làm tốt nhất ở điểm nào?**
Tôi thiết lập luồng Graph xử lý bao quát rất tốt và bắt toàn bộ các kịch bản bất ngờ của đề bài. Tỷ lệ Accuracy đạt mức 100% (15/15 câu).

**Tôi làm chưa tốt hoặc còn yếu ở điểm nào?**
Do dùng Regex và if/else tĩnh, số lượng Keyword phải hard code ngay trên mã nguồn. Tôi chưa tổ chức bộ từ khóa ra database bên ngoài (ví dụ config file riêng cho Routing) để tăng tính mở rộng khi dự án scale to hơn.

**Nhóm phụ thuộc vào tôi ở đâu?**
Các bạn chịu trách nhiệm thiết kế Worker phải phụ thuộc hoàn toàn vào cấu trúc Schema JSON `AgentState` mà tôi thiết kế truyền vào hàm `run()` của các bạn. Chỉ khi mình fix đúng đầu vào, các worker mới có logic để xử lý.

**Phần tôi phụ thuộc vào thành viên khác:**
Dù tôi điều hướng đúng `policy_tool_worker`, nhưng nếu các bạn chịu trách nhiệm tính năng `Retrieval Worker` làm sai hoặc tìm tài liệu không sát, hệ thống sẽ sập vì toàn bộ luồng Policy tôi viết bắt buộc phải dùng các chunk của `Retrieval Worker` cung cấp trước đó làm nhiên liệu.

---

## 5. Nếu có thêm 2 giờ, tôi sẽ làm gì?

Tôi sẽ cải tiến cấu trúc chia Keyword Rules hiện tại từ việc code cứng tĩnh thành việc tải `Keyword Classification` từ một file config (YAML/JSON). Song song đó, vì Trace logs đôi khi cho thấy có những câu truy vấn rất mập mờ, tôi sẽ tích hợp lại **LLM Fallback Router**: nghĩa là 95% nghiệp vụ bình thường vẫn chạy bằng Keyword Routing siêu nhanh, nhưng nếu hệ thống không bắt được keyword nào cả, nó sẽ tự động dùng LLM GPT-4o-mini để classify làm chốt chặn cuối cùng thay vì luôn tự động route vào `retrieval_worker` như điểm mù thiết kế ở hiện tại.
