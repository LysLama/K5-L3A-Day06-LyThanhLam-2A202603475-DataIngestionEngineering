# Báo cáo cá nhân — Day 6: Data Ingestion Engineering

- **Họ và tên:** Lý Thanh Lâm
- **MSSV:** `2A202602475` · **Lớp:** `L3A`

## 1. Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:**
  - Bộ tài liệu gồm 2 giáo trình lịch sử thế giới mở của **OpenStax (Rice University, ©2026)**, giấy phép **CC BY-NC-SA 4.0**, lưu tại `starter/data/` và khai báo đầy đủ trong [data/manifest.json](../../data/manifest.json):
    1. `wh_vol1` — *World History, Volume 1: to 1500*, **Chapter 3: Early Civilizations and Urban Societies**, chọn **38 trang nội dung** (trang PDF vật lý `79–116`; tổng file 791 trang).
    2. `wh_vol2` — *World History, Volume 2: from 1400*, **Chapter 2: Exchange in East Asia and the Indian Ocean**, chọn **37 trang nội dung** (trang PDF vật lý `43–79`; tổng file 723 trang).
  - **Tổng cộng:** **75 trang nội dung hợp lệ** (`205.657` ký tự sau chuẩn hóa: `96.795` ký tự từ `wh_vol1` và `108.862` ký tự từ `wh_vol2`).
  - **Quy tắc tính trang:** Chỉ tính trang PDF vật lý chứa nội dung học thuật chính (văn bản chính, chú thích hình, hộp tư liệu gốc). Loại bỏ bìa, mục lục, lời nói đầu và các trang cuối chương (`Key Terms`, `Section Summary`, `Assessments`: tr. `117–121` ở Vol 1 và tr. `80–84` ở Vol 2).
- **Chuẩn hóa, lỗi và giới hạn:**
  - Trích xuất bằng `pypdf 6.19.0` với `extraction_mode="layout"` để bảo toàn ranh giới đoạn `\n\n` và tránh lỗi dính chữ/tách chữ (`LINK TO LEARNING`).
  - Hàm `normalize_text` trong [starter/src/ingestion.py](../../starter/src/ingestion.py) chuẩn hóa Unicode `NFC`, loại bỏ header/footer lặp lại của OpenStax (`Access for free at openstax.org` và dòng tiêu đề số trang/chương), ghép dòng bị ngắt trong cùng đoạn (giữ nguyên gạch nối thật như `self-sufficient`, `Mohenjo-Daro`), đồng thời giữ tách dòng cho gạch đầu dòng `•`.
  - **Log lỗi / trang rỗng:** `0` trang lỗi hay rỗng trong 75 trang đã chọn (`error_or_empty_pages: []` trong [ingestion_stats.json](../../starter/results/ingestion_stats.json)).
  - **Giới hạn xử lý:** Chú thích ảnh (`FIGURE ...`) và hộp `LINK TO LEARNING` được giữ lại trong văn bản trang; ngoài ra, nội dung bảng/timeline dạng ảnh không trích được chữ nằm trong hình.
- **Embedding model, Vector Store, Metric, Lệnh chạy và Chi phí:**
  - **Sự cố provider ban đầu & phương án dự phòng:** Lần chạy đầu thử `gemini-embedding-001` (768 chiều) gặp lỗi `HTTP 429 RESOURCE_EXHAUSTED` (vượt quota rate-limit của Gemini API ngay batch đầu của Strategy A). Lỗi được ghi nhận đầy đủ tại [provider_fallback_log.json](../../starter/results/provider_fallback_log.json) và chuyển sang **OpenRouter** cho toàn bộ các chiến lược để giữ một model đồng nhất.
  - **Embedding model cố định cho A, B, C:** `openai/text-embedding-3-small` qua **OpenRouter** (`provider: openrouter`, `1536` chiều, batch size `50`). Dùng cùng hàm gọi embedding chuẩn hóa cho cả document và query (`openai` SDK `2.44.0`).
  - **Vector store & metric:** **PostgreSQL `16.16` + pgvector `0.8.7`** chạy qua Docker (`starter/docker-compose.yml`, bảng `rag_chunks_1536`), sử dụng **cosine distance** (`<=>`, điểm similarity `= 1 - cosine_distance`) với exact sequential scan (không dùng approximate index nên filter không làm rơi kết quả top-k).
  - **Kiểm chứng idempotency (nạp lại):** Đã chạy nạp lại Strategy B trong `run.py`; số dòng trong PostgreSQL trước và sau khi upsert giữ nguyên `322 -> 322` (`passed: true`), đồng thời có logic xóa chunk cũ nếu cùng `document_id` thay đổi số lượng chunk.
  - **Lệnh chạy (từ `starter/`):**
    ```bash
    docker compose -p day6_student_starter up -d --wait
    python scripts/check_setup.py
    python -m src.run
    ```
  - **Chi phí & tài nguyên:** Tổng `39` API requests, `831` lượt nhúng (`487` A + `322` B + `22` queries; C dùng lại vector của B), `92.494` tokens, chi phí ước tính theo biểu giá OpenRouter (`$0.02 / 1M tokens`) là **$0.00185 USD**.
- **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn:**
  - Bắt buộc trên mọi chunk: `document_id` (`wh_vol1`/`wh_vol2`), `source` (đường dẫn file PDF), `page` (số trang PDF vật lý).
  - Bổ sung phục vụ filter: `volume` (`"Volume 1"` / `"Volume 2"`), `chapter`, và `section` (ví dụ `"3.2 Ancient Mesopotamia"`, `"2.2 The Malacca Sultanate"`), lấy trực tiếp từ outline mục lục chính thức của PDF (`section_ranges` trong `manifest.json`).
  - **Lý do chọn:** Corpus gồm 2 tập lịch sử thuộc hai thời kỳ khác nhau (trước 1500 vs từ 1400) và các chủ đề thương mại/chính trị dễ trùng từ khóa (như thu thuế, giao thương đường biển, quy định tôn giáo), nên `volume` và `section` giúp khoanh vùng truy xuất khi người dùng hỏi theo phạm vi mục/tập cụ thể.

---

## 2. Cấu hình và benchmark

- **Ba chiến lược so sánh có kiểm soát (cùng corpus 75 trang, cùng `openai/text-embedding-3-small`, cùng `PgVectorStore`):**
  - **A (Baseline Fixed-size):** `fixed_size(size=500, overlap=50)`, giữ metadata nguồn/trang/section, **không** áp dụng metadata filter khi truy vấn.
  - **B (Structure-aware Paragraph):** `by_paragraph(max_size=900)`, gom các đoạn văn `\n\n` đến tối đa `900` ký tự (đoạn dài hơn `900` được cắt theo ranh giới câu), **không** áp dụng metadata filter. *(Thay đổi duy nhất A→B: chiến lược chunking).*
  - **C (Paragraph + Metadata Filter):** Chunking và vector **giống hệt B**, chỉ áp dụng thêm SQL `WHERE` filter theo `volume` hoặc `section` đối với các câu hỏi có khai báo `filter` từ ngữ cảnh người dùng trước truy xuất. *(Thay đổi duy nhất B→C: metadata filtering).*
- **Bộ benchmark (`22` câu hỏi tại [benchmark/questions.json](../../benchmark/questions.json)):**
  - Tách cố định trước khi chạy: **`16` câu `dev`** (`q01–q16`) và **`6` câu `test`** (`q17–q22`), cân bằng `11` câu từ `wh_vol1` và `11` câu từ `wh_vol2`, phủ các dạng: trực tiếp (`truc_tiep`), diễn đạt lại (`dien_dat_lai`), điều kiện/ngoại lệ (`dieu_kien_ngoai_le`).
  - Có **`5` câu kiểm tra metadata filter** (`q05`, `q07`, `q13` ở `dev`; `q18`, `q21` ở `test`), giá trị filter lấy từ phạm vi tập/mục trong câu hỏi người dùng, **không** đưa `gold_source_id`, `gold_page` hay đáp án vào query/filter.
  - **Kiểm chứng evidence:** Trước khi nhúng vector, script kiểm tra xác nhận 22/22 `evidence_quote` tồn tại nguyên văn (sau chuẩn hóa khoảng trắng/dấu nháy) trên đúng `gold_page`, và 5/5 câu có `filter` khớp metadata trang đích.
- **Tiêu chí chọn chiến lược chốt trước khi xem test:** Ưu tiên `Evidence Hit@3` trên tập `dev`, sau đó tới `Hit@1` và chất lượng `Grounding`/`Coherence` thủ công, cuối cùng xét số chunk, thời gian index/search và chi phí.

---

## 3. Kết quả

### Bảng 1: Hiệu năng truy xuất và tài nguyên (`dev = 16`, `test = 6`)

| Strategy | Split |   Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian index / search |                   Chi phí |
| -------- | ----- | ------------: | -------------------: | ----------: | ------------------------: | -------------------------: |
| A        | dev   |  9/16 (56.3%) |        13/16 (81.3%) |         487 |          20.915s / 0.726s |                  ~\$0.0011 |
| A        | test  |   3/6 (50.0%) |          3/6 (50.0%) |         487 |          *(như trên)* |           *(như trên)* |
| B        | dev   | 10/16 (62.5%) |        15/16 (93.8%) |         322 |          11.187s / 0.742s |                  ~\$0.0007 |
| B        | test  |   5/6 (83.3%) |         6/6 (100.0%) |         322 |          *(như trên)* |           *(như trên)* |
| C        | dev   | 10/16 (62.5%) |        15/16 (93.8%) |         322 |         11.204s* / 0.612s | \$0* (dùng lại vector B) |
| C        | test  |   5/6 (83.3%) |         6/6 (100.0%) |         322 |          *(như trên)* |           *(như trên)* |

*\*Ghi chú: Tổng chi phí thực tế toàn bộ phiên chạy (A + B + 22 queries) là `$0.00185` cho `92.494` tokens (`5.707s` đọc & chuẩn hóa 75 trang PDF).*

### Bảng 2: Đánh giá chất lượng thủ công (Coherence, Metadata, Grounding)

| Strategy | Coherence TB /2 (`n=10`) | Metadata đủ & đúng (`n/N`) |                    Filter tuân thủ (`n/N`) | Grounding 0/1/2 (số câu `test`) & TB /2 |
| -------- | -------------------------: | -------------------------------: | ---------------------------------------------: | ------------------------------------------- |
| A        |   0.50 / 2 (0:5, 1:5, 2:0) |                            17/18 | N/A (không bật filter; tự nhiên khớp 5/5) | `0:0 / 1:4 / 2:2` → **1.33 / 2**   |
| B        |   1.80 / 2 (0:0, 1:2, 2:8) |                            17/18 | N/A (không bật filter; tự nhiên khớp 5/5) | `0:0 / 1:0 / 2:6` → **2.00 / 2**   |
| C        |   1.80 / 2 (0:0, 1:2, 2:8) |                            17/18 |                        5/5 (dev 3/3, test 2/2) | `0:0 / 1:0 / 2:6` → **2.00 / 2**   |

- **Đường dẫn trace, nhãn thủ công và cách chọn mẫu:**
  - Trace top-3 đầy đủ (đồng bộ tại `results/` và `starter/results/`): [traces_A.json](../../results/traces_A.json), [traces_B.json](../../results/traces_B.json), [traces_C.json](../../results/traces_C.json)
  - Tổng hợp tự động & thống kê ingest: [benchmark_summary.json](../../results/benchmark_summary.json), [ingestion_stats.json](../../results/ingestion_stats.json)
  - Mẫu và nhãn chấm thủ công: [manual_audit_samples.json](../../results/manual_audit_samples.json), [manual_audit_completed.json](../../results/manual_audit_completed.json)
  - **Cách chọn mẫu:** 10 vị trí coherence được chốt trước theo 5 trang đầu tiên của mỗi volume (`wh_vol1` tr. `79–83`, `wh_vol2` tr. `43–47`); toàn bộ `6` câu `test` (`q17–q22`, `6 × 3 = 18` bản ghi mỗi cấu hình) được đọc kiểm tra cho Metadata và Grounding. Người chấm là tác giả bài làm (có nguy cơ thiên lệch nhận thức khi tự viết câu hỏi và tự chấm).

---

## 4. Phân tích và kết luận

- **So sánh A → B (tác động của Chunking):**
  - Chuyển từ cắt cố định 500 ký tự (A) sang gom đoạn tối đa 900 ký tự (B) giúp **giảm 33.9% số chunk** (`487 → 322`), giảm gần một nửa thời gian embed (`19.84s → 10.48s`), đồng thời tăng mạnh **Evidence Hit@3** từ `13/16` lên `15/16` trên `dev` và từ `3/6` lên `6/6` trên `test`, tăng **Coherence** từ `0.50/2` lên `1.80/2`, và nâng **Grounding** trên `test` từ `1.33/2` lên `2.00/2`.
  - Nguyên nhân chính: A liên tục cắt ngang giữa từ và giữa câu với overlap chỉ 50 ký tự, làm đứt đoạn bằng chứng hoặc tách mất chủ ngữ của câu.
- **So sánh B → C (tác động của Metadata Filtering):**
  - C tuân thủ filter `5/5` (`100%`) và giảm thời gian search trên 22 câu từ `0.742s` xuống `0.612s` (`-17.5%`).
  - Trên bộ 22 câu này, `Hit@1` và `Evidence Hit@3` của B và C bằng nhau (`10/16` dev, `5/6` test) vì các câu hỏi có filter đã chứa nhiều từ khóa đặc trưng của từng vùng/triều đại (`Hammurabi`, `Djoser`, `shahbandar`, `Khufu`, `Ming China`), nên tìm kiếm vector của B tự nhiên cũng khớp đúng phạm vi (`5/5`). Ngược lại, ở các câu không khai báo filter nhưng dùng khái niệm tổng quát như `q19` (hỏi về thương mại và thuế ở nền văn minh sông Ấn), không có bộ lọc khiến chunk `wh_vol1_p0095_para_005` thuộc mục `3.2 Ancient Mesopotamia` (nói về thu thuế nông sản) xếp ở `rank 1` trên chunk đúng thuộc `3.4 The Indus Valley Civilization` (`rank 2`, tr. 115). Đây là minh họa trực tiếp cho giá trị của **C** trong việc chặn nhiễu xuyên section/volume khi người dùng hỏi theo ngữ cảnh tổng quát.
- **Hai ví dụ lỗi điển hình từ trace:**
  1. **Lỗi cắt đứt bằng chứng ở Strategy A — Câu `q17` (`test`, `wh_vol1` tr. 101):** Câu hỏi về lý do người Ai Cập có thế giới quan trật tự hơn Lưỡng Hà. A truy xuất đúng trang 101 ở `rank 1` (`wh_vol1_p0101_fixed_004`, score `0.68938`) và `rank 3` (`wh_vol1_p0101_fixed_003`, score `0.551182`), nhưng câu văn đối chiếu sông Tigris–Euphrates và sông Nile bị cắt làm đôi giữa hai chunk (`"In lat"` / `"balance. In contrast..."`), khiến cả top-3 đều trượt `Evidence Hit@3` và Grounding chỉ đạt `1/2`. Ở B/C, `wh_vol1_p0101_para_003` giữ trọn đoạn và đạt `Hit@1`. *(Tương tự ở `q18`, `q14`, `q15`, `q22` của A).*
  2. **Lỗi tách đoạn quá `max_size` không overlap ở Strategy B/C — Câu `q06` (`dev`, `wh_vol1` tr. 96):** Câu hỏi về giới hạn giai cấp của luật Hammurabi (*"an eye for an eye"*). Đoạn văn gốc trên trang 96 dài ~1.070 ký tự (vượt `max_size=900`), nên `by_paragraph` cắt đôi đoạn theo ranh giới câu thành `wh_vol1_p0096_para_002` (rank 3) và `wh_vol1_p0096_para_003` (rank 2), vô tình chia cắt hai câu liên tiếp trong `evidence_quote`; trong khi `rank 1` lại lấy hộp luật Hammurabi tr. 94 (`wh_vol1_p0094_para_001`). Kết quả là `q06` trở thành ca miss `Hit@3` duy nhất của B/C trên `dev`.
  3. **Lỗi biên metadata cấp trang (Metadata audit — Câu `q20` rank 1, tr. 66 `wh_vol2`):** Ở trang 66, ba đoạn đầu viết về người Hà Lan ở quần đảo Banda/Ambon (kết mục `2.2 The Malacca Sultanate`), còn tiêu đề `2.3 Exchange in East Asia` bắt đầu ở giữa trang. Do gán `section` theo cấp trang, chunk `wh_vol2_p0066_para_001` bị gán nhầm `section = "2.3 Exchange in East Asia"` (`17/18` bản ghi test đúng hoàn toàn metadata).
- **Chiến lược lựa chọn, đánh đổi & điều kiện áp dụng:**
  - Chọn **Chiến lược C** (`by_paragraph`, `max_size=900` + metadata filter khi có ngữ cảnh lọc).
  - **Đánh đổi:** Kích thước chunk lớn hơn (`637.3` vs `464.3` ký tự TB) giúp trọn vẹn ngữ cảnh và ít tốn chi phí embedding hơn, nhưng đoạn dài vượt `900` ký tự bị tách không có overlap (gây lỗi `q06`), và lọc `section` theo trang có rủi ro loại nhầm đoạn ở trang chuyển tiếp nếu không gán `section` theo từng đoạn.
  - **Điều kiện áp dụng:** Phù hợp khi tài liệu có ranh giới đoạn văn rõ ràng (giáo trình, tài liệu kỹ thuật, chính sách) và giao diện/người dùng có thể cung cấp bộ lọc (`volume`, `section`) tin cậy trước khi truy xuất; khi không có filter rõ ràng, C tự động lùi về hành vi của B mà không làm hẹp tập ứng viên ngoài ý muốn.
- **Hạn chế:**
  - **Dữ liệu, nhãn và tập test nhỏ:** Tập `test` chỉ có `6` câu nên sai số lấy mẫu lớn (`100%` trên B/C chỉ mang tính minh họa cho corpus này, không suy rộng tổng quát); chưa tách biệt hoàn toàn nhiễu caption `FIGURE` hoặc hộp `LINK TO LEARNING`; cùng tác giả vừa viết câu hỏi vừa tự chấm tay.
  - **Đo thời gian & chi phí:** Thời gian `embed_seconds` (`19.84s` / `10.482s`) và `search_seconds` (`0.612s–0.742s`) chỉ đo trong 1 lần chạy đơn lẻ qua API OpenRouter và local Docker PostgreSQL (chịu ảnh hưởng dao động mạng, độ trễ cold-start và kích thước corpus nhỏ `322–487` chunks); chi phí `$0.00185` là ước lượng từ `usage.total_tokens` (`92.494` tokens × `$0.02/1M`) theo bảng giá công bố của OpenRouter cho `openai/text-embedding-3-small`.
- **Sử dụng AI và cách kiểm tra (theo `RULES.md`):**
  - **Công cụ sử dụng:** Dùng trợ lý lập trình AI hỗ trợ rà soát kiến trúc pipeline `starter/src/`, viết script kiểm tra trích xuất PDF, đồng bộ manifest và cấu trúc báo cáo.
  - **Cách kiểm tra độc lập:** Toàn bộ số liệu trong báo cáo được sinh trực tiếp từ lần chạy thực tế `python -m src.run` với PostgreSQL `16.16` + pgvector `0.8.7` và OpenRouter `openai/text-embedding-3-small`; `22/22` `evidence_quote` được đối chiếu nguyên văn với văn bản chuẩn hóa trên đúng trang PDF nguồn; các mẫu coherence (`n=10`), metadata (`18` bản ghi test/chiến lược) và grounding (`6` câu test/chiến lược) được đọc kiểm tra trực tiếp trên chunk text và trace tại [manual_audit_completed.json](../../results/manual_audit_completed.json).
- **Hướng cải tiến nếu làm lại / mở rộng:**
  1. Gán `section` theo **từng đoạn văn** dựa vào vị trí xuất hiện heading `X.Y` thay vì theo cấp trang để loại bỏ lỗi ranh giới trang 66 (`q20`).
  2. Thêm **1 câu overlap** khi buộc phải tách đoạn dài vượt `max_size` để tránh lỗi ranh giới như `q06`.
  3. Bổ sung **hybrid retrieval** (pgvector cosine + PostgreSQL `tsvector` BM25/FTS) cho các tên riêng/thuật ngữ lịch sử (`shahbandar`, `hangul`, `Imhotep`) và thử nghiệm trên tập câu hỏi chung chung không nêu tên địa danh để đo rõ lợi ích `Hit@1`/`Hit@3` của metadata filter.
