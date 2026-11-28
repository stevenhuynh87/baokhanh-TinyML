# S01 — Literature reading notes and reading evidence

**Issue:** [S01 #1](https://github.com/stevenhuynh87/baokhanh-TinyML/issues/1)

**PR:** [#27](https://github.com/stevenhuynh87/baokhanh-TinyML/pull/27)

**CSV:** [lit_table_v0.csv](lit_table_v0.csv)

## Reading status and AI disclosure

- FULL_PAPER: mức đọc toàn văn do sinh viên tự xác nhận; không phải bằng chứng kiểm chứng độc lập.
- ABSTRACT_ONLY: mới xác nhận đọc Abstract; ghi chú dựa trên phân tích AI không thay thế việc đọc toàn văn.
- AI được sử dụng để giải thích, phân tích bài báo, gợi ý nội dung và nhận xét mẫu, đối chiếu Section/Figure, kiểm tra số liệu và chỉnh sửa cách trình bày.
- Một số ghi chú dựa sát vào gợi ý của AI; không khẳng định đã đáp ứng hoàn toàn yêu cầu tự viết.
- Các ghi chú cần được sinh viên đối chiếu với bài gốc; chưa được thầy nghiệm thu.
- Không tuyên bố đã tự tái lập thí nghiệm.

| Paper | Reading status | Notes status |
|---|---|---|
| L1 | ABSTRACT_ONLY | Chưa xác nhận đọc toàn văn |
| L5 | FULL_PAPER | 7 ý, chờ review |
| L7 | ABSTRACT_ONLY | Chưa xác nhận đọc toàn văn |
| L13 | FULL_PAPER | 5 ý, chờ review |
| L14 | FULL_PAPER | 5 ý có AI hỗ trợ; chờ review |
| L15 | ABSTRACT_ONLY | Nhận xét Abstract; 5 ý đọc sâu còn thiếu |
| L24 | ABSTRACT_ONLY | Có nhận xét Abstract |

---

## L5 — SONIC/TAILS (ASPLOS 2019) — B2

**Paper:** [Intelligence Beyond the Edge: Inference on Intermittent Embedded Systems](https://doi.org/10.1145/3297858.3304011)

**Reading status:** `FULL_PAPER`

**AI assistance:** YES

**Ghi nhận:** Em đã đọc TOÀN VĂN PDF gốc L5 (FULL_PAPER) và đã viết ghi chú dài có dẫn Section/Figure. Có sử dụng AI hỗ trợ học tập: giải thích thuật ngữ, hướng dẫn đọc và đối chiếu Section/Figure, kiểm tra số liệu, chỉnh lỗi chính tả và cách trình bày. Nội dung có AI hỗ trợ; không tuyên bố ghi chú được tạo hoàn toàn độc lập với AI. Trạng thái FULL_PAPER dựa trên lời xác nhận của em, chưa được kiểm chứng độc lập; ghi chú chờ thầy xem xét. Không tuyên bố đã tự tái lập thí nghiệm.

### Reading notes

1. **§2.1; §6.2.1–6.2.2** — Khi thiết bị mất điện, SONIC hướng tới giải quyết cả hai vấn đề: giữ được tiến độ để không tính lại quá nhiều và bảo đảm dữ liệu vẫn đúng khi khôi phục. §2.1 giải thích các vấn đề của thực thi gián đoạn, còn cơ chế SONIC được trình bày trong §6.2.1–6.2.2.

2. **§6.2 và §6.2.1; Figure 6** — Tile-5 lãng phí năng lượng do khi mất điện thì task chưa hoàn thành sẽ phải thực hiện lại những lần lặp đó sau khi bật lên. Tile-12 cần nhiều năng lượng hơn thiết bị có thể tích trữ trong một chu kỳ, khiến thiết bị cứ chạy rồi hết điện, khởi động lại và thất bại (hiện tượng non-termination). Với SONIC, khi mất điện thì tiến độ không bị xóa hoàn toàn; khi có điện trở lại, chương trình tiếp tục tại lần lặp gần với vị trí đã dừng.

3. **§6.2.2; Figure 7** — Chỉ giữ tiến độ thôi chưa đủ: nếu mất điện khi lần lặp đang cập nhật dữ liệu thì kết quả có thể bị sai dù thiết bị vẫn lưu dữ liệu trong FRAM. SONIC giải quyết yêu cầu idempotence bằng hai kỹ thuật loop-ordered buffering và sparse undo-logging. Loop continuation giải quyết tiến độ; buffering và undo-logging bảo đảm tính đúng của dữ liệu. Figure 7 minh họa loop-ordered buffering; phần sparse undo-logging được giải thích bằng chữ tại §6.2.2, trang in 207.

4. **§5.2–5.3; Figures 4–5** — GENESIS nén và lựa chọn cấu hình DNN dựa trên bộ nhớ, chi phí suy luận và hiệu quả ứng dụng. GENESIS áp dụng pruning và separation, sau đó đánh giá độ chính xác, số phép toán, bộ nhớ và hiệu quả toàn ứng dụng. §6.2; Figures 6–7 — SONIC giảm chi phí thực thi DNN gián đoạn, giữ tiến độ và tính đúng. §7.1–7.2 — TAILS dùng DMA và bộ tăng tốc LEA để thực hiện phép toán hiệu quả hơn; DMA chuyển dữ liệu và LEA thực hiện phép tính vector. Tuy nhiên, không phải phép toán nào cũng tăng tốc hiệu quả; các lớp fully-connected thưa vẫn chạy bằng phần mềm theo cách của SONIC.

5. **Baseline B2; §8; Figure 8** — Đây là thông số thí nghiệm của tác giả: vi điều khiển TI MSP430FR5994, tần số 16 MHz; bộ thu năng lượng Powercast P2210B; bộ phát RF công suất 3 W cách 1 m; tụ điện 100 µF, 1 mF, 50 mF; các mạng MNIST, HAR, OkG. Các baseline so sánh gồm bản thông thường không chịu được mất điện và các biến thể Tile-8, Tile-32, Tile-128 dựa trên Alpaca.

6. **§9.1, Figure 9; §9.3, Figure 11** — Tác giả báo cáo SONIC và TAILS cải thiện hiệu năng/thời gian suy luận trung bình lần lượt 6,9× và 12,2× so với tiled Alpaca; Figure 11 cho thấy mức tiết kiệm năng lượng tương ứng trong thí nghiệm. Đây không phải cải thiện độ chính xác của DNN, cũng không phải kết quả do tôi tự chạy.

7. **§9.4; Figure 12; §10** — Tác giả nêu khoảng 26% năng lượng SONIC dành cho lệnh điều khiển và 14% năng lượng hệ thống dành cho việc ghi chỉ số vòng lặp vào FRAM; những chi phí này vẫn đáng kể. Section 10 đề xuất nghiên cứu phần cứng hỗ trợ tốt hơn cho việc lưu chỉ số và thực thi gián đoạn. SONIC giảm việc thực hiện lại công việc nhưng vẫn tiêu tốn năng lượng cho điều khiển và cập nhật trạng thái. Theo tôi cần nghiên cứu cách giảm chi phí bảo toàn trạng thái trong suy luận gián đoạn.

---

## L13 — Zygarde (IMWUT 2020) — B4

**Paper:** [Zygarde: Time-Sensitive On-Device Deep Inference and Adaptation on Intermittently-Powered Systems](https://doi.org/10.1145/3411808)

**Reading status:** `FULL_PAPER`

**AI assistance:** YES

**Ghi nhận:** Em đã đọc TOÀN VĂN PDF gốc L13 (FULL_PAPER) và đã viết 5 ý ghi chú có dẫn Section/Figure. Có sử dụng AI hỗ trợ học tập: giải thích thuật ngữ, hướng dẫn đọc và đối chiếu Section/Figure, kiểm tra số liệu, chỉnh lỗi chính tả và cách trình bày. Nội dung có AI hỗ trợ; không tuyên bố ghi chú được tạo hoàn toàn độc lập với AI. Trạng thái FULL_PAPER dựa trên lời xác nhận của em, chưa được kiểm chứng độc lập; ghi chú chờ thầy xem xét. Không tuyên bố đã tự tái lập thí nghiệm.

### Reading notes

1. **Section 1, Figure 1** — Do nguồn điện liên tục bị gián đoạn nên thiết bị không có đủ thời gian hoạt động để hoàn thành toàn bộ công việc. Dù EDF ưu tiên công việc gần deadline nhất nhưng J2 vẫn bị trễ trong Figure 1(b). Zygarde giải quyết bằng cách chia công việc thành hai phần: Mandatory là phần tính toán bắt buộc để có kết quả đạt mức tin cậy cần thiết; Optional nhằm cải thiện độ chính xác nếu còn thời gian và năng lượng. Ở Figure 1(c), phần Mandatory của cả J1 và J2 hoàn thành đúng hạn. Theo tôi, khi điện không ổn định, ưu tiên kết quả đủ tin cậy đúng hạn hữu ích hơn việc cố chạy hết DNN nhưng trả kết quả quá muộn.

2. **Section 4.1–4.3, Figures 5–8** — Mỗi mẫu dữ liệu có độ khó phân loại khác nhau. Zygarde sử dụng Agile DNN và bộ phân loại k-Means ở từng lớp. Figure 5 cho thấy hệ thống so sánh khoảng cách đến hai tâm cụm gần nhất: nếu một tâm cụm gần hơn rõ rệt, kết quả tương đối tin cậy và hệ thống có thể dừng sớm; nếu hai khoảng cách gần giống nhau thì hệ thống chạy tiếp lớp sau. Figure 7 minh họa cách huấn luyện Agile DNN bằng Siamese Network và layer-aware loss. Cách này có thể tiết kiệm tính toán, nhưng nếu đánh giá độ tin cậy sai thì kết quả dừng sớm có thể sai.

3. **Section 3.3, Figure 4; Section 5.2, Equation 7** — Zygarde cần η-factor thay vì chỉ kiểm tra năng lượng hiện tại vì năng lượng hiện tại chỉ cho biết thiết bị đang có bao nhiêu điện, còn η-factor mô tả mức độ có quy luật của nguồn năng lượng theo thời gian. Nếu chỉ xét năng lượng hiện tại, Scheduler có thể xem các trường hợp nguồn khác nhau như nhau. Khi xét thêm η-factor, hệ thống có thêm thông tin để đánh giá điều kiện năng lượng và quyết định ưu tiên Mandatory hay cân nhắc thêm Optional. Tuy nhiên, η-factor chỉ là mô hình ước lượng, không bảo đảm dự đoán chính xác mọi lần mất điện.

4. **Section 8.4, Figure 16; Section 9.2, Figure 23** — Figure 23 cho thấy Zygarde hoàn thành đúng hạn 93% số job đã vào hệ thống, cao hơn SONIC-EDF (55%) và SONIC-RR (11%). Tuy nhiên, tác giả ghi nhận 37% sự kiện ban đầu không được cảm biến thu nhận vì camera cần nhiều năng lượng. Figure 16 còn cho thấy cơ chế dừng sớm giảm thời gian suy luận trung bình 4–26% trong các trường hợp thử nghiệm. Theo tôi, Zygarde cải thiện khả năng đúng hạn nhưng 93% không có nghĩa thiết bị thành công với 93% mọi sự kiện ngoài thực tế vì vẫn có thể bỏ lỡ sự kiện từ bước thu dữ liệu.

5. **Section 11.5** — Zygarde không bảo đảm tất cả các job hoàn thành Mandatory trước deadline vì nguồn điện có thể bị gián đoạn. Nếu thiết bị mất điện quá lâu thì Scheduler không thể thực hiện tính toán do thiếu năng lượng. Ngoài ra, hệ thống không chủ động lập lịch thời điểm thức dậy nên có thể bỏ lỡ sự kiện khi đang tắt nguồn. Kích thước hàng đợi bị giới hạn bởi bộ nhớ; trong thí nghiệm ở Section 8, tác giả dùng hàng đợi 3 job. Theo tôi, Zygarde cải thiện khả năng suy luận đúng hạn nhưng vẫn chịu ảnh hưởng của năng lượng không ổn định và dữ liệu cảm biến bị bỏ lỡ.

---

## L14 — Wu et al. (DAC 2020) — B4

**Paper:** [Intermittent Inference with Nonuniformly Compressed Multi-Exit Neural Network for Energy Harvesting Powered Devices](https://doi.org/10.1109/DAC18072.2020.9218526)

**Reading status:** `FULL_PAPER`

**AI assistance:** YES

**Ghi nhận:** Em xác nhận đã trực tiếp đọc toàn văn L14 (FULL_PAPER) và cung cấp 5 ý nhận xét có dẫn Section/Figure dựa trên phần phân tích và gợi ý của AI. AI hỗ trợ giải thích, gợi ý nhận xét, đối chiếu vị trí Section/Figure và chỉnh chính tả. Đã đủ 5 ý về số lượng; mức đọc dựa trên lời xác nhận của em, chưa được kiểm chứng độc lập. Ghi chú chờ thầy đánh giá; không tuyên bố được tạo độc lập với AI hoặc đã tự tái lập thí nghiệm.

### Reading notes

1. **Section III, Figure 1(b)** — Nếu nén quá mạnh ở các lớp đầu thì độ chính xác của early-exit có thể bị giảm đáng kể. Nén không đồng đều sẽ giúp hạn chế sự suy giảm này, đặc biệt đối với các điểm thoát thường xuyên được sử dụng.

2. **Section V.C, Figure 5** — Mặc dù độ chính xác trên các sự kiện đã được xử lý thấp hơn các baseline, phương pháp của Wu xử lý được nhiều sự kiện hơn. Theo tôi, không nên chỉ đánh giá độ chính xác của từng lần suy luận mà còn phải xem xét số sự kiện bị bỏ lỡ do thiếu năng lượng.

3. **Sections IV, VI–VII** — Theo tôi, có thể nghiên cứu kết hợp phương pháp lựa chọn exit của L14 với cơ chế bảo toàn tiến độ của SONIC trong L5 để giảm chi phí tính toán lại khi mất điện. Đây là hướng nghiên cứu đề xuất, chưa được kiểm chứng bằng thực nghiệm.

4. **Section II, Figure 1(a), Figure 1(c)** — Việc sử dụng mạng multi-exit giúp thiết bị không cần chờ tích lũy đủ năng lượng để chạy hết DNN. Khi năng lượng hạn chế, thiết bị có thể đưa ra kết quả tại điểm thoát sớm và tiếp tục suy luận nếu kết quả chưa đủ tin cậy và có thêm năng lượng. Theo tôi, phương pháp này giúp giảm thời gian chờ đợi khi nguồn điện không ổn định.

5. **Section V.E, Figure 7(a), Figure 7(b)** — Q-learning giúp thiết bị điều chỉnh cách lựa chọn điểm thoát dựa trên điều kiện năng lượng thay vì sử dụng một chính sách cố định. Theo tôi, không phải lúc nào lựa chọn điểm thoát có độ chính xác cao nhất cũng mang lại hiệu quả tốt nhất, mà cần cân bằng giữa độ chính xác và năng lượng dành cho những sự kiện tiếp theo.

---

## L15 — HarvNet (MobiSys 2023) — B5

**Paper:** [HarvNet: Resource-Optimized Operation of Multi-Exit Deep Neural Networks on Energy Harvesting Devices](https://doi.org/10.1145/3581791.3596845)

**Reading status:** `ABSTRACT_ONLY`

**AI assistance:** YES

**Deep notes status:** `PENDING_STUDENT_AUTHORED_5_LINES`

**Ghi nhận:** Sinh viên xác nhận chỉ đọc Abstract và cung cấp nhận xét ở mức Abstract. Có AI hỗ trợ phân tích và chỉnh chính tả. Chưa đọc toàn văn; chưa hoàn thành 5 ý ghi chú có dẫn Section/Figure. Không tuyên bố đã kiểm chứng độc lập kết quả thực nghiệm.

### Personal reflection

**Abstract** — HarvNet tối ưu cả cách lựa chọn điểm thoát khi suy luận và kiến trúc của mạng neural ngay từ trước khi triển khai. Việc kết hợp HarvNAS và HarvSched có khả năng giúp thiết bị sử dụng bộ nhớ và năng lượng hiệu quả hơn. Tuy nhiên, Abstract chưa giải thích chi tiết điều kiện để hệ thống đạt được zero downtime và chưa cung cấp đủ thông tin để đánh giá khả năng duy trì kết quả này khi nguồn năng lượng thay đổi bất thường.

### Pending work

Phần còn thiếu: 5 ý ghi chú có dẫn Section/Figure sau khi đọc sâu bài gốc; chưa xác nhận các điều kiện thí nghiệm và khả năng duy trì zero downtime. Giữ ABSTRACT_ONLY và báo cáo phần chưa hoàn thành.

---

## L24 — Early-Exit Survey (ACM CSUR)

**Paper:** [Early-Exit Deep Neural Network - A Comprehensive Survey](https://doi.org/10.1145/3698767)

**Reading status:** `ABSTRACT_ONLY`

**AI assistance:** YES

**Ghi nhận:** Sinh viên xác nhận chỉ đọc Abstract và cung cấp nhận xét ở mức Abstract. Có AI hỗ trợ phân tích và chỉnh chính tả. Chưa đọc toàn văn; không tuyên bố đã hoàn thành survey review. PDF citation says Nov 2024; database volume/issue is 2025, confirm citation convention if needed.

### Personal reflection

**Abstract** — Việc sử dụng early-exit giúp tránh những phép tính không cần thiết đối với dữ liệu dễ phân loại, từ đó giảm thời gian suy luận. Ngoài ra, bài báo còn giúp hiểu rõ hơn về cách thiết kế các điểm thoát và lựa chọn exit phù hợp với thiết bị có tài nguyên hạn chế. Tuy nhiên, Abstract chưa cung cấp chi tiết về cách triển khai khi nguồn điện bị gián đoạn hoặc cách bảo toàn trạng thái khi mất điện.

---

## L1 và L7 — Các bài còn lại

### L1 — Chinchilla (B3)

**Paper:** [Adaptive Dynamic Checkpointing for Safe Efficient Intermittent Computing](https://www.usenix.org/conference/osdi18/presentation/maeng)

**Reading status:** `ABSTRACT_ONLY`

Chưa xác nhận đọc toàn văn.

### L7 — Stateful Neural Networks

**Paper:** [Stateful Neural Networks for Intermittent Systems](https://doi.org/10.1109/TCAD.2022.3197513)

**Reading status:** `ABSTRACT_ONLY`

Chưa xác nhận đọc toàn văn.

---

## Summary

| Paper | Reading status | Deep notes | Advisor review |
|---|---|---|---|
| L5 | FULL_PAPER | 7 notes | Pending |
| L13 | FULL_PAPER | 5 notes | Pending |
| L14 | FULL_PAPER | 5 notes | Pending |
| L15 | ABSTRACT_ONLY | Incomplete | Pending |

## Pending items before advisor acceptance

- [ ] Đối chiếu ghi chú với PDF gốc; L14 đã xác nhận đọc toàn văn, chờ thầy đánh giá.
- [ ] Tự kiểm chứng và diễn đạt lại các nhận xét còn phụ thuộc nhiều vào gợi ý AI.
- [ ] Bổ sung 5 ý có Section/Figure cho L15 sau khi đọc sâu.
- [ ] Tiếp tục đọc L1, L7 và L24 theo phạm vi yêu cầu.
- [ ] Đồng bộ CSV và Markdown trên PR #27; báo cáo phần còn thiếu trong Issue #1.
- [ ] Chờ thầy đánh giá trước khi đánh dấu hoàn thành.
