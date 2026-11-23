# Research & Algorithm Development Roadmap — 12 tháng

**Đề tài:** *State-Preserving Intermittent Deep Learning via Dynamic Early-Exit and Non-Volatile Memory Mapping on Micro-Watt IoT Nodes* (DT-01)
**Sinh viên:** Bảo Khánh · **Cố vấn:** TS. Huỳnh Võ Trung Dũng (IU-VNUHCMC)
**Ngày lập:** 30/09/2026 · **Phạm vi thời gian:** Sprint 1 bắt đầu 05/10/2026 → Sprint 26 kết thúc 03/10/2027
**Ràng buộc cố định:** chỉ algorithm + model + simulation + experimental validation trên GPU hiện có tại trường. Không hardware acquisition, không deployment, không field test.

**Quy ước nhãn trong tài liệu**

| Nhãn | Ý nghĩa |
|---|---|
| **[FILE]** | Nội dung có trong file nguồn của folder |
| **[VERIFIED]** | Đã kiểm chứng bằng nguồn công khai (link ở Phụ lục A) |
| **[INFERENCE]** | Suy luận của người lập roadmap, chưa phải sự kiện |
| **[TBD]** | Thiếu thông tin — kèm assumption và cách kiểm chứng |
| **[TARGET]** | Ngưỡng thành công đặt ra để kiểm định giả thuyết. **Không phải kết quả.** |

Tài liệu không chứa bất kỳ kết quả thực nghiệm nào. Mọi con số dạng "≥ 2×", "≥ +2 pp" là **[TARGET]** do cố vấn chốt tại các gate, có thể điều chỉnh.

---

## 1. Executive Summary

**Bottom line.** Đề tài **FEASIBLE WITH CONDITIONS**. Hướng nghiên cứu làm được bằng simulation trên GPU, nhưng **không giữ được nguyên formulation trong file nguồn**. Ba điều kiện bắt buộc:

1. **Định vị lại novelty.** "Early-exit + intermittent inference" đã được công bố nhiều lần (SONIC/TAILS 2019, Zygarde 2020, Wu et al. DAC 2020, HarvNet MobiSys 2023, DualAdaptNet DATE 2025…). Thuật ngữ "stateful" cũng đã dùng (Stateful Neural Networks for Intermittent Systems, IEEE journal 2022; tên tạp chí cụ thể cần xác minh). Ý "zero-copy NVM mapping" trùng với cách SONIC tính toán trực tiếp trên FRAM. Roadmap này đề xuất trọng tâm novelty mới: **học trạng thái cần lưu (Learned Persistent-State Bottleneck) + huấn luyện có điều kiện theo phân phối năng lượng + chính sách exit/persist được giải bằng MDP có tính chi phí ghi NVM**. Novelty này phải qua **Month-1 Novelty Gate** trước khi đầu tư quy mô lớn.
2. **Đổi venue.** File nguồn nhắm SenSys/MobiSys/RTSS (A/A\*) và chính file nguồn ghi rằng Rank A "không chấp nhận chỉ chạy mô phỏng". Với ràng buộc simulation-only, mục tiêu hợp lý là **ISLPED 2027** (stretch, ICORE A theo nguồn thứ cấp) hoặc **EWSN 2027** (CORE B theo nguồn thứ cấp) ở tháng 6, và **IEEE Internet of Things Journal** (Q1, SJR 2025) ở tháng 12.
3. **Xây simulator có hiệu chỉnh (calibrated intermittent-execution simulator)** là critical path. Toàn bộ evidence phụ thuộc vào độ tin cậy của simulator này. Nó phải được kiểm tra với xu hướng đã công bố của ít nhất 2 baseline trước cuối tháng 3.

**Quyết định do cấp C13 nào chi phối:** *Academic quality* (cấp 1) — một paper simulation-only nhắm venue đòi hardware sẽ bị reject vì lý do phương pháp, không phải vì thuật toán yếu.

**Next step (owner: Bảo Khánh, deadline 18/10/2026):** hoàn thành systematic literature search theo protocol ở Mục 5.4 và nộp novelty memo 2 trang cho cố vấn tại Gate M1.

### 1.1 Các vấn đề phải nêu trước (pushback)

| # | Vấn đề | Loại | Tác động | Xử lý trong roadmap |
|---|---|---|---|---|
| P1 | Prompt yêu cầu phân tích 3DGS, semantic 3DGS, path planning, navigation, traversability, collision avoidance. **Không mục nào xuất hiện trong file nguồn hay trong đề tài DT-01.** Đây có vẻ là template của một đề tài khác. | Sai lệch template | Nếu làm theo sẽ tạo nội dung bịa | Đánh dấu **NOT APPLICABLE**; ánh xạ các yêu cầu template sang đại lượng tương đương của đề tài (Mục 2.3) |
| P2 | Đề tài gốc là bài toán **embedded systems**, target venue gốc đòi testbed phần cứng; ràng buộc mới cấm hardware | Mâu thuẫn phạm vi | Rank A systems venue không khả thi | Đổi venue (Mục 22–23); đóng góp chuyển sang algorithm/model + simulator có hiệu chỉnh |
| P3 | File nguồn ghi Chinchilla là "ACM ASPLOS" | Trích dẫn sai | Mất uy tín khi phản biện | Chinchilla thuộc **USENIX OSDI 2018** (Maeng & Lucia) [VERIFIED] |
| P4 | File nguồn ghi Coala là "ACM SenSys" | Trích dẫn sai | như trên | Coala được công bố tại **ACM Transactions on Sensor Networks (2020)**, "Dynamic Task-based Intermittent Execution for Energy-harvesting Devices" [VERIFIED] |
| P5 | Baseline "Intermittent-Inference Engine (IIE)" | Không xác định được | Không reproduce được | **NEEDS VERIFICATION** — hỏi người soạn file nguồn; nếu không xác định trong Sprint 1 thì loại |
| P6 | "Sập nguồn 10–100 lần mỗi giây", "triệt tiêu 90% chi phí trễ" | Số liệu không có nguồn | Không được dùng làm claim | Chuyển thành tham số simulation (tần suất sập nguồn là biến quét) và giả thuyết cần đo |
| P7 | Công thức exit $T_{exec}(f_k) \le E_{cap}(t)/P_{active}$ bỏ qua năng lượng thu được trong khi chạy, chi phí checkpoint/restore và chi phí boot | Formulation chưa đầy đủ | Policy tối ưu sai | Thay bằng mô hình năng lượng đầy đủ + MDP (Mục 7) |

---

## 2. Current Research Status

### 2.1 Danh mục file đã đọc

| File | Nội dung thực tế | Liên quan DT-01 |
|---|---|---|
| `2026_09_26_BaoKhanh_Research_Topics_AIoT_RankA.md` | Danh mục 10 đề tài AIoT; DT-01 là mục đầu tiên; kèm ma trận chọn đề tài, kế hoạch 4 giai đoạn 6–9 tháng, action items | Mô tả DT-01 (≈1 trang) + các phần chung |
| `2026_09_26_BaoKhanh_Research_Topics_AIoT_RankA.pdf` | Bản PDF biên dịch từ file .md trên (cùng tiêu đề, cùng dung lượng chữ) | Không có nội dung mới |
| `Planning_BaoKhanh.docx` | Chính là prompt nhiệm vụ này | Không có nội dung nghiên cứu |
| `~$anning_BaoKhanh.docx` | File khóa tạm của Word (162 byte) | Bỏ qua |

**Kết luận:** Toàn bộ "trạng thái nghiên cứu" của DT-01 là **một đề xuất đề tài**. Chưa có code, dữ liệu, model, thực nghiệm hay kết quả.

### 2.2 Phân loại trạng thái

| Hạng mục | Nội dung trong file | Trạng thái |
|---|---|---|
| Research problem | Nút IoT dùng năng lượng thu hoạch sập nguồn liên tục; checkpoint toàn bộ activation vào FRAM gây *checkpointing thrashing* [FILE] | **PARTIAL** — có mô tả định tính, chưa có định nghĩa hình thức |
| Research gap | Checkpoint toàn bộ activation tốn năng lượng hơn tính toán [FILE] | **NEEDS VERIFICATION** — gap đã bị thu hẹp bởi SONIC/TAILS, HAWAII, MII, WARio (Mục 5) |
| Research question | "Mạng tự co giãn độ sâu theo đường bao công suất và bảo toàn trạng thái với chi phí ghi bằng không" [FILE] | **PARTIAL** — một câu hỏi tổng, chưa tách thành RQ kiểm định được; "chi phí ghi bằng không" không khả thi về vật lý |
| Research objectives | Không nêu tách riêng | **NOT STARTED** |
| Existing methodology | (1) Early-exit policy qua MDP với ràng buộc năng lượng tụ; (2) Zero-copy heterogeneous memory partitioning; (3) Temporal Bayesian Evidence Accumulator (trong phần phản biện) [FILE] | **PARTIAL** — ý tưởng mức concept, chưa có thuật toán chi tiết |
| Existing code/model | Không có | **NOT STARTED** |
| Existing datasets | Google Speech Commands, CWRU, Visual Wake Words [FILE] | **NOT STARTED** (chưa tải, chưa chia split) — dataset đều công khai |
| Existing baselines | Chinchilla, Coala, IIE [FILE] | **NEEDS VERIFICATION** — 2/3 ghi sai venue, 1/3 không xác định (P3–P5) |
| Existing experiments | Thiết kế thí nghiệm trên MSP430FR5994 + BQ25570 [FILE] | **NOT APPLICABLE** dưới ràng buộc mới (hardware); chuyển thành simulation |
| Metrics | µJ/inference hợp lệ, tỷ lệ hoàn thành khi mất nguồn ngẫu nhiên 50 Hz, độ sụt accuracy [FILE] | **PARTIAL** — giữ và mở rộng |
| Current progress | Đang ở bước chọn đề tài ("Matching Session") [FILE] | **NOT STARTED** |
| GPU hiện có tại trường | Không được nêu | **TBD** — kiểm chứng trong Sprint 1 (`nvidia-smi`, dung lượng VRAM, số GPU, chính sách chia sẻ) |
| Năng lực lập trình của sinh viên | Không được nêu | **TBD** — kiểm tra bằng bài pilot Sprint 1 |

### 2.3 Các hạng mục template không áp dụng

| Hạng mục template | Trạng thái | Đại lượng tương đương trong DT-01 |
|---|---|---|
| 3DGS / Semantic 3DGS / Semantic scene representation | **NOT APPLICABLE** | Biểu diễn trạng thái trung gian của mạng (intermediate activation / persistent state) |
| Path planning / Navigation | **NOT APPLICABLE** | Chính sách thực thi (exit / continue / persist) theo thời gian — bài toán quyết định tuần tự |
| Traversability / Occupancy | **NOT APPLICABLE** | "Energy feasibility map": tập các block có thể hoàn thành với năng lượng hiện có |
| Collision avoidance | **NOT APPLICABLE** | Tránh *non-progress* (thrashing) và tránh sai nhất quán do write-after-read trên NVM |
| FPS / planning latency | Thay thế | Thời gian hoàn thành inference (bao gồm thời gian sạc), overhead ra quyết định (cycles), throughput của simulator |

---

## 3. Research Problem & Gap

### 3.1 Research problem (phát biểu hình thức)

Một nút cảm biến không pin chạy mạng nơ-ron $F$ chia thành $K$ block $\{B_1,\dots,B_K\}$, mỗi block $k$ có một exit head $f_k$. Nguồn năng lượng là chuỗi công suất thu hoạch ngẫu nhiên $P_h(t)$ nạp vào tụ $C$. Thiết bị bật khi $V \ge V_{on}$, tắt khi $V \le V_{off}$. SRAM mất dữ liệu khi tắt; NVM (FRAM) giữ dữ liệu nhưng ghi tốn năng lượng.

Mục tiêu: với mỗi mẫu đầu vào $x$ (hoặc chuỗi mẫu), chọn **khi nào dừng (exit)**, **lưu gì vào NVM (persist)**, và **tiếp tục hay bỏ qua** để tối đa hóa số dự đoán đúng hoàn thành trong deadline trên mỗi đơn vị năng lượng thu được.

### 3.2 Research gap sau khi kiểm tra literature [INFERENCE, cần xác nhận ở Gate M1]

| Nhóm công trình | Đã làm gì | Còn thiếu gì |
|---|---|---|
| Checkpoint/runtime (Chinchilla OSDI'18, Coala TOSN'20, WARio PLDI'22, MII TCAD'24) | Giảm số lần/điểm checkpoint, chọn cơ chế checkpoint theo layer và điều kiện năng lượng | Coi trạng thái cần lưu là **cố định** (activation đầy đủ); không học cách nén trạng thái |
| Inference trên FRAM (SONIC/TAILS ASPLOS'19, HAWAII 2020) | Tiến trình được bảo toàn bằng tính toán trực tiếp trên NVM, loop continuation | Chi phí ghi NVM vẫn tỷ lệ với kích thước activation; không có early-exit theo năng lượng |
| Multi-exit trên nguồn thu hoạch (Zygarde IMWUT'20, Wu et al. DAC'20, HarvNet MobiSys'23, DualAdaptNet DATE'25, ATM-Net arXiv'25, RL-SEP SenSys'25 poster) | Chọn exit/độ rộng/độ chính xác số theo năng lượng; có dùng RL | Chưa thấy công trình **đồng tối ưu** exit + nội dung trạng thái được lưu + chi phí ghi NVM trong cùng một mục tiêu [INFERENCE] |
| Training nhận biết gián đoạn (NExUME/DynFit arXiv 2024/25, iNAS TECS'21) | Đưa trace năng lượng vào training (dropout/quantization) hoặc vào NAS | Không dùng exit; không học biểu diễn trạng thái lưu trữ |
| Tính đúng đắn của dynamic NN khi gián đoạn ("Catch Non-determinism If You Can", ACM TECS) | Xử lý bất định do mạng động bị ngắt giữa chừng | Tập trung tính đúng, không tối ưu accuracy–energy [INFERENCE từ tiêu đề; cần đọc toàn văn] |

**Gap đề xuất (ứng viên):** *Chưa có phương pháp học — end-to-end — một biểu diễn trạng thái trung gian nén được để lưu vào NVM tại các ranh giới exit, đồng thời huấn luyện các exit theo phân phối năng lượng thực tế và ra quyết định exit/persist bằng chính sách tính đủ chi phí ghi NVM, sạc lại và deadline.*

Gap này **chưa được xác nhận**. Gate M1 sẽ giữ, thu hẹp, hoặc loại nó.

---

## 4. Research Questions & Objectives

### 4.1 Objectives

- **O1.** Xây simulator thực thi gián đoạn có hiệu chỉnh, mã nguồn mở, tái lập được.
- **O2.** Đề xuất và kiểm định Learned Persistent-State Bottleneck (PSB) để giảm chi phí ghi NVM.
- **O3.** Đề xuất Energy-Trace-Conditioned Training (ETCT) cho mạng multi-exit.
- **O4.** Đề xuất chính sách exit/persist dựa trên MDP có tính đủ chi phí, triển khai được dưới dạng lookup table nhỏ.
- **O5 (journal).** Mở rộng sang tích lũy bằng chứng theo thời gian và kiểm định tổng quát hóa giữa các loại nguồn năng lượng, loại NVM và dataset.

### 4.2 Research Questions — chuỗi RQ → Hypothesis → Method → Experiment → Metric → Evidence → Contribution

| | RQ1 — Persistence | RQ2 — Training | RQ3 — Policy | RQ4 — Temporal evidence (journal) | RQ5 — Generalization (journal) |
|---|---|---|---|---|---|
| **Câu hỏi** | Trạng thái trung gian được học-nén có giảm năng lượng ghi NVM trên mỗi inference hoàn thành so với checkpoint activation đầy đủ mà không giảm accuracy đáng kể không? | Huấn luyện exit theo xác suất thực sự dừng tại exit đó dưới trace năng lượng có tăng expected accuracy so với huấn luyện multi-exit chuẩn không? | Chính sách exit/persist giải bằng MDP có vượt policy ngưỡng năng lượng về số dự đoán đúng hoàn thành trên mJ không? | Khi thu hoạch yếu kéo dài, tích lũy posterior qua nhiều mẫu/chu kỳ có phục hồi accuracy so với exit nông độc lập không? | Lợi ích của RQ1–RQ3 có giữ nguyên khi đổi loại nguồn, kích thước tụ, loại NVM và dataset không? |
| **Hypothesis** | H1: PSB giảm bytes ghi NVM ≥ 2× với accuracy giảm ≤ 1 pp [TARGET] | H2: ETCT tăng expected accuracy ≥ 2 pp dưới trace test [TARGET] | H3: MDP policy tăng *correct-completions/mJ* ≥ 10% so với policy ngưỡng tốt nhất [TARGET] | H4: Accumulation phục hồi ≥ 50% khoảng cách accuracy giữa exit nông và exit cuối trong chế độ thu hoạch thấp, trên task có nhãn dừng theo cửa sổ [TARGET] | H5: Thứ hạng phương pháp giữ nguyên trong ≥ 80% cấu hình quét [TARGET] |
| **Method** | PSB (M3) | ETCT (M4) | MDP + value iteration, LUT (M5) | Bayesian log-posterior fusion có calibration (M6) | Ma trận quét (Mục 14) |
| **Experiment** | E3, E5 | E4, E5 | E5, E6 | E9 | E7, E8, E10 |
| **Metric** | NVM bytes/inference, E_persist, accuracy | Expected accuracy $\mathbb{E}_\tau[\text{Acc}]$, accuracy theo exit | Correct completions/mJ, ICR, latency trong deadline | Accuracy theo mức công suất thu | Kendall-τ của thứ hạng, Δ metric |
| **Expected evidence** | Đường cong rate–accuracy; bảng so với B2/B3 | Bảng so với uniform/Wu-style weighting | Pareto accuracy–energy; so với oracle | Đường accuracy vs mean harvest power | Heatmap độ nhạy |
| **Paper contribution** | C1 (conference + journal) | C2 (conference + journal) | C3 (conference + journal) | C4 (journal only) | C5 (journal only) |

**Nguyên tắc:** mọi thí nghiệm ở Mục 14 đều ánh xạ về một RQ. Thí nghiệm không ánh xạ được sẽ bị đánh dấu OPTIONAL.

---

## 5. Novelty Analysis

### 5.1 Existing methods (đã kiểm chứng tồn tại, xem Phụ lục A)

| Công trình | Venue/năm | Ý chính | Code |
|---|---|---|---|
| Chinchilla — Adaptive Dynamic Checkpointing | USENIX OSDI 2018 | Checkpoint động, an toàn | TBD |
| SONIC & TAILS — Intelligence Beyond the Edge | ACM ASPLOS 2019 | DNN inference gián đoạn trên MSP430, loop continuation trên FRAM | Có: `CMUAbstract/SONIC` |
| Coala — Dynamic Task-based Intermittent Execution | ACM TOSN 2020 | Runtime task động, gộp task | Có: `TUDSSL/Coala` |
| Zygarde | ACM IMWUT 2020 | Inference nhạy thời gian, early-exit (mandatory/optional layers) trên hệ gián đoạn | TBD |
| Wu et al. — Nonuniformly Compressed Multi-Exit NN | DAC 2020 | Multi-exit nén không đều cho thiết bị thu hoạch năng lượng | TBD |
| ePerceptive | 2020 (TU Delft Research Portal; venue cụ thể **NEEDS VERIFICATION**) | Inference "best-effort", phản ứng theo năng lượng | TBD |
| HAWAII — Everything Leaves Footprints | IEEE journal, 2020 | Intermittent inference có tăng tốc phần cứng (LEA) | Có: `EMCLab-Sinica/HAWAII_Project` |
| iNAS | ACM TECS 2021 | NAS nhận biết gián đoạn, có mô hình hiệu năng | Có: `EMCLab-Sinica/Intermittent-aware-NAS` |
| Stateful Neural Networks for Intermittent Systems | IEEE journal, 2022 (có thể là TCAD — **NEEDS VERIFICATION**) | Mạng "stateful" cho hệ gián đoạn | TBD |
| WARio | PLDI 2022 | Sinh mã giảm checkpoint do phụ thuộc WAR | TBD |
| Keep in Balance (DynBal) | ACM TECS 2023 | Tái cấu hình runtime | Có: `EMCLab-Sinica/DynBal` |
| HarvNet | ACM MobiSys 2023 | Vận hành tối ưu tài nguyên cho multi-exit DNN trên thiết bị thu hoạch | TBD |
| MII | IEEE TCAD 2024 | Chọn cơ chế checkpoint theo layer, offline profiling + online adaptation; so với Zygarde | TBD |
| NExUME / DynFit | arXiv 2408.13696 (2024–2025) | Đưa trace năng lượng vào training (dropout, quantization) | TBD |
| DualAdaptNet | DATE 2025 | Thích ứng cả width và depth theo công suất và deadline | TBD |
| ATM-Net | arXiv 2502.09822 (2025) | Adaptive termination + multi-precision cho thiết bị thu hoạch | TBD |
| RL-SEP | SenSys 2025 (poster) | Chọn exit point bằng RL | TBD |
| Catch Non-determinism If You Can | ACM TECS | Intermittent inference của dynamic NN | TBD |
| Timing-Constrained Composable Inference … RL | ACM TECS | Inference ghép được + RL cho hệ gián đoạn | TBD |
| Energy-Adaptive Checkpoint-Free Intermittent Inference | arXiv 2503.06663 (2025) | Chế độ low-power + suy luận không checkpoint | TBD |

### 5.2 Đánh giá novelty của formulation gốc

| Thành phần gốc (file nguồn) | Mức trùng lặp | Nhận định |
|---|---|---|
| Dynamic Power-Envelope Early-Exit | **Cao** — Zygarde, Wu DAC'20, HarvNet, DualAdaptNet, RL-SEP | Không đủ làm novelty chính |
| MDP chọn exit | **Trung bình–cao** — RL-SEP, TECS "Timing-Constrained … RL" | Chỉ mới khi MDP có thành phần chi phí persist học được |
| Zero-copy heterogeneous memory partitioning | **Cao** — SONIC/TAILS, HAWAII vận hành trực tiếp trên FRAM; MSP430FR vốn có không gian địa chỉ thống nhất | Mức kỹ thuật hệ thống, khó chứng minh bằng simulation; loại khỏi contribution chính |
| "State-preserving" | **Trùng tên** với Stateful NN (IEEE, 2022) | Đổi thuật ngữ trong title để tránh nhầm lẫn |
| Temporal Bayesian Evidence Accumulator | **Chưa thấy trùng trực tiếp** [INFERENCE] | Giữ làm contribution journal (RQ4) |

### 5.3 Core novelty đề xuất

| Loại novelty (theo danh sách prompt) | Thành phần | Trạng thái |
|---|---|---|
| New representation | **Learned Persistent-State Bottleneck (PSB)**: trạng thái lưu NVM là biến tiềm ẩn nén được học, không phải activation thô | Ứng viên chính |
| New loss function | **Rate-aware persistence loss** (task + distortion + λ·bytes) và **ETCT loss** (trọng số exit = xác suất dừng thực tế dưới phân phối trace) | Ứng viên chính |
| Uncertainty-aware decision | MDP dùng confidence đã calibrate (temperature scaling) làm biến trạng thái | Hỗ trợ |
| Real-time optimization / efficiency | Policy dạng LUT ≤ 2 KB, overhead ra quyết định mô hình hóa theo cycles | Hỗ trợ |
| Dynamic scene handling → tương đương "dynamic energy" | Tổng quát hóa giữa họ trace (cross-trace) | Journal |

**Working title đề xuất (conference):** *Learning What to Persist: Rate-Aware State Bottlenecks and Energy-Conditioned Early Exits for Intermittent TinyML*
(Đây là đề xuất, cố vấn chốt tại Gate M1.)

### 5.4 Protocol kiểm tra novelty (Sprint 1–2)

- **Nguồn:** IEEE Xplore, ACM DL, arXiv, Google Scholar; proceedings 2019–2026 của SenSys, MobiSys, IPSN, EWSN, DAC, DATE, ICCAD, ISLPED, ESWEEK (CODES+ISSS/EMSOFT/CASES), ASPLOS; journals TCAD, TECS, TOSN, IoT-J, TC, TMC.
- **Từ khóa:** `intermittent inference`, `energy harvesting DNN`, `batteryless inference`, `multi-exit intermittent`, `early exit energy harvesting`, `checkpoint compression`, `activation compression checkpoint`, `learned checkpoint`, `NVM write reduction neural network`, `stateful intermittent`, `anytime inference energy harvesting`.
- **Forward citation** của: SONIC/TAILS, Zygarde, Wu DAC'20, HarvNet, iNAS, Stateful NN.
- **Đầu ra:** bảng ≥ 30 công trình (method, venue, năm, có early-exit?, có học trạng thái lưu?, có trace năng lượng?, simulation hay hardware, code).
- **Quy tắc quyết định:** nếu tồn tại công trình đã làm **đồng thời** PSB + exit theo năng lượng + chi phí ghi NVM trong một mục tiêu → kích hoạt Kill K1 (Mục 21).

---

## 6. Feasibility Assessment

Mỗi dòng: **Evidence → Risk → Mitigation**.

| Dimension | Đánh giá | Evidence | Risk | Mitigation |
|---|---|---|---|---|
| Research Gap | 🟡 | Lĩnh vực có ≥ 15 công trình liên quan trực tiếp 2018–2025 (Mục 5.1); gap ứng viên (học trạng thái lưu) chưa thấy trong danh sách đã kiểm tra | Một công trình chưa tìm thấy đã làm điều tương tự | Protocol 5.4 + Gate M1; Plan B là RQ tổng quát hóa cross-trace |
| Novelty | 🟡 | 3/5 ý tưởng gốc trùng lặp cao (Mục 5.2); PSB + ETCT là loss/representation mới ở mức ứng viên | Reviewer coi là "kết hợp các kỹ thuật đã có" (bottleneck, early-exit, MDP) | Chứng minh bằng ablation rằng *đồng tối ưu* tốt hơn tổng các phần tối ưu riêng (E6) |
| Algorithmic Feasibility | 🟢 | Mọi thành phần là kỹ thuật chuẩn: CNN nhỏ, bottleneck, quantization, value iteration trên không gian rời rạc nhỏ | Không gian trạng thái MDP phình to | Rời rạc hóa thô (≤ 16 mức năng lượng × K × 8 mức confidence); kiểm tra độ nhạy theo số bin |
| Model Feasibility | 🟢 | Backbone TinyML (DS-CNN, MobileNetV1-0.25, 1D-CNN) được dùng rộng rãi trong TinyML benchmark | Ràng buộc bộ nhớ MSP430 (8 KB SRAM) làm model quá nhỏ, accuracy thấp | Mô hình bộ nhớ phân tích (activation tile) thay vì ép toàn mạng vào SRAM; giới hạn tham số ≤ 256 KB FRAM |
| Dataset | 🟢 | GSC, VWW, CWRU công khai; trace ánh sáng trong nhà EnHANTs công khai (IEEE DataPort, DOI 10.15783/C77P47) [VERIFIED] | Trace EnHANTs lấy mẫu 30 s — quá thô so với chu kỳ nguồn cỡ ms | Up-sampling có nhiễu ngẫu nhiên có kiểm soát + synthetic Markov-modulated traces; khai báo rõ là giả định |
| Simulation | 🟡 | Có tiền lệ mô hình hiệu năng cho intermittent (iNAS-PMod; ScEpTIC; BatterylessSim) | Simulator thiếu độ trung thực → reviewer nghi ngờ mọi kết quả | Hiệu chỉnh tham số từ datasheet + số liệu đã công bố; kiểm tra tái hiện xu hướng của ≥ 2 baseline; phân tích độ nhạy ±50% tham số chi phí; mở mã nguồn |
| Computational | 🟢 | Model ≤ ~10⁵–10⁶ tham số; simulator chủ yếu CPU/NumPy, có thể vector hóa trên GPU | GPU cụ thể tại trường chưa biết [TBD] | Pilot benchmark Sprint 2 (Mục 11.3) |
| Baseline | 🟡 | Code gốc của SONIC, HAWAII, iNAS, DynBal, Coala là firmware MSP430 | Không chạy được trực tiếp trong simulator Python | Tái hiện **mô hình chi phí** của baseline trong simulator; kiểm tra với số liệu tương đối đã công bố; ghi rõ "re-implemented in simulation" |
| Real-time | 🟢 | Real-time ở đây = hoàn thành inference trong deadline dưới trace + overhead quyết định nhỏ; đo được trong simulator | Overhead policy trên MCU bị đánh giá thấp | Mô hình overhead theo cycles của LUT lookup; báo cáo tỷ lệ overhead/energy |
| Experimental | 🟡 | Ma trận 10 thí nghiệm ánh xạ đủ 5 RQ (Mục 14) | Một sinh viên, khối lượng lớn | Ưu tiên E1–E6 cho conference; E7–E10 cho journal |
| Conference ≤ 6M | 🟡 | ISLPED 2026 có deadline 09/03/2026; EWSN 2026 có deadline 24/04/2026 [VERIFIED qua CFP/nguồn thứ cấp] → các kỳ 2027 dự kiến rơi vào tháng 6–7 của roadmap | Deadline 2027 chưa công bố; rank lấy từ nguồn thứ cấp | Kiểm tra lại tại Sprint 8; có 2 venue dự phòng |
| Journal ≤ 12M | 🟡 | IoT-J, TCAD, TMC, TC đều Q1 theo SJR 2025 [VERIFIED] | Reviewer Q1 đòi đo trên phần cứng | Calibrated simulator + sensitivity + open-source; nếu lab đã sẵn có board MSP430, cân nhắc spot-check (ngoài phạm vi roadmap — quyết định của cố vấn) |

**Kết luận sơ bộ:** 🟡 **FEASIBLE WITH CONDITIONS** — điều kiện: qua Gate M1 (novelty), Gate M3 (simulator calibration), và chấp nhận đổi venue.

---

## 7. Proposed Algorithm

### 7.1 Kiến trúc tổng thể (thay cho pipeline 3DGS của template)

```
Public dataset (GSC / VWW / CWRU)          Energy traces (EnHANTs indoor light,
            │                               synthetic Markov-modulated, TBD outdoor/kinetic)
            ▼                                               │
  M2  Multi-exit TinyML backbone (INT8, K exits)            ▼
            │                                   M1  Calibrated Intermittent-Execution
            ▼                                       Simulator (capacitor + MCU/NVM cost model)
  M3  Persistent-State Bottleneck at exit boundaries        │
            │                                               │
            ▼                                               │
  M4  Energy-Trace-Conditioned Training  ◄──── exit-probabilities p_k(τ) ────┘
            │
            ▼
  M5  Exit / Persist / Skip policy (MDP → value iteration → LUT ≤ 2 KB)
            │
            ▼
  [Journal] M6  Temporal evidence accumulation across samples / power cycles
            │
            ▼
  Evaluation in simulator: accuracy, ICR, correct-completions/mJ, NVM bytes, latency-in-deadline
```

### 7.2 M1 — Calibrated Intermittent-Execution Simulator (IES)

- **Input:** trace công suất $P_h(t)$; đồ thị block của mạng với số phép toán và kích thước tensor; policy; tham số phần cứng dạng file cấu hình (không mua hardware).
- **Output:** với mỗi inference — thời điểm hoàn thành, exit được dùng, dự đoán, năng lượng tính toán / persist / restore / boot, bytes ghi NVM, số lần sập nguồn.
- **Mô hình năng lượng tụ:**
  $$E(t)=\tfrac12 C\big(V(t)^2-V_{off}^2\big),\qquad \frac{dE}{dt}=\eta\,P_h(t)-P_{load}(t)$$
  Thiết bị bật khi $V\ge V_{on}$, tắt khi $V\le V_{off}$ (hysteresis). $\eta$ là hiệu suất bộ chuyển đổi (tham số).
- **Mô hình chi phí block $k$:**
  $$E^{comp}_k=\sum_{o\in\mathcal{O}_k} n_o\,e_o,\qquad E^{pers}(s)=|s|_{B}\,e^{w}_{NVM}+e^{ovh}_{w},\qquad E^{rest}(s)=|s|_{B}\,e^{r}_{NVM}+E_{boot}$$
  $e_o$: năng lượng mỗi loại phép toán (MAC trên SRAM/FRAM, LEA nếu mô hình hóa), $|s|_B$ số byte. **Mọi tham số lấy từ datasheet MSP430FR5994 và số liệu đã công bố; trạng thái NEEDS VERIFICATION cho đến Sprint 3.**
- **Thuật toán:** mô phỏng sự kiện rời rạc theo bước thời gian $\Delta t$ (tham số, mặc định 1 ms [TBD]); vector hóa theo lô trace bằng NumPy/PyTorch để chạy song song hàng nghìn trace.
- **Kiểm chứng simulator (bắt buộc):** (a) unit test bảo toàn năng lượng; (b) so với nghiệm giải tích cho trace hằng; (c) tái hiện **xu hướng tương đối** đã công bố của ≥ 2 baseline (ví dụ thứ tự overhead giữa các chiến lược checkpoint); (d) phân tích độ nhạy ±50% từng tham số.
- **Complexity:** $O(N_{trace}\cdot T/\Delta t)$ bước; bộ nhớ $O(N_{trace})$.

### 7.3 M5 — Chính sách Exit/Persist/Skip (MDP)

- **Trạng thái:** $x=(\,\bar E,\ \bar P_h,\ k,\ \bar c_k,\ \rho\,)$ — mức năng lượng rời rạc, mức công suất thu ước lượng, block hiện tại, confidence đã calibrate rời rạc hóa, cờ "đã persist".
- **Hành động:** `CONT` (chạy block $k{+}1$), `EXIT` (xuất $\hat y_k$), `PERSIST` (ghi $s_k$ vào NVM rồi ngủ chờ $V_{on}$), `SKIP` (bỏ mẫu hiện tại).
- **Phần thưởng:**
  $$r=\widehat{\mathrm{Acc}}_k(\bar c_k)\cdot\mathbb 1[t\le D]\;-\;\mu\,E_{used}$$
  $\widehat{\mathrm{Acc}}_k(c)$ là ánh xạ confidence→accuracy trên tập validation; $D$ là deadline; $\mu$ là trọng số năng lượng.
- **Chuyển trạng thái:** $\bar P_h$ theo chuỗi Markov ước lượng từ trace dev; $\bar E$ cập nhật bằng mô hình M1; $\bar c_{k+1}$ theo phân phối thực nghiệm có điều kiện $p(\bar c_{k+1}\mid\bar c_k)$ đo trên validation.
- **Giải:** value iteration trên không gian rời rạc; xuất LUT $\pi(\bar E,\bar P_h,k,\bar c)\in\mathcal A$.
- **Kích thước LUT:** ví dụ $16\times4\times K\times8$ mục; với $K=4$ là 2048 mục × 2 bit = 512 byte [INFERENCE]. Mục tiêu ≤ 2 KB FRAM.
- **Complexity:** $O(|S|\,|A|\,|S'|\cdot I)$ offline; online $O(1)$ (một lần tra bảng).

### 7.4 M6 — Temporal Evidence Accumulation (journal)

- Áp dụng cho task có nhãn ổn định trong một cửa sổ (tình trạng máy CWRU, hoạt động người). **Không áp dụng** cho keyword spotting (sự kiện ngắn) — giới hạn này phải ghi rõ trong paper.
- $$\log p(y\mid x_{1:t})\propto\sum_{i=1}^{t} w_i\,\log p_{k_i}(y\mid x_i;\,T_{k_i})$$
  $T_k$ là nhiệt độ calibrate của exit $k$; $w_i$ giảm theo thời gian (hệ số quên $\gamma$). Xuất kết quả khi $\max_y p(y\mid x_{1:t})\ge\gamma_{safe}$.
- Trạng thái lưu NVM chỉ là vector log-posterior $|\mathcal Y|$ phần tử — nhỏ, persist rẻ.

---

## 8. Proposed Model Architecture

### 8.1 M2 — Multi-exit TinyML backbone

| Dataset | Backbone | Input | K exits | Ghi chú |
|---|---|---|---|---|
| Google Speech Commands v2 (12 lớp) | DS-CNN (small) | MFCC 49×10 [TBD theo preprocessing chuẩn] | 3–4 | Exit sau mỗi 1–2 khối depthwise-separable |
| Visual Wake Words | MobileNetV1 α=0.25 | 96×96 (grayscale theo ràng buộc file nguồn "thị giác đơn sắc") | 3–4 | Ảnh đơn sắc làm giảm accuracy — cần báo cáo |
| CWRU bearing | 1D-CNN 4–6 lớp | Cửa sổ tín hiệu rung (độ dài TBD) | 3 | Chia split theo tải (load) để kiểm tra tổng quát hóa |

- **Exit head:** GAP → FC (≤ 1 lớp ẩn). Chi phí exit head được tính vào $E^{comp}$.
- **Quantization:** INT8 QAT (PyTorch). Ràng buộc: tổng trọng số ≤ 256 KB (FRAM), activation tile ≤ SRAM khả dụng theo mô hình bộ nhớ phân tích.

### 8.2 M3 — Persistent-State Bottleneck (PSB)

- **Input:** activation $h_k\in\mathbb R^{C_k\times H_k\times W_k}$ tại ranh giới exit $k$.
- **Output:** trạng thái lưu $s_k=Q_b\big(g_k(h_k)\big)$ và trạng thái khôi phục $\hat h_k=r_k(s_k)$.
- **Kiến trúc:** $g_k$ = conv 1×1 giảm kênh $C_k\to m_k$ + (tùy chọn) stride pooling; $Q_b$ = lượng tử b-bit có straight-through estimator; $r_k$ = conv 1×1 $m_k\to C_k$ (+ upsampling nếu có pooling). Kênh được chọn bằng gate học được $z\in\{0,1\}^{m_k}$ (hard-concrete relaxation).
- **Loss:**
  $$\mathcal L_{PSB}=\sum_k\Big[\mathrm{CE}\big(f_{>k}(\hat h_k),y\big)+\alpha\,\lVert h_k-\hat h_k\rVert_2^2\Big]+\lambda_R\sum_k R(s_k),\qquad R(s_k)=\frac{b\cdot \lVert z\rVert_0\cdot H'_kW'_k}{8}\ \text{(bytes)}$$
- **Chi phí suy luận:** PSB chỉ chạy khi policy chọn `PERSIST`; khi `CONT` đi thẳng $h_k$. Overhead $g_k,r_k$ được tính vào simulator.

### 8.3 M4 — Energy-Trace-Conditioned Training (ETCT)

- $$\mathcal L_{ETCT}=\mathbb E_{\tau\sim\mathcal T_{dev}}\sum_{k=1}^{K}p_k(\tau;\pi,\theta)\,\mathrm{CE}\big(f_k(x),y\big)+\beta\sum_{k<K}\mathrm{KL}\big(f_K(x)\,\Vert\,f_k(x)\big)$$
  $p_k$ = xác suất inference dừng ở exit $k$ dưới trace $\tau$ và policy $\pi$, ước lượng bằng Monte Carlo trong M1.
- **Training procedure (xen kẽ):**
  1. Train multi-exit chuẩn (trọng số đều) → $\theta_0$.
  2. Lặp $i=1..I$ (I ≈ 3–5 [TBD]): (a) calibrate confidence; (b) giải policy $\pi_i$ (M5); (c) mô phỏng → $p_k^{(i)}$; (d) fine-tune $\theta_i$ với $\mathcal L_{ETCT}+\mathcal L_{PSB}$.
  3. Dừng khi expected accuracy trên validation traces tăng < 0.2 pp giữa 2 vòng [TARGET].
- **Inference procedure:** chạy block → nếu tới ranh giới exit: tính $c_k$ → tra LUT → hành động.

### 8.4 Hyperparameters (khởi tạo, sẽ tinh chỉnh trên validation)

| Tham số | Khởi tạo | Dải quét |
|---|---|---|
| Optimizer | AdamW | — |
| LR | 1e-3 (cosine) | 3e-4 – 3e-3 |
| Batch | 128 | 64 – 256 |
| Epoch (từ đầu / fine-tune) | TBD theo pilot | — |
| $\alpha$ (distortion) | 0.1 | 0.01 – 1 |
| $\lambda_R$ (rate) | quét | log-scale 6 mức |
| b (bit của $s_k$) | 8 | {2, 4, 8} |
| $\beta$ (self-distill) | 0.5 | 0 – 1 |
| $\mu$ (năng lượng trong reward) | quét | 5 mức |
| Seeds | 5 | cố định {0..4} |

**Data augmentation:** GSC — time shift, background noise (theo recipe chuẩn); VWW — flip, crop; CWRU — random crop cửa sổ, nhiễu Gauss. Không thêm augmentation không phục vụ RQ.

---

## 9. Dataset & Simulation Strategy

### 9.1 Dữ liệu tác vụ (task data)

| Vai trò | GSC v2 | VWW | CWRU |
|---|---|---|---|
| Development | train split chính thức | train split theo script tạo VWW | Tải 0–1 HP |
| Validation (tuning, calibration, ước lượng $p(c_{k+1}\mid c_k)$) | validation list chính thức | minival | Tải 2 HP (một phần) |
| Test | test list chính thức | val split chuẩn | Tải 3 HP (giữ riêng) |
| Cross-condition | — | — | Train 0–2 HP → test 3 HP |

- Tất cả dataset là công khai [FILE + INFERENCE]. Giấy phép và script tải chính thức cần xác nhận trong Sprint 1 [TBD].
- Split của CWRU theo tải là đề xuất [INFERENCE]; cần kiểm tra số mẫu mỗi tải đủ cho 5 seeds.

### 9.2 Dữ liệu năng lượng (energy traces)

| Họ trace | Nguồn | Trạng thái | Vai trò |
|---|---|---|---|
| T1 — Ánh sáng trong nhà | CRAWDAD `columbia/enhants`, IEEE DataPort, DOI 10.15783/C77P47; 6 vị trí cố định + 4 kịch bản di động; lấy mẫu 30 s; 08/2009–11/2010 | [VERIFIED] | Dev: 3 vị trí; Val: 1 vị trí; Test: 2 vị trí giữ riêng |
| T2 — Synthetic Markov-modulated | Sinh bằng code: chuỗi Markov 2–4 trạng thái công suất, tham số hóa bởi công suất trung bình và độ bùng nổ (burstiness) | Tự xây | Quét độ nhạy; stress test "thu hoạch yếu kéo dài" |
| T3 — Mặt trời ngoài trời | Ứng viên: NREL MIDC | [TBD] — kiểm tra quyền truy cập, độ phân giải | Cross-trace (journal) |
| T4 — Động năng (kinetic) | Ứng viên: dữ liệu của Gorlatova et al. (SIGMETRICS 2014); bộ dữ liệu CSIRO "Solar and Kinetic Energy Harvesting Dataset for Human…" | [TBD] — trang CSIRO cần JavaScript, chưa đọc được nội dung | Cross-modality (journal) |
| T5 — RF | Chưa xác định nguồn công khai | [TBD] | Nếu không có: dùng T2 với tham số RF từ literature |

**Giả định chuyển đổi irradiance → công suất:** $P_h=\eta_{pv}\,A_{pv}\,G(t)$ với $G$ tính bằng µW/cm². Diện tích pin $A_{pv}$ và hiệu suất $\eta_{pv}$ là tham số quét, không phải hardware cần mua.

**Giả định up-sampling:** trace 30 s được nội suy xuống $\Delta t$ ms, cộng nhiễu nhân tính có phương sai là tham số quét. Đây là giả định lớn nhất về độ trung thực; phải báo cáo trong phần *Threats to validity*.

### 9.3 Ground truth của simulation

| Ground truth | Cách tạo |
|---|---|
| Nhãn đúng | Từ dataset |
| Accuracy liên tục (continuous-power upper bound) | Chạy mạng không gián đoạn |
| Oracle policy | Quy hoạch động offline biết trước toàn bộ trace (upper bound cho RQ3) |
| Năng lượng thực tế mỗi hành động | Tính từ mô hình chi phí M1 |

### 9.4 Simulation có trả lời được RQ không?

| RQ | Cần gì từ simulation | Đáp ứng? |
|---|---|---|
| RQ1 | Chi phí ghi NVM theo byte, số lần persist | Có — nếu $e^w_{NVM}$ được hiệu chỉnh |
| RQ2 | Phân phối điểm dừng dưới trace | Có |
| RQ3 | Tương tác năng lượng–thời gian–deadline | Có |
| RQ4 | Chuỗi mẫu liên tiếp + thu hoạch yếu kéo dài | Có (T1 + T2) |
| RQ5 | Nhiều họ trace, nhiều loại NVM | Một phần — phụ thuộc T3–T5 [TBD]; NVM khác (STT-MRAM, ReRAM) chỉ bằng tham số từ literature |

---

## 10. Baseline Strategy

Các loại baseline của template (classical / learning-based / semantic-aware / 3DGS+planning) được ánh xạ sang tương đương của đề tài.

| ID | Loại (tương đương) | Method | Paper | Code | Độ khó tái hiện | Metric | Hạn chế dự kiến | Vai trò |
|---|---|---|---|---|---|---|---|---|
| B0 | Upper bound | Continuous power, exit cuối | — | Tự viết | Thấp | Accuracy | Không thực tế | Trần accuracy |
| B1 | Classical, không checkpoint | Restart-from-scratch | — | Tự viết | Thấp | ICR, µJ | Không tiến triển khi năng lượng < 1 inference | Sàn |
| B2 | Classical checkpoint (task/tile) | Tile-level progress trên FRAM, kiểu SONIC/TAILS | ASPLOS 2019 | `CMUAbstract/SONIC` (firmware) | Trung bình — dựng lại mô hình chi phí | NVM bytes, µJ, ICR | Ghi NVM tỷ lệ với activation | Đối chứng chính cho RQ1 |
| B3 | Classical JIT checkpoint | Checkpoint khi điện áp chạm ngưỡng (kiểu Hibernus/Chinchilla) | OSDI 2018 | TBD | Trung bình | như B2 | Phụ thuộc ngưỡng; ghi toàn trạng thái | Đối chứng RQ1 |
| B4 | Learning-based early-exit | Multi-exit + ngưỡng năng lượng (kiểu Zygarde / Wu DAC'20) | IMWUT 2020; DAC 2020 | TBD | Trung bình | Accuracy, ICR, correct/mJ | Không học trạng thái lưu; heuristic | Đối chứng chính cho RQ2–RQ3 |
| B5 | Resource-aware multi-exit SOTA | HarvNet | MobiSys 2023 | TBD | Cao nếu không có code | như B4 | TBD sau khi đọc toàn văn | SOTA comparison |
| B6 | Intermittency-aware training | NExUME/DynFit | arXiv 2024/25 | TBD | Trung bình | Expected accuracy | Không dùng exit | Đối chứng RQ2 |
| B7 | Width+depth adaptive | DualAdaptNet | DATE 2025 | TBD | Trung bình | Accuracy, deadline | TBD | Optional |
| B8 | Oracle | DP biết trước trace | — | Tự viết | Thấp | correct/mJ | Không khả thi thực tế | Trần của policy |

**Backup rule:** baseline nào tái hiện quá 2 tuần mà chưa khớp xu hướng công bố → thay bằng phiên bản "-like" đơn giản hóa (B4-like, B5-like) và ghi rõ trong paper. Tối thiểu cho conference: **B0, B1, B2, B4, B8**. Cho journal: thêm **B3, B5 hoặc B6**.

---

## 11. Computational Plan

### 11.1 Ước lượng [INFERENCE — sẽ thay bằng số đo pilot]

| Hạng mục | Ước lượng | Căn cứ |
|---|---|---|
| VRAM training | < 4 GB mỗi run | Model ≤ 10⁶ tham số, batch 128, input nhỏ |
| Thời gian train 1 model từ đầu | vài chục phút – vài giờ tùy dataset | TinyML models; VWW lớn nhất |
| Số run conference | 3 dataset × ~6 cấu hình × 5 seeds ≈ 90 run | E1–E6 |
| Số run journal thêm | ≈ 150–250 run | E7–E10 + quét tham số |
| Simulation | CPU-bound; vector hóa 10³–10⁴ trace/lô | Mỗi mô phỏng gồm các phép cộng/so sánh đơn giản |
| Tổng GPU-hours | ~150–400 GPU-h trong 12 tháng | Chưa tính lỗi/chạy lại; cần xác nhận bằng pilot |

### 11.2 Thông tin còn thiếu [TBD]

- Model GPU, VRAM, số GPU, chính sách chia sẻ/thời lượng đặt chỗ tại trường. **Kiểm chứng:** Sprint 1, chạy `nvidia-smi` và hỏi quản trị lab.
- Số nhân CPU của máy chạy simulator. **Kiểm chứng:** Sprint 1.

### 11.3 Pilot benchmark (Sprint 2)

```
GSC only, DS-CNN, 2 exits
      ↓
10% train data, 5 epochs
      ↓
Đo: VRAM đỉnh, thời gian/epoch, thời gian inference/batch
      ↓
Simulator: 100 trace T2 × 60 s mô phỏng
      ↓
Đo: thời gian mô phỏng, bộ nhớ
      ↓
Ngoại suy tuyến tính → ngân sách GPU-h cho E1–E10
      ↓
Nếu > 60% quỹ GPU khả dụng: giảm seeds (5→3 cho quét phụ), giảm dataset (bỏ VWW khỏi quét)
```

---

## 12. Real-time Simulation Plan

"Real-time" trong đề tài này không phải FPS camera. Nó gồm ba tầng, đều đo trong simulation.

| Tầng | Định nghĩa | Target |
|---|---|---|
| R1 — Inference latency dưới gián đoạn | Thời gian từ khi mẫu đến tới khi có dự đoán, **gồm thời gian sạc** | Tỷ lệ hoàn thành trong deadline $D$; $D\in\{1, 5, 30\}$ s [TBD theo task] |
| R2 — Overhead quyết định trên MCU | Số cycle của policy (tra LUT + tính confidence) | ≤ 1% năng lượng của block nhỏ nhất [TARGET] |
| R3 — Throughput simulator | Số trace-giây mô phỏng / giây thực | Đủ để chạy ma trận E1–E10 trong ngân sách GPU/CPU |

**Phân rã độ trễ (thay cho $T_{semantic}+T_{3DGS}+T_{planning}$):**
$$T_{total}=T_{comp}+T_{policy}+T_{persist}+T_{restore}+T_{boot}\cdot N_{fail}+T_{recharge}$$

**Nếu không đạt:** giảm số exit; LUT thô hơn; bỏ PSB ở exit đầu; INT4 cho $s_k$; gộp block để giảm số lần ra quyết định; policy ngưỡng (fallback).

**Thời điểm kiểm tra:** Real-time Gate cuối tháng 3 (Sprint 6), không để đến cuối dự án.

---

## 13. Evaluation Framework

### 13.1 Metrics theo RQ

| Metric | Định nghĩa | RQ |
|---|---|---|
| Accuracy / F1 | Trên các inference hoàn thành | RQ1–5 |
| Expected accuracy $\mathbb E_\tau[\mathrm{Acc}]$ | Trung bình trên tập trace test, tính cả inference không hoàn thành là sai | RQ2 |
| ICR — Inference Completion Rate | Số inference hoàn thành / số mẫu đến | RQ3 |
| Correct-completions per mJ | Số dự đoán đúng hoàn thành trong deadline / năng lượng thu được | RQ3 (metric chính) |
| µJ / valid inference | Theo file nguồn | RQ1, RQ3 |
| NVM bytes written / inference | Tổng byte ghi NVM | RQ1 |
| $E_{persist}$ share | Tỷ lệ năng lượng dành cho persist+restore | RQ1 |
| Latency-in-deadline | R1 | RQ3 |
| Policy overhead | R2 | RQ3 |
| Accuracy drop vs continuous | B0 − method | RQ1–2 |
| Kendall-τ ranking stability | Độ ổn định thứ hạng giữa cấu hình | RQ5 |
| Footprint | Bytes FRAM (weights + LUT + state), bytes SRAM đỉnh (phân tích) | Ràng buộc |

### 13.2 Chuẩn thống kê

- 5 seeds cho mọi cấu hình chính; báo cáo mean ± 95% CI.
- So sánh cặp (cùng seed, cùng trace): paired t-test hoặc Wilcoxon signed-rank, $p<0.05$, kèm effect size (Cohen's d) — theo chuẩn trong file nguồn [FILE].
- Hiệu chỉnh đa kiểm định (Holm) khi so nhiều baseline.
- Không chọn trace/seed sau khi xem kết quả test; split trace cố định từ Sprint 3.

---

## 14. Experiment Matrix

| ID | Tên | RQ | Dataset | Trace | Methods | Metric chính | Phase | Bắt buộc cho |
|---|---|---|---|---|---|---|---|---|
| E1 | Simulator validation | (nền tảng) | — | hằng + T2 | B1, B2, B3 | Sai số so với giải tích; thứ tự overhead | S3–S6 | Conf + Journal |
| E2 | Baseline reproduction | RQ1–3 | GSC | T1, T2 | B0, B1, B2, B4 | Mọi metric | S4–S6 | Conf |
| E3 | PSB rate–accuracy | RQ1 | GSC, CWRU (+VWW) | T1 | PSB vs B2/B3 | NVM bytes, Acc | S6–S8 | Conf |
| E4 | ETCT vs uniform/Wu-style weighting | RQ2 | GSC, CWRU, VWW | T1 | ETCT vs multi-exit chuẩn | $\mathbb E_\tau[\mathrm{Acc}]$ | S7–S9 | Conf |
| E5 | Main comparison | RQ1–3 | 3 dataset | T1 test | Full vs B0–B4, B8 | correct/mJ, ICR, Acc | S9–S10 | Conf |
| E6 | Ablation | RQ1–3 | GSC, CWRU | T1 | Full; −PSB; −ETCT; −MDP (ngưỡng); PSB+ETCT tối ưu riêng rẽ | correct/mJ | S9–S10 | Conf |
| E7 | Parameter sensitivity | RQ5 | GSC | T2 | Full vs B2, B4 | Δ metric, Kendall-τ | S13–S17 | Journal |
| E8 | Cross-trace / cross-modality generalization | RQ5 | GSC, CWRU | Train T1 → test T2/T3/T4 | Full vs B4, B5/B6 | $\mathbb E_\tau[\mathrm{Acc}]$ | S15–S19 | Journal |
| E9 | Temporal evidence accumulation | RQ4 | CWRU (+ HAR TBD) | T2 low-power | M6 vs exit độc lập | Acc vs mean $P_h$ | S13–S17 | Journal |
| E10 | NVM technology & capacitor sweep | RQ5 | GSC | T1 | Full vs B2 | $E_{persist}$ share | S17–S19 | Journal |

Tham số quét của E7: $C$ (dung lượng tụ), $V_{on}/V_{off}$, $e^w_{NVM}$ (±50%), công suất thu trung bình, burstiness, deadline $D$, số exit $K$.

---

## 15. 12-Month Roadmap

### 15.1 Backward planning (từ publication ngược về research)

```
Q1 Journal (IoT-J), nộp S25 — 19/09/2027
  ↑ cần: C1–C3 đã có ở conference + ít nhất 1 contribution thuật toán mới (C4) + đánh giá tổng quát hóa (C5)
  ↑ cần: E7–E10 hoàn tất, thống kê đầy đủ, gói tái lập
  ↑ cần: M6 (temporal accumulation) + trace đa dạng T2–T4 + baseline B5/B6
Conference (ISLPED 2027 hoặc EWSN 2027), nộp S11–S14
  ↑ cần: E1–E6 hoàn tất, 3 dataset, 5 seeds
  ↑ cần: PSB v2 + ETCT v1 + MDP policy v1
  ↑ cần: simulator hiệu chỉnh (E1) + baseline B0–B4, B8 (E2)
  ↑ cần: multi-exit backbone + split dữ liệu/trace cố định
  ↑ cần: gap & novelty đã xác nhận (Gate M1)
```

Mọi task trong Mục 16 ghi rõ evidence nó tạo ra. Task không tạo evidence cho paper đã bị loại hoặc đánh dấu OPTIONAL.

### 15.2 Các phase

| Phase | Tháng | Sprint | Mục tiêu | Evidence tạo ra | Gate kết thúc |
|---|---|---|---|---|---|
| P1 — Gap, novelty & nền tảng simulation | M1–M2 | S1–S4 | Xác nhận novelty; simulator v0; baseline B1–B3; backbone | Bảng literature; E1 phần 1 | Gate M1 (S2) |
| P2 — Prototype end-to-end | M3 | S5–S6 | Pipeline chạy trọn; PSB v1; policy ngưỡng; kiểm tra real-time | E1 hoàn tất; E2 | Prototype + Real-time Gate (S6) |
| P3 — Proposed method | M4–M5 | S7–S9 | MDP policy; PSB v2; ETCT | E3, E4, E5 sơ bộ | Conference Research Gate (S8) |
| P4 — Conference experiments & paper | M5–M6 | S9–S12 | E5, E6 đầy đủ; bản thảo; nộp | Bảng chính, ablation | Conference Readiness (S10), Conference Milestone (S12) |
| P5 — Journal extension | M7–M8 | S13–S16 | M6; trace mới; baseline SOTA thêm | E9; E8 phần đầu | Journal Extension Gate (S16) |
| P6 — Large-scale experiments | M9–M10 | S17–S20 | E7, E8, E10; gói tái lập | Heatmap độ nhạy; tổng quát hóa | Journal Experiment Gate (S20) |
| P7 — Journal paper | M11–M12 | S21–S26 | Viết, mock review, nộp | Bản thảo Q1 | Journal Milestone (S25) |

---

## 16. 26 Biweekly Sprints

### 16.0 Cấu trúc repository (tham chiếu cho mọi sprint)

```
ipi/                                  # Intermittent Persistent Inference
├── configs/{data,hw,train,sim}/*.yaml   # hw/msp430fr5994.yaml: mỗi tham số ghi nguồn
├── data/        gsc.py  vww.py  cwru.py  traces.py
├── sim/         capacitor.py  mcu_cost.py  executor.py  executor_batch.py  trace_synth.py
├── models/      backbones.py  multiexit.py  psb.py  quant.py
├── train/       train_multiexit.py  train_psb.py  etct.py
├── policy/      threshold.py  mdp.py  value_iteration.py  oracle.py  lut_export.py  overhead.py
├── baselines/   restart.py  tile_ckpt.py  jit_ckpt.py  energy_threshold_exit.py  harvnet_like.py
├── temporal/    accumulate.py            # journal
├── eval/        metrics.py  calibration.py  stats.py  run_matrix.py  plots.py
├── tests/       test_*.py (pytest, chạy trong CI)
└── reports/     tech_report_*.md  figures/  tables/
```

Quy ước: mọi thí nghiệm chạy bằng `eval/run_matrix.py --exp E#` với config cố định; log kèm git hash, seed, trace split.

---

### Sprint 1 — 05/10 → 18/10/2026 (M1)
- **Objective:** Bắt đầu kiểm tra novelty; dựng môi trường; xác nhận tài nguyên tính toán.
- **Research:** Protocol 5.4 bước 1 — ≥ 20 công trình vào bảng; đọc sâu SONIC/TAILS, Zygarde, Wu DAC'20, HarvNet. Xác minh baseline "IIE" (P5).
- **Algorithm/Model:** Viết `problem_formulation.md` (Mục 3.1 + mô hình năng lượng Mục 7.2) — 2 trang toán.
- **Coding:** Khởi tạo repo `ipi/`, CI pytest. `data/gsc.py: GSCDataset(split) → (mfcc[49×10], label)`; phụ thuộc: torchaudio. `data/traces.py: EnhantsLoader.load(location_id) → np.ndarray[N,2] (t, µW/cm²)`. Test: số mẫu mỗi split khớp danh sách chính thức; trace không có NaN.
- **Simulation:** Chưa chạy; tải dữ liệu GSC, EnHANTs.
- **Expected results:** Bảng literature v0; thông số GPU/CPU tại trường được ghi nhận.
- **Milestone:** Môi trường sẵn sàng.
- **Deliverables:** `lit_table_v0.csv`, `problem_formulation.md`, repo commit đầu, `compute_inventory.md`.
- **Risk & Plan B:** Không truy cập được IEEE DataPort → dùng T2 synthetic cho tới khi có T1.

### Sprint 2 — 19/10 → 01/11/2026 (M1)
- **Objective:** Kết luận novelty; đo pilot tính toán.
- **Research:** Hoàn tất ≥ 30 công trình + forward citation; viết **novelty memo** (gap, novelty, overlap, quyết định).
- **Algorithm/Model:** Chốt RQ1–RQ3 và các [TARGET] H1–H3 với cố vấn.
- **Coding:** `sim/capacitor.py: CapacitorModel(C, V_on, V_off, eta).step(P_h, P_load, dt) → (V, is_on)`; test bảo toàn năng lượng và so thời gian sạc với nghiệm giải tích. `sim/trace_synth.py: MarkovTraceGen(states, P_mean, burstiness).sample(T, dt)`. `models/backbones.py: DSCNN`; `train/train_multiexit.py` (chỉ exit cuối).
- **Simulation:** Pilot benchmark theo Mục 11.3 (GSC 10%, 5 epoch; 100 trace × 60 s).
- **Expected results:** Số đo VRAM, thời gian/epoch, throughput simulator.
- **Milestone:** 🔶 **Gate M1 — Research & Novelty (30/10/2026)**.
- **Deliverables:** `novelty_memo.pdf`, `pilot_report.md`, `lit_table_v1.csv`.
- **Risk & Plan B:** Novelty bị trùng → Kill K1 → chuyển trọng tâm sang Plan B-RQ (Mục 21).

### Sprint 3 — 02/11 → 15/11/2026 (M2)
- **Objective:** Mô hình chi phí MCU/NVM và executor v0.
- **Research:** Trích tham số năng lượng/cycle từ datasheet MSP430FR5994 và từ bài SONIC/TAILS, HAWAII; ghi nguồn từng tham số.
- **Algorithm/Model:** Chốt công thức $E^{comp}, E^{pers}, E^{rest}$.
- **Coding:** `sim/mcu_cost.py: MCUCostModel(cfg).block_cost(block_spec) → (energy_J, cycles)`; `configs/hw/msp430fr5994.yaml`. `sim/executor.py: IntermittentExecutor(model_graph, cost, cap).run(trace, policy) → RunLog`. Test: năng lượng vô hạn ⇒ kết quả trùng suy luận liên tục; trace bằng 0 ⇒ không tiến triển.
- **Simulation:** Cố định split trace T1 (3 vị trí dev / 1 val / 2 test) — **đóng băng từ đây**.
- **Expected results:** Bảng tham số có nguồn; executor chạy 1 trace.
- **Milestone:** Simulator v0.
- **Deliverables:** `hw_param_table.md`, `sim/` v0, `trace_splits.json`.
- **Risk & Plan B:** Thiếu tham số → dùng dải giá trị từ literature, đưa vào quét độ nhạy E7.

### Sprint 4 — 16/11 → 29/11/2026 (M2)
- **Objective:** Baseline B1–B3 trong simulator; backbone multi-exit.
- **Research:** Đọc sâu Chinchilla, Coala, MII để mô hình hóa đúng chi phí checkpoint.
- **Algorithm/Model:** `MultiExitNet` với K = 3–4 exit; trọng số loss đều.
- **Coding:** `baselines/restart.py` (B1), `baselines/tile_ckpt.py` (B2, tile-level progress trên FRAM), `baselines/jit_ckpt.py` (B3, checkpoint tại ngưỡng điện áp). `models/multiexit.py: MultiExitNet(backbone, exit_points) → list[logits]`. Test: mỗi baseline hoàn thành inference khi năng lượng dư; B1 thất bại khi năng lượng/cycle < chi phí inference.
- **Simulation:** E1 phần 1 — so sánh B1/B2/B3 trên trace hằng và T2.
- **Expected results:** Thứ tự overhead giữa B1–B3; accuracy theo từng exit trên GSC (số đo thật).
- **Milestone:** Baseline sim-ready.
- **Deliverables:** Bảng accuracy theo exit (GSC), figure overhead B1–B3.
- **Risk & Plan B:** Thứ tự overhead trái với công bố → rà lại mô hình chi phí; quá 2 tuần → dùng bản đơn giản hóa và ghi rõ.

### Sprint 5 — 30/11 → 13/12/2026 (M3)
- **Objective:** Pipeline end-to-end, vector hóa, B4, PSB v1.
- **Research:** Đọc sâu Zygarde, Wu DAC'20 để dựng B4 đúng tinh thần.
- **Algorithm/Model:** PSB v1: $m_k$ cố định, b ∈ {4, 8}, không gate.
- **Coding:** `sim/executor_batch.py: BatchExecutor.run(traces[N,T], policy) → RunLogBatch`; `baselines/energy_threshold_exit.py` (B4); `eval/calibration.py: TemperatureScaling.fit(logits, y)`; `models/psb.py: PSB(C_in, m, bits)`; `train/train_psb.py`. Test: kết quả batch == kết quả tuần tự trên 10 trace; ECE giảm sau calibration.
- **Simulation:** E2 trên GSC với B0, B1, B2, B4; T1 dev.
- **Expected results:** Throughput R3 đo được; bảng E2 sơ bộ.
- **Milestone:** End-to-end chạy trọn.
- **Deliverables:** `E2_gsc_prelim.csv`, figure throughput.
- **Risk & Plan B:** Vector hóa khó → multiprocessing CPU.

### Sprint 6 — 14/12 → 27/12/2026 (M3)
- **Objective:** Qua Prototype Gate và Real-time Gate.
- **Research:** Tổng hợp E1 (so với xu hướng công bố) vào tech report.
- **Algorithm/Model:** Quét rate–accuracy PSB v1 trên GSC (6 mức $m_k$ × 2 mức b).
- **Coding:** `policy/threshold.py` (policy ngưỡng năng lượng+confidence); `policy/overhead.py: lut_cycles(n_entries)`, `confidence_cycles(n_classes)`; `eval/metrics.py` (ICR, correct/mJ, NVM bytes, latency-in-deadline). Test: metric trên log giả lập có đáp án biết trước.
- **Simulation:** E1 hoàn tất; PSB v1 vs B2 trên T1 dev.
- **Expected results:** Đường cong rate–accuracy v1; overhead R2 ước lượng.
- **Milestone:** 🔶 **Prototype Gate (M2–3) + Real-time Gate (M3) — 27/12/2026**.
- **Deliverables:** `tech_report_1_simulator_baselines.md`, `fig_rate_accuracy_v1.pdf`.
- **Risk & Plan B:** Kill K2 / K3 (Mục 21).

### Sprint 7 — 28/12/2026 → 10/01/2027 (M4)
- **Objective:** Policy MDP v1 và PSB v2.
- **Research:** Đọc RL-SEP, TECS "Timing-Constrained … RL" để phân biệt rõ với MDP của mình.
- **Algorithm/Model:** Xây MDP (Mục 7.3); PSB v2 thêm hard-concrete gate và $\lambda_R$.
- **Coding:** `policy/mdp.py: build_mdp(val_stats, cost_model, bins) → MDP`; `policy/value_iteration.py: solve(mdp, tol) → (V, π)`; `policy/lut_export.py: to_lut(π) → bytes`; `policy/oracle.py` (B8); `models/psb.py: HardConcreteGate`. Test: VI hội tụ; LUT ≤ 2 KB; oracle ≥ mọi policy trên cùng trace.
- **Simulation:** GSC, T1 dev: MDP vs ngưỡng vs oracle.
- **Expected results:** Khoảng cách MDP–oracle và MDP–ngưỡng (số đo).
- **Milestone:** Proposed algorithm v1.
- **Deliverables:** `policy_v1_report.md`, LUT mẫu.
- **Risk & Plan B:** MDP không hơn ngưỡng → tăng thông tin trạng thái (ước lượng $P_h$ tốt hơn) hoặc giảm số bin; nếu vẫn không → policy ngưỡng tối ưu bằng grid search làm phương án chính.

### Sprint 8 — 11/01 → 24/01/2027 (M4)
- **Objective:** ETCT v1; mở Conference Research Gate.
- **Research:** Kiểm tra CFP ISLPED 2027 và EWSN 2027 (deadline, page limit, track); viết nháp Related Work + Method.
- **Algorithm/Model:** ETCT xen kẽ (Mục 8.3), I = 3 vòng.
- **Coding:** `train/etct.py: estimate_exit_probs(sim, policy, traces) → p[K]`; `train/etct.py: ETCTTrainer.fit()`. Test: $\sum_k p_k \le 1$; phần còn lại = tỷ lệ không hoàn thành.
- **Simulation:** E4 trên GSC + CWRU; E3 trên CWRU.
- **Expected results:** Bảng ETCT vs trọng số đều (số đo); đường rate–accuracy CWRU.
- **Milestone:** 🔶 **Conference Research Gate (M4–5) — 24/01/2027**: phải có evidence sơ bộ cho ≥ 2/3 RQ.
- **Deliverables:** `E3_E4_prelim.csv`, draft `sec_related.tex`, `sec_method.tex`, `venue_check.md`.
- **Risk & Plan B:** Evidence yếu → thu hẹp contribution (Kill K4).

### Sprint 9 — 25/01 → 07/02/2027 (M5)
- **Objective:** Method v2 tích hợp; chạy E5, E6 trên 3 dataset.
- **Research:** Không đọc thêm; chỉ bổ sung trích dẫn khi viết.
- **Algorithm/Model:** Train VWW multi-exit + PSB + ETCT.
- **Coding:** `data/vww.py`, `data/cwru.py` hoàn chỉnh; `eval/run_matrix.py --exp E5,E6`; `eval/stats.py: paired_test(a, b) → (p, d, CI)`.
- **Simulation:** E5 (main), E6 (ablation) — 3 dataset × 5 seeds, T1 **test**.
- **Expected results:** Bảng chính và ablation (số đo).
- **Milestone:** Toàn bộ run chính trong hàng đợi.
- **Deliverables:** Log thô E5/E6.
- **Risk & Plan B:** Kỳ nghỉ Tết Nguyên đán 2027 rơi vào cuối sprint này (ngày cụ thể cần kiểm tra lịch) → xếp run dài vào hàng đợi trước kỳ nghỉ. VWW quá chậm → báo cáo VWW với 3 seeds.

### Sprint 10 — 08/02 → 21/02/2027 (M5)
- **Objective:** Hoàn tất kết quả và bản thảo conference.
- **Research:** Viết Introduction, Experiments, Discussion, Threats to validity.
- **Algorithm/Model:** Không thay đổi thuật toán (freeze).
- **Coding:** `eval/plots.py` (Pareto accuracy–energy, rate–accuracy, ablation bar).
- **Simulation:** Chạy bù các run lỗi; thống kê đầy đủ.
- **Expected results:** Bản thảo đầy đủ 6 trang (theo khuôn ISLPED 2026: 6 trang + 1 trang tham khảo — cần kiểm tra lại cho 2027).
- **Milestone:** 🔶 **Conference Readiness (M5) — 21/02/2027**.
- **Deliverables:** `conf_draft_v1.pdf`, bảng/hình cuối.
- **Risk & Plan B:** Draft chưa xong → dời sang EWSN 2027 (deadline dự kiến muộn hơn).

### Sprint 11 — 22/02 → 07/03/2027 (M6)
- **Objective:** Mock review, sửa, nộp.
- **Research:** Mock review nội bộ (cố vấn + 1 đồng nghiệp ngoài nhóm) theo tiêu chí reviewer.
- **Algorithm/Model:** Chỉ thí nghiệm bổ sung nhỏ nếu review yêu cầu.
- **Coding:** Chuẩn bị artifact ẩn danh (anonymized repo).
- **Simulation:** Run bổ sung ≤ 20 GPU-h.
- **Expected results:** `conf_draft_v2`.
- **Milestone:** Nộp ISLPED 2027 **nếu** deadline thực tế nằm trong sprint này [TBD].
- **Deliverables:** Bản nộp, artifact ẩn danh.
- **Risk & Plan B:** Deadline ISLPED sớm hơn dự kiến hoặc gate chưa đạt → EWSN 2027.

### Sprint 12 — 08/03 → 21/03/2027 (M6)
- **Objective:** Chốt conference milestone; buffer.
- **Research:** Viết tech report dài (sẽ thành khung journal).
- **Algorithm/Model:** Liệt kê hướng mở rộng journal (M6, robust policy).
- **Coding:** Dọn code, tài liệu, test coverage ≥ 70% cho `sim/` và `policy/` [TARGET].
- **Simulation:** —
- **Expected results:** Bản nộp conference (ISLPED) hoặc bản sẵn sàng cho EWSN.
- **Milestone:** 🔶 **Conference Milestone (M6) — 21/03/2027**.
- **Deliverables:** `tech_report_2_full_method.md`, release nội bộ v1.0.
- **Risk & Plan B:** Chưa nộp được → nộp EWSN 2027 ở S13–S14.

### Sprint 13 — 22/03 → 04/04/2027 (M7)
- **Objective:** Thiết kế journal extension; M6 v1.
- **Research:** Literature về evidence accumulation / anytime inference / sequential decision cho streaming sensing.
- **Algorithm/Model:** Temporal accumulation (Mục 7.4), calibrate $T_k$ từng exit.
- **Coding:** `temporal/accumulate.py: EvidenceAccumulator(gamma, T).update(logits_k) → posterior`; tích hợp vào executor như trạng thái persist. Test: posterior chuẩn hóa; $\gamma=0$ ⇒ tương đương exit độc lập.
- **Simulation:** E9 pilot trên CWRU, T2 công suất thấp.
- **Expected results:** Đường accuracy vs mean $P_h$ (pilot).
- **Milestone:** Journal extension plan được duyệt.
- **Deliverables:** `journal_extension_plan.md`.
- **Risk & Plan B:** CWRU không đủ dài theo thời gian → ghép cửa sổ liên tiếp cùng nhãn; hoặc thêm dataset HAR [TBD].

### Sprint 14 — 05/04 → 18/04/2027 (M7)
- **Objective:** E9 đầy đủ; trace mới.
- **Research:** Kiểm chứng và tải T3/T4 (Mục 9.2); nếu nộp EWSN 2027 thì hoàn thiện bản nộp.
- **Algorithm/Model:** Tinh chỉnh $\gamma$, $\gamma_{safe}$ trên validation.
- **Coding:** `data/traces.py` thêm loader T3/T4 (nếu có).
- **Simulation:** E9 — 5 seeds.
- **Expected results:** Bảng E9 (số đo).
- **Milestone:** C4 có evidence.
- **Deliverables:** `E9_results.csv`, figure E9.
- **Risk & Plan B:** Không có T3/T4 → mở rộng T2 theo tham số lấy từ literature cho solar/kinetic/RF.

### Sprint 15 — 19/04 → 02/05/2027 (M8)
- **Objective:** Chuẩn bị E8 và baseline SOTA thứ hai.
- **Research:** Đọc toàn văn HarvNet / NExUME để dựng B5 hoặc B6.
- **Algorithm/Model:** —
- **Coding:** `baselines/harvnet_like.py` hoặc `baselines/nexume_like.py`. Test: tái hiện xu hướng định tính trong bài gốc trên cấu hình gần nhất.
- **Simulation:** E8 setup: train trên T1, test trên T2/T3/T4.
- **Expected results:** Baseline B5/B6 chạy được.
- **Milestone:** Baseline journal đủ.
- **Deliverables:** Báo cáo tái hiện baseline.
- **Risk & Plan B:** > 2 tuần không tái hiện được → dùng B4-like mạnh nhất + B6-like, ghi rõ.

### Sprint 16 — 03/05 → 16/05/2027 (M8)
- **Objective:** Qua Journal Extension Gate.
- **Research:** Viết "delta statement": journal khác conference ở đâu (contribution, experiment, % nội dung mới).
- **Algorithm/Model:** Quyết định có làm robust policy (ambiguity set trên ma trận chuyển $P_h$) hay không — chỉ làm nếu E8 cho thấy policy suy giảm khi đổi trace.
- **Coding:** Xử lý kết quả conference (camera-ready nếu được nhận — thông báo ISLPED 2026 là 18/05/2026, kỳ 2027 [TBD]).
- **Simulation:** E8 phần 1.
- **Expected results:** Delta statement; E8 sơ bộ.
- **Milestone:** 🔶 **Journal Extension Gate (M8) — 16/05/2027**.
- **Deliverables:** `delta_statement.md`, E8 prelim.
- **Risk & Plan B:** Kill K5.

### Sprint 17 — 17/05 → 30/05/2027 (M9)
- **Objective:** E7 độ nhạy tham số.
- **Research:** —
- **Algorithm/Model:** (Nếu kích hoạt) robust MDP v1.
- **Coding:** `eval/run_matrix.py --exp E7` với lưới tham số; `eval/stats.py: kendall_tau()`.
- **Simulation:** E7 trên GSC, T2.
- **Expected results:** Heatmap Δ metric và Kendall-τ.
- **Milestone:** C5 phần 1.
- **Deliverables:** `E7_heatmaps.pdf`.
- **Risk & Plan B:** Lưới quá lớn → Latin hypercube 200 điểm thay cho full grid.

### Sprint 18 — 31/05 → 13/06/2027 (M9)
- **Objective:** E8 đầy đủ.
- **Research:** —
- **Algorithm/Model:** So robust MDP vs MDP chuẩn (nếu có).
- **Coding:** —
- **Simulation:** E8 — 2 dataset × 3–4 họ trace × 5 seeds.
- **Expected results:** Bảng tổng quát hóa cross-trace.
- **Milestone:** C5 phần 2.
- **Deliverables:** `E8_results.csv`, figure.
- **Risk & Plan B:** Mất tổng quát hóa → báo cáo trung thực như một finding; thêm robust policy làm contribution.

### Sprint 19 — 14/06 → 27/06/2027 (M10)
- **Objective:** E10 (NVM technology, tụ điện).
- **Research:** Thu thập tham số năng lượng ghi/đọc của FRAM, STT-MRAM, ReRAM từ literature (ghi nguồn).
- **Algorithm/Model:** —
- **Coding:** `configs/hw/nvm_*.yaml`.
- **Simulation:** E10 trên GSC.
- **Expected results:** Tỷ lệ $E_{persist}$ theo công nghệ NVM.
- **Milestone:** Hoàn tất ma trận thí nghiệm journal.
- **Deliverables:** `E10_results.csv`.
- **Risk & Plan B:** Thiếu tham số → chỉ báo cáo FRAM + 1 công nghệ có nguồn đáng tin.

### Sprint 20 — 28/06 → 11/07/2027 (M10)
- **Objective:** Qua Journal Experiment Gate; gói tái lập.
- **Research:** Rà soát thống kê toàn bộ (Holm correction).
- **Algorithm/Model:** Freeze.
- **Coding:** Gói tái lập: script một lệnh sinh lại mọi bảng/hình; Docker/conda env.
- **Simulation:** Chạy lại từ đầu một bảng chính để xác nhận tái lập.
- **Expected results:** Mọi bảng/hình journal có sẵn.
- **Milestone:** 🔶 **Journal Experiment Gate (M10) — 11/07/2027**.
- **Deliverables:** `repro_package/`, `all_tables.xlsx`.
- **Risk & Plan B:** Thiếu thí nghiệm → dùng S24 làm buffer.

### Sprint 21 — 12/07 → 25/07/2027 (M11)
- **Objective:** Viết journal — Introduction, Related Work, Problem Formulation.
- **Research:** Cập nhật literature 2026–2027 (forward citation lần 2) để chắc chắn không bị vượt mặt.
- **Algorithm/Model:** Freeze.
- **Coding:** Chỉ sửa lỗi phát hiện khi viết.
- **Simulation:** —
- **Expected results:** 3 section đầu hoàn chỉnh.
- **Milestone:** Khung journal.
- **Deliverables:** `journal_sec1-3.tex`.
- **Risk & Plan B:** Trùng văn bản với conference → viết lại hoàn toàn; kiểm tra similarity trước khi nộp.

### Sprint 22 — 26/07 → 08/08/2027 (M11)
- **Objective:** Method (mở rộng) + Experiments.
- **Research:** —
- **Algorithm/Model:** —
- **Coding:** `eval/plots.py` bản cuối, phong cách thống nhất.
- **Simulation:** —
- **Expected results:** Mọi hình/bảng đặt trong bản thảo.
- **Milestone:** Bản thảo có đủ evidence.
- **Deliverables:** `journal_sec4-6.tex`, `figures_final/`.
- **Risk & Plan B:** Vượt giới hạn trang → chuyển bớt sang supplementary.

### Sprint 23 — 09/08 → 22/08/2027 (M12)
- **Objective:** Discussion, Limitations, Threats to validity (đặc biệt độ trung thực của simulator); bản thảo v1.
- **Research:** Liệt kê trung thực mọi giả định (up-sampling trace, tham số chi phí, không đo phần cứng).
- **Algorithm/Model:** —
- **Coding:** —
- **Simulation:** —
- **Expected results:** Bản thảo v1 đầy đủ.
- **Milestone:** Draft v1.
- **Deliverables:** `journal_draft_v1.pdf`.
- **Risk & Plan B:** Chậm tiến độ → cắt phần phụ (E10) sang supplementary.

### Sprint 24 — 23/08 → 05/09/2027 (M12)
- **Objective:** Mock review mức Q1 (cố vấn + reviewer ngoài nhóm).
- **Research:** Đọc 3–5 bài gần nhất đăng trên IoT-J cùng chủ đề để khớp kỳ vọng.
- **Algorithm/Model:** —
- **Coding:** —
- **Simulation:** Thí nghiệm bổ sung nhỏ ≤ 40 GPU-h.
- **Expected results:** Danh sách sửa đổi đã xử lý.
- **Milestone:** Draft v2.
- **Deliverables:** `mock_review.md`, `journal_draft_v2.pdf`.
- **Risk & Plan B:** Yêu cầu thí nghiệm lớn → hoãn nộp 1 sprint (dùng S26).

### Sprint 25 — 06/09 → 19/09/2027 (M12)
- **Objective:** Hoàn thiện và nộp IEEE IoT-J.
- **Research:** Tuyên bố quan hệ với bài conference theo chính sách của tạp chí (kiểm tra chính sách IEEE về bản mở rộng).
- **Algorithm/Model:** —
- **Coding:** Công khai artifact (sau khi cố vấn duyệt).
- **Simulation:** —
- **Expected results:** Bản nộp hợp lệ.
- **Milestone:** 🔶 **Journal Milestone (M12) — 19/09/2027**.
- **Deliverables:** Bản nộp, cover letter, artifact công khai.
- **Risk & Plan B:** Desk reject vì scope → chuyển IEEE TCAD hoặc IEEE TC (Mục 23).

### Sprint 26 — 20/09 → 03/10/2027 (M12)
- **Objective:** Buffer; lưu trữ; đồng bộ với khóa luận tốt nghiệp (nếu áp dụng).
- **Research:** Tổng kết bài học; đề xuất hướng tiếp theo.
- **Algorithm/Model:** —
- **Coding:** Lưu trữ repo, tag phiên bản.
- **Simulation:** —
- **Expected results:** Dự án được lưu trữ đầy đủ.
- **Milestone:** Đóng dự án 12 tháng.
- **Deliverables:** `project_archive/`, `next_steps.md`.
- **Risk & Plan B:** Nếu S25 bị trễ, S26 là sprint nộp.

---

## 17. Algorithm Development Plan

| Version | Sprint | Problem addressed | Algorithm change | Expected improvement (hypothesis) | Experiment | Success criteria [TARGET] |
|---|---|---|---|---|---|---|
| Baseline | S4–S5 | Cần sàn/trần và đối chứng | B0–B4, B8 trong simulator | — | E1, E2 | Thứ tự overhead khớp xu hướng công bố ở ≥ 2 baseline |
| Baseline enhancement | S5–S6 | B4 dùng ngưỡng thô | B4 + confidence đã calibrate + ngưỡng tối ưu grid search | Baseline mạnh hơn → so sánh công bằng | E2 | B4-enhanced ≥ B4 về correct/mJ |
| Proposed v1 | S6–S7 | Persist tốn năng lượng; exit heuristic | PSB v1 (m, b cố định) + MDP policy | Giảm NVM bytes; tăng correct/mJ | E3, E5 (GSC) | PSB giảm bytes ≥ 2× với ΔAcc ≤ 1 pp trên GSC |
| Ablation v1 | S7–S8 | Phần nào tạo lợi ích? | Tắt từng thành phần | Xác định đóng góp riêng | E6 (GSC) | Mỗi thành phần đóng góp dương có ý nghĩa thống kê (p < 0.05) |
| Proposed v2 | S8–S9 | Exit huấn luyện không khớp phân phối dừng thực | PSB v2 (gate học được) + ETCT | Tăng expected accuracy | E4, E5 | ETCT ≥ +2 pp $\mathbb E_\tau[\mathrm{Acc}]$ trên ≥ 2/3 dataset |
| Optimization | S9–S10 | Overhead policy, kích thước LUT | Rời rạc hóa tối ưu; INT4 cho $s_k$ nếu cần | Overhead R2 ≤ 1% | E5 | LUT ≤ 2 KB, R2 ≤ 1% |
| Final (conference) | S10 | — | Freeze | — | E5, E6 đủ 3 dataset | H1–H3 đạt hoặc báo cáo trung thực vùng đạt/không đạt |
| Journal v3 | S13–S18 | Thu hoạch yếu kéo dài; đổi phân phối trace | + M6 temporal accumulation; (tùy chọn) robust MDP | Phục hồi accuracy; ổn định khi đổi trace | E8, E9 | H4, H5 |

---

## 18. Model Development Plan

| Hạng mục | Quyết định | Gắn với hypothesis |
|---|---|---|
| Architecture | DS-CNN / MobileNetV1-0.25 / 1D-CNN + K exit heads | Nền cho H1–H3 |
| Input/output | Theo bảng Mục 8.1; output logits mỗi exit | — |
| Feature representation | Activation tại ranh giới exit; trạng thái nén $s_k$ | H1 |
| Loss | $\mathcal L_{ETCT}+\mathcal L_{PSB}$ (Mục 8.2–8.3) | H1, H2 |
| Training strategy | Pre-train đều → xen kẽ policy/ETCT; QAT INT8 ở giai đoạn cuối | H2 |
| Validation strategy | Chọn hyperparameter trên validation split + trace val; test chỉ chạy một lần cho mỗi cấu hình đã chốt | Tránh rò rỉ |
| Hyperparameters | Mục 8.4 | — |
| Data augmentation | Recipe chuẩn tối thiểu (Mục 8.4) | Không thêm nếu không phục vụ RQ |
| Ablation | −PSB, −ETCT, −MDP, −calibration, K ∈ {2,3,4}, b ∈ {2,4,8} | H1–H3 |
| Model complexity | Tham số ≤ 256 KB (INT8); FLOPs báo cáo theo exit | Ràng buộc |
| Inference latency | Tính bằng cycles từ mô hình chi phí (không đo thiết bị) | R1, R2 |

**Nguyên tắc:** không thêm attention, transformer hay NAS nếu không có RQ cần đến. NAS kết hợp vị trí exit chỉ là hướng tương lai, không nằm trong 12 tháng.

---

## 19. Milestones & Feasibility Gates

| Gate | Thời điểm | Tiêu chí GREEN | YELLOW | RED → hành động |
|---|---|---|---|---|
| **M1 — Research & Novelty** | 30/10/2026 (S2) | Novelty memo xác nhận PSB+ETCT+MDP chưa có công trình làm đồng thời; dataset và trace tải được; pilot chạy | Có công trình gần, còn khác biệt rõ | Kill K1: đổi RQ theo Plan B-RQ |
| **M2–3 — Prototype** | 27/12/2026 (S6) | Pipeline end-to-end chạy; E1 đạt; baseline B0–B4 chạy trên GSC; PSB v1 có đường rate–accuracy | 1 baseline chưa khớp | Kill K3 nếu simulator không tái hiện xu hướng |
| **M3 — Real-time** | 27/12/2026 (S6) | R2 ≤ 1%; LUT dự kiến ≤ 2 KB; R3 đủ cho ngân sách | R2 1–5% | Giảm số exit/bin; dùng policy ngưỡng |
| **M4–5 — Conference Research** | 24/01/2027 (S8) | Evidence sơ bộ ủng hộ ≥ 2/3 H1–H3 trên ≥ 2 dataset | 1/3 | Kill K4: thu hẹp contribution |
| **M5 — Conference Readiness** | 21/02/2027 (S10) | Đủ: thuật toán, baseline, E5, E6, bản thảo | Thiếu VWW | Dời sang EWSN 2027 |
| **M6 — Conference Milestone** | 21/03/2027 (S12) | Đã nộp hoặc sẵn sàng nộp | — | Nộp venue dự phòng |
| **M8 — Journal Extension** | 16/05/2027 (S16) | ≥ 1 contribution thuật toán mới (C4) có evidence + kế hoạch E7/E8/E10 khả thi | Chỉ có evaluation mới | Kill K5 |
| **M10 — Journal Experiment** | 11/07/2027 (S20) | E7–E10 xong; robustness, generalization, ablation, efficiency, real-time; gói tái lập | Thiếu 1 thí nghiệm phụ | Dùng buffer S24/S26 |
| **M12 — Journal Milestone** | 19/09/2027 (S25) | Nộp IoT-J | — | Nộp S26 |

---

## 20. Critical Path

```
[S1–S2] Novelty gate ──► [S3–S6] Simulator + calibration (E1) ──► [S4–S6] Baselines in sim (E2)
                                                                         │
[S2–S4] Multi-exit backbones (song song) ────────────────────────────────┤
                                                                         ▼
                                       [S5–S7] PSB v1→v2 + MDP policy ──► [S8] ETCT
                                                                         │
                                                                         ▼
                                       [S9–S10] E5 + E6 ──► [S10–S12] Conference paper
                                                                         │
                                                                         ▼
                        [S13–S16] M6 + trace mới + SOTA baseline ──► [S17–S20] E7–E10
                                                                         │
                                                                         ▼
                                                        [S21–S25] Journal paper
```

| Hạng mục | Nội dung |
|---|---|
| Critical tasks | Novelty gate; hiệu chỉnh simulator (E1); PSB+MDP; E5/E6; M6 |
| Dependencies | Mọi thí nghiệm phụ thuộc simulator; ETCT phụ thuộc policy; E8 phụ thuộc trace T3/T4 hoặc T2 mở rộng |
| Parallel tasks | Train backbone ‖ xây simulator (S2–S4); viết Related Work ‖ chạy E5 (S8–S9); đọc HarvNet/NExUME ‖ E9 (S14–S15) |
| Bottlenecks | (1) Độ trung thực simulator; (2) tái hiện baseline không có code; (3) một sinh viên làm toàn bộ; (4) kỳ nghỉ Tết giữa S9–S10 |
| Buffer | S12 (sau conference), S24 (trước nộp journal), S26 (cuối) ≈ 6 tuần |

---

## 21. Risk & Contingency

| Risk | Trigger | Plan A | Plan B |
|---|---|---|---|
| Novelty yếu | Tìm thấy công trình làm đồng thời PSB + exit + chi phí NVM | Nhấn mạnh phần khác biệt (ETCT, LUT MDP) | Plan B-RQ: "tổng quát hóa của policy intermittent inference khi phân phối năng lượng thay đổi" (cross-trace robust policy) |
| Simulator thiếu trung thực | Không tái hiện xu hướng ≥ 2 baseline tới S6 | Rà lại mô hình chi phí | Định vị paper là thuật toán + mô hình năng lượng phân tích + phân tích độ nhạy; venue thiên ML (ví dụ IJCNN 2028) |
| Model không hội tụ (PSB/ETCT) | Val loss dao động / accuracy exit cuối giảm > 2 pp | Giảm $\lambda_R$, warm-up, tách giai đoạn | Bỏ gate học được, dùng $m_k$ cố định |
| Baseline không tái hiện được | > 2 tuần | Debug mô hình chi phí | Bản "-like" đơn giản hóa, ghi rõ |
| Policy không cải thiện | MDP ≤ ngưỡng tối ưu | Thêm thông tin trạng thái, tinh chỉnh bin | Chọn policy ngưỡng tối ưu; contribution chuyển sang PSB + ETCT |
| Real-time (R2) không đạt | Overhead > 5% | Giảm bin, gộp block | Policy ngưỡng 2 tham số |
| Dataset/trace không phù hợp | Trace thô gây kết quả phi thực tế | Up-sampling có kiểm soát | Chỉ dùng T2 với tham số từ literature, báo cáo rõ |
| Conference evidence yếu | Gate M4–5 YELLOW/RED | Thêm dataset/seed, tìm vùng hoạt động có lợi | Thu hẹp claim về vùng thu hoạch thấp; nộp workshop (ví dụ ENSsys) để lấy phản hồi |
| Journal novelty yếu | Gate M8 RED | M6 + robust policy | Chiều đánh giá mới (NVM technologies, cross-modality) + kéo dài timeline, không nộp Q1 ở M12 |
| Reviewer đòi phần cứng | Review conference yêu cầu testbed | Bổ sung calibration, sensitivity, so sánh với số liệu công bố | Cố vấn quyết định có dùng thiết bị lab sẵn có hay không (ngoài phạm vi roadmap) |
| GPU tại trường hạn chế | Hàng đợi > 48 h | Giảm seeds cho quét phụ | Ưu tiên CPU cho simulator; train trên dataset nhỏ trước |

### Kill criteria

| ID | Thời điểm | Điều kiện | Hành động |
|---|---|---|---|
| K1 | S2 | Novelty memo xác định trùng lặp chính | Dừng PSB làm contribution chính; chuyển Plan B-RQ |
| K2 | S6 | PSB không giảm ≥ 2× bytes với ΔAcc ≤ 1 pp trên ≥ 2/3 dataset (thử trên GSC + CWRU) | PSB thành ablation; contribution chính = ETCT + policy |
| K3 | S6 | Simulator không tái hiện xu hướng ≥ 2 baseline | Chuyển sang mô hình phân tích + sensitivity; đổi venue |
| K4 | S8/S10 | Không cấu hình nào hơn baseline tốt nhất về correct/mJ với p < 0.05 | Thu hẹp claim; lùi conference sang kỳ sau |
| K5 | S16 | Không có contribution thuật toán mới cho journal | Không nộp Q1 ở M12; kéo dài 3–6 tháng |

---

## 22. Conference Strategy

**Nguyên tắc:** rank và deadline dưới đây lấy từ CFP chính thức hoặc trang tổng hợp thứ cấp (myhuiban.com — hiển thị ICORE/CORE). Cổng CORE chính thức không truy cập được từ công cụ tìm kiếm (robots.txt), nên **mọi rank cần được xác nhận lại trên portal.core.edu.au trong Sprint 8**. Deadline 2027 chưa công bố trừ khi ghi rõ.

| Venue | Rank (nguồn) | Deadline tham chiếu | Phù hợp scope | Yêu cầu thực nghiệm | Đánh giá |
|---|---|---|---|---|---|
| **ISLPED 2027** | ICORE A; CCF C (myhuiban) | 2026: abstract 02/03/2026, full paper 09/03/2026, 6 trang + 1 trang tham khảo (CFP chính thức) → 2027 dự kiến đầu tháng 3/2027 [TBD] | Track 2 "EDA, Systems, and Software": power-aware IoT/embedded, energy-aware software (CFP 2026) | Kết quả năng lượng thường từ mô hình/đo; simulation có hiệu chỉnh có cơ hội [INFERENCE] | **Primary (stretch)** — khớp S11 |
| **EWSN 2027** | CORE B; QUALIS A2 (myhuiban) | 2026: 24/04/2026 → 2027 dự kiến tháng 4/2027 [TBD] | Embedded wireless / batteryless systems | Cộng đồng thiên hệ thống, có thể hỏi về hardware [INFERENCE] | **Fallback chính** — khớp S13–S14 |
| DCOSS-IoT 2027 | Chưa xác minh | 27/01/2027 (dcoss.org) | IoT, distributed smart systems | — | Không khuyến nghị: deadline rơi vào S8, trước Conference Readiness |
| IJCNN 2027 | ICORE B; CCF C (myhuiban) | 31/01/2027 (CFP chính thức), 6 trang | Neural networks, rộng | Chấp nhận simulation | Không khuyến nghị cho vòng này (deadline S9); là lựa chọn nếu Kill K3 kích hoạt ở kỳ 2028 |
| ESWEEK 2027 (CODES+ISSS / EMSOFT / CASES) | TBD | 2026: full paper 30/03/2026; bài được nhận đăng ở IEEE TCAD | Rất khớp scope | Kỳ vọng thực nghiệm cao | Không khuyến nghị: xuất bản dạng journal (TCAD) sẽ làm khó bước mở rộng journal |

**Khuyến nghị:** ISLPED 2027 là mục tiêu chính, EWSN 2027 là dự phòng. Quyết định cuối tại Gate M5 (21/02/2027) dựa trên độ mạnh của E5/E6. Cấp C13 quyết định: *Academic quality* — chọn venue nơi phương pháp simulation được chấp nhận, thay vì venue rank cao hơn nhưng dễ reject vì phương pháp luận.

**Contribution kỳ vọng cho conference:** C1 (PSB), C2 (ETCT), C3 (MDP policy LUT) + simulator mở.

---

## 23. Journal Strategy

Trạng thái Q1 lấy từ SCImago (SJR 2025) [VERIFIED]. APC là ước tính của SCImago cho open access; các tạp chí IEEE dưới đây là hybrid — xuất bản truyền thống không bắt buộc APC, nhưng có thể có phí trang vượt giới hạn [TBD — kiểm tra hướng dẫn tác giả].

| Journal | Q (SJR 2025) | Scope khớp | Kỳ vọng thực nghiệm | APC OA (ước tính SCImago) | Vai trò |
|---|---|---|---|---|---|
| **IEEE Internet of Things Journal** | Q1 ở cả 6 category, SJR 2.144 | Rất khớp (IoT, embedded AI, energy harvesting) | Cao; simulation quy mô lớn thường gặp [INFERENCE] | ~US$4,107 | **Primary** |
| IEEE Transactions on Computer-Aided Design (TCAD) | Q1 ở 3 category, SJR 0.928 | Khớp (system-level design, embedded) | Cao; đã đăng bài intermittent (MII 2024) | TBD | Secondary |
| IEEE Transactions on Computers | Q1 ở 4 category, SJR 1.243 | Khớp một phần | Cao | ~US$3,432 | Secondary |
| IEEE Transactions on Mobile Computing | Q1, SJR 2.397 | Khớp một phần (mobile/embedded) | Rất cao | TBD | Stretch |
| ACM TECS | **Q2** (SJR 2025: 0.864) | Rất khớp | — | ~US$3,195 | **Không đạt mục tiêu Q1** — chỉ dùng nếu mục tiêu thay đổi |

**Yêu cầu contribution cho journal so với conference:**
- C4 (temporal evidence accumulation) — mới về thuật toán.
- C5 (tổng quát hóa cross-trace, NVM technologies, sensitivity) — mới về đánh giá.
- Thêm ≥ 1 baseline SOTA (B5 hoặc B6), thêm dataset/trace.
- Tuân thủ chính sách của tạp chí về bản mở rộng từ bài hội nghị (tỷ lệ nội dung mới, trích dẫn bài gốc) — kiểm tra trong S16.

---

## 24. Final Feasibility Report

| Category | Status | Evidence | Risk | Mitigation | Confidence |
|---|---|---|---|---|---|
| Research Gap | 🟡 | Gap ứng viên chưa thấy trong ≥ 15 công trình đã kiểm tra; lĩnh vực đông | Công trình chưa tìm thấy | Protocol 5.4, Gate M1 | Trung bình |
| Novelty | 🟡 | PSB (representation), ETCT và rate-aware loss (loss) là mới ở mức ứng viên; formulation gốc trùng lặp cao | Bị coi là tổ hợp kỹ thuật có sẵn | Ablation đồng tối ưu (E6) | Trung bình |
| Algorithm | 🟢 | Thành phần chuẩn, MDP nhỏ | Không gian trạng thái | Rời rạc hóa + sensitivity | Cao |
| Model | 🟢 | Backbone TinyML phổ biến | Accuracy thấp dưới ràng buộc bộ nhớ | Mô hình bộ nhớ phân tích | Cao |
| Dataset | 🟢 | GSC, VWW, CWRU, EnHANTs công khai | Trace thô 30 s | Up-sampling + synthetic | Cao (task) / Trung bình (trace) |
| Simulation | 🟡 | Có tiền lệ mô hình hiệu năng (iNAS-PMod, ScEpTIC, BatterylessSim) | Độ trung thực | Hiệu chỉnh + E1 + sensitivity + mở mã | Trung bình |
| Computation | 🟢 | Model nhỏ; ước lượng 150–400 GPU-h | GPU cụ thể chưa biết | Pilot S2 | Cao (sau pilot) |
| Baseline | 🟡 | Có code firmware của SONIC, HAWAII, iNAS, DynBal, Coala | Phải dựng lại mô hình chi phí | Bản "-like" + ghi rõ | Trung bình |
| Real-time | 🟢 | Đo được trong simulator (R1–R3) | Overhead bị đánh giá thấp | Mô hình cycles bảo thủ | Cao |
| Conference ≤ 6M | 🟡 | Ma trận E1–E6 vừa với S1–S10; ISLPED/EWSN có deadline phù hợp theo mẫu năm 2026 | Deadline 2027 chưa công bố; rank từ nguồn thứ cấp | 2 venue, quyết định tại M5 | Trung bình |
| Journal ≤ 12M | 🟡 | Có đường mở rộng rõ (C4, C5); IoT-J Q1 và khớp scope | Reviewer Q1 đòi hardware | Calibration + sensitivity + tái lập; quyết định spot-check thuộc cố vấn | Trung bình–thấp |

### Kết luận

🟡 **FEASIBLE WITH CONDITIONS.**

Điều kiện, theo thứ tự ưu tiên:
1. Gate M1 xác nhận novelty của PSB + ETCT + MDP (hoặc chuyển Plan B-RQ).
2. Simulator tái hiện được xu hướng đã công bố của ≥ 2 baseline trước 27/12/2026.
3. Chấp nhận đổi venue: ISLPED/EWSN 2027 cho conference, IEEE IoT-J cho journal — không nhắm SenSys/MobiSys như file nguồn.

Rủi ro lớn nhất cho mục tiêu Q1 là không có đo đạc phần cứng. Đây là giới hạn do ràng buộc dự án, không phải do thuật toán, và phải được nêu thẳng trong phần *Threats to validity* của cả hai bài.

---

## 25. Success Criteria

| Mốc | Tiêu chí đo được |
|---|---|
| Tháng 1 | Novelty memo được duyệt; ≥ 30 công trình trong bảng; pilot có số đo VRAM/thời gian |
| Tháng 3 | Simulator qua E1; B0–B4, B8 chạy trên GSC; PSB v1 có đường rate–accuracy; R2 ≤ 1% |
| Tháng 5 | E5, E6 đầy đủ 3 dataset × 5 seeds; kiểm định thống kê; bản thảo conference hoàn chỉnh |
| Tháng 6 | 1 bài nộp ISLPED 2027 hoặc EWSN 2027 (rank ≥ B theo nguồn đã xác nhận lại) |
| Tháng 8 | C4 có evidence (E9); delta statement được duyệt |
| Tháng 10 | E7–E10 xong; gói tái lập chạy lại được một bảng chính từ đầu |
| Tháng 12 | 1 bài nộp IEEE IoT-J (hoặc Q1 dự phòng); repo công khai |
| Xuyên suốt | Không có kết quả nào được báo cáo mà không có log, seed và git hash tương ứng |

**Next step:** Bảo Khánh bắt đầu Sprint 1 ngày 05/10/2026; cố vấn chốt lịch họp Gate M1 vào tuần 26–30/10/2026.

---

## Phụ lục A — Nguồn đã kiểm chứng (truy cập 30/09/2026)

**Literature**
- Maeng & Lucia, *Adaptive Dynamic Checkpointing for Safe Efficient Intermittent Computing* (Chinchilla), USENIX OSDI 2018 — https://www.usenix.org/conference/osdi18/presentation/maeng
- Gobieski, Lucia, Beckmann, *Intelligence Beyond the Edge: Inference on Intermittent Embedded Systems* (SONIC/TAILS), ASPLOS 2019 — https://dl.acm.org/doi/10.1145/3297858.3304011 ; code: https://github.com/CMUAbstract/SONIC
- *Dynamic Task-based Intermittent Execution for Energy-harvesting Devices* (Coala), ACM TOSN — https://dl.acm.org/doi/10.1145/3360285 ; code: https://github.com/TUDSSL/Coala
- Islam & Nirjon, *Zygarde*, ACM IMWUT 4(3), 2020 — https://dl.acm.org/doi/10.1145/3411808
- Wu et al., *Intermittent Inference with Nonuniformly Compressed Multi-Exit Neural Network for Energy Harvesting Powered Devices*, DAC 2020 — https://ieeexplore.ieee.org/document/9218526/
- *EPerceptive: Energy reactive embedded intelligence for batteryless sensors* — https://research.tudelft.nl/en/publications/eperceptive-energy-reactive-embedded-intelligence-for-batteryless
- *Everything Leaves Footprints: Hardware Accelerated Intermittent Deep Inference* (HAWAII) — https://ieeexplore.ieee.org/document/9211553/ ; code: https://github.com/EMCLab-Sinica/HAWAII_Project
- *Intermittent-Aware Neural Architecture Search*, ACM TECS — https://dl.acm.org/doi/10.1145/3476995 ; code: https://github.com/EMCLab-Sinica/Intermittent-aware-NAS
- *Stateful Neural Networks for Intermittent Systems*, IEEE — https://ieeexplore.ieee.org/document/9925685/
- Kortbeek et al., *WARio*, PLDI 2022 — https://par.nsf.gov/servlets/purl/10344721
- *Keep in Balance: Runtime-reconfigurable Intermittent Deep Inference*, ACM TECS — https://dl.acm.org/doi/full/10.1145/3607918 ; code: https://github.com/EMCLab-Sinica/DynBal
- Jeon et al., *HarvNet*, ACM MobiSys 2023 — https://dl.acm.org/doi/10.1145/3581791.3596845
- Zhang, Liu, Kim, *MII*, IEEE TCAD 2024 — https://dl.acm.org/doi/10.1109/TCAD.2024.3443710
- Mishra et al., *Revisiting DNN Training for Intermittently-Powered Energy-Harvesting Micro-Computers* (NExUME) — https://arxiv.org/abs/2408.13696
- Zhao, Xun, Chauhan, Merrett, *Power- and deadline-aware dynamic inference on intermittent computing systems* (DualAdaptNet), DATE 2025 — https://eprints.soton.ac.uk/497948
- Solanki et al., *ATM-Net* — https://arxiv.org/abs/2502.09822
- *RL-SEP* (poster), SenSys 2025 — https://dl.acm.org/doi/10.1145/3715014.3724046
- *Catch Non-determinism If You Can: Intermittent Inference of Dynamic Neural Networks*, ACM TECS — https://dl.acm.org/doi/10.1145/3757917
- *Timing-Constrained Composable Inference for Intermittent Systems Using Reinforcement Learning*, ACM TECS — https://doi.org/10.1145/3814956
- *Energy-Adaptive Checkpoint-Free Intermittent Inference for Low Power Energy Harvesting Systems* — https://arxiv.org/abs/2503.06663
- Zhang, *Accelerate Intermittent Deep Inference* — https://arxiv.org/abs/2407.14514
- Barjami, Miele, Mottola, *On the Sweet Spot of Intermittent Inference in the Battery-less IoT* — https://www.neslab.it/wp-content/uploads/2025/10/barjami25sweet.pdf

**Simulation / trace**
- CRAWDAD columbia/enhants (IEEE DataPort) — https://ieee-dataport.org/open-access/crawdad-columbiaenhants
- ScEpTIC documentation — https://neslabpolimi.bitbucket.io/ScEpTIC/
- BatterylessSim — https://github.com/PERSISTLab/BatterylessSim
- CSIRO Solar and Kinetic Energy Harvesting Dataset (nội dung chưa đọc được) — https://data.csiro.au/collection/csiro:65940

**Venue**
- ISLPED 2026 CFP — https://ieee-ceda.org/files/ieeeceda/2026-02/islped2026cfp_v2-1.pdf ; rank: https://www.myhuiban.com/conference/559
- EWSN 2026 (rank, deadline) — https://www.myhuiban.com/conference/904?lang=en_us
- DCOSS-IoT 2027 — https://dcoss.org/
- IJCNN 2027 CFP — https://ijcnn.org/2027/authors/call-for-papers ; rank: https://www.myhuiban.com/conference/428
- ESWEEK 2026 — https://www.sigarch.org/call-contributions/esweek-2026/

**Journal (SCImago)**
- IEEE IoT-J — https://www.scimagojr.com/journalsearch.php?q=21100338350&tip=sid
- IEEE TCAD — https://www.scimagojr.com/journalsearch.php?q=27724&tip=sid
- IEEE TC — https://www.scimagojr.com/journalsearch.php?q=25033&tip=sid
- IEEE TMC — https://www.scimagojr.com/journalsearch.php?q=25038&tip=sid
- ACM TECS — https://www.scimagojr.com/journalsearch.php?q=10300153313&tip=sid

## Phụ lục B — Các điểm cần kiểm chứng (TBD)

| Mục | Assumption hiện tại | Cách kiểm chứng | Sprint |
|---|---|---|---|
| GPU tại trường | Có ≥ 1 GPU ≥ 8 GB VRAM | `nvidia-smi`, hỏi quản trị lab | S1 |
| Baseline IIE | Không xác định | Hỏi người soạn file nguồn | S1 |
| Tham số năng lượng MSP430FR5994 | Lấy từ datasheet + bài đã công bố | Bảng tham số có nguồn | S3 |
| Venue cụ thể của ePerceptive | 2020 | Đọc DOI gốc | S1 |
| Tạp chí của "Stateful Neural Networks for Intermittent Systems" | IEEE journal 2022 | IEEE Xplore document 9925685 | S1 |
| Rank CORE/ICORE của ISLPED, EWSN | A / B (nguồn thứ cấp) | portal.core.edu.au | S8 |
| Deadline ISLPED 2027, EWSN 2027 | Tháng 3 và tháng 4/2027 | CFP chính thức | S8 |
| Trace T3/T4/T5 | Có thể không có | Tìm và tải | S13–S14 |
| Chính sách mở rộng từ conference của IoT-J | Cần tỷ lệ nội dung mới đáng kể | Hướng dẫn tác giả IEEE | S16 |
| Ngày Tết Nguyên đán 2027 | Trong S9–S10 | Lịch chính thức | S1 |

