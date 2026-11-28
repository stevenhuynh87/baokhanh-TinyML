# Problem Formulation — State-Preserving Intermittent TinyML

**Sprint:** S01

**Related issue:** #1

**Research topic:** State-Preserving Intermittent Deep Learning via Dynamic Early-Exit and Non-Volatile Memory Mapping on Micro-Watt IoT Nodes.

**Status:** Proposed mathematical formulation — pending advisor review.

**Reference:** BaoKhanh_Planning_12_Month_Research_Roadmap.md, Sections 3.1, 7.2, 7.3, 8.2 and 13.1.

## 1. Research problem

Thiết bị IoT không pin sử dụng năng lượng thu hoạch từ môi trường để thực hiện suy luận DNN. Năng lượng được lưu trong tụ điện và có thể thay đổi theo thời gian.

Khi năng lượng không đủ, thiết bị bị mất điện. Dữ liệu trong SRAM bị mất, trong khi dữ liệu đã ghi thành công vào NVM (ví dụ FRAM) vẫn được giữ lại.

Việc lưu toàn bộ trạng thái trung gian có thể làm tăng chi phí năng lượng và thời gian suy luận. Ngược lại, nếu không lưu trạng thái, thiết bị có thể phải thực hiện lại các phép tính sau khi có điện.

Bài toán nghiên cứu là lựa chọn khi nào tiếp tục suy luận, dừng sớm, lưu trạng thái hoặc bỏ qua mẫu để tối đa hóa số dự đoán đúng hoàn thành trong deadline trên mỗi đơn vị năng lượng thu hoạch.

Phạm vi nghiên cứu là algorithm, model và simulation; chưa thực hiện triển khai trên phần cứng thật.

## 2. Mathematical notation

| Symbol | Definition | Unit |
|---|---|---|
| $x_i$ | Input sample | — |
| $y_i$ | Ground-truth label | — |
| $\hat y_i$ | Predicted label | — |
| $K$ | Number of DNN blocks/exits | — |
| $k$ | Current block index | — |
| $h_k$ | Intermediate activation | — |
| $s_k$ | Persistent compressed state | — |
| $c_k$ | Calibrated prediction confidence | — |
| $C$ | Capacitance | F |
| $V(t)$ | Capacitor voltage | V |
| $V_{on}$ | Power-on voltage threshold | V |
| $V_{off}$ | Power-off voltage threshold | V |
| $V_{max}$ | Maximum capacitor voltage | V |
| $P_h(t)$ | Harvested electrical power | W |
| $P_{load}(t)$ | Device power consumption | W |
| $\eta$ | Energy conversion efficiency | — |
| $E(t)$ | Usable stored energy | J |
| $D_i$ | Relative inference deadline | s |
| $T_i$ | Inference completion latency | s |
| $\pi$ | Runtime decision policy | — |

## 3. Multi-exit DNN model

Cho mạng DNN gồm K block:

$$
F=\{B_1,B_2,\ldots,B_K\}
$$

Trạng thái đầu vào:

$$
h_0=x_i
$$

Activation sau block thứ k:

$$
h_k=B_k(h_{k-1})
$$

Dự đoán tại exit thứ k:

$$
\hat y_k=\arg\max_j p_k(j\mid h_k)
$$

Độ tin cậy tại exit:

$$
c_k=\max_j p_k(j\mid h_k)
$$

Trong đó xác suất đầu ra cần được hiệu chỉnh trên tập validation.

Mạng multi-exit cho phép trả kết quả tại một block trung gian thay vì bắt buộc thực hiện toàn bộ K block.

## 4. Energy harvesting and capacitor dynamics

### 4.1 Harvested energy

Năng lượng thu được trong khoảng thời gian từ t đến t + dt:

$$
E_h(t,\Delta t)
=
\int_t^{t+\Delta t}\eta P_h(u)\,du
$$

Nếu công suất gần như không đổi trong bước mô phỏng:

$$
E_h(t,\Delta t)
\approx \eta P_h(t)\Delta t
$$

Đối với trace ánh sáng EnHANTs, công suất điện thu hoạch được mô hình hóa như sau:

$$
P_h(t)=\eta_{pv}A_{pv}G(t)
$$

Trong đó G(t) là mật độ công suất ánh sáng, A_pv là diện tích pin mặt trời và eta_pv là hiệu suất chuyển đổi quang điện.

Các đơn vị phải được chuyển đổi nhất quán trước khi tính toán.

### 4.2 Capacitor energy

Năng lượng vật lý trong tụ:

$$
U(t)=\frac{1}{2}CV(t)^2
$$

Năng lượng có thể sử dụng trước khi đạt ngưỡng mất điện:

$$
E(t)=
\max\left(0,
\frac{1}{2}C
\left[V(t)^2-V_{off}^2\right]
\right)
$$

Mô hình biến thiên năng lượng:

$$
\frac{dU(t)}{dt}
=
\eta P_h(t)-P_{load}(t)
$$

Dạng rời rạc cho simulator:

$$
U_{t+\Delta t}
=
\min\left(
U_{max},
\max\left(
0,U_t+
[\eta P_h(t)-P_{load}(t)]\Delta t
\right)
\right)
$$

Trong đó:

$$
U_{max}=\frac{1}{2}CV_{max}^2
$$

Điện áp được khôi phục từ năng lượng vật lý:

$$
V(t)=\sqrt{\frac{2U(t)}{C}}
$$

Quy tắc hoạt động:

- Thiết bị bật khi V(t) >= V_on.
- Thiết bị tắt khi V(t) <= V_off.
- Khi điện áp nằm giữa hai ngưỡng, thiết bị giữ trạng thái nguồn trước đó.
- SRAM mất trạng thái khi thiết bị tắt.
- NVM giữ những checkpoint đã được ghi thành công.

## 5. Computation and NVM energy costs

### 5.1 Computation cost

Năng lượng thực hiện block thứ k:

$$
E_k^{comp}
=
\sum_{o\in\mathcal O_k}n_o e_o
$$

Trong đó n_o là số phép toán loại o và e_o là năng lượng tiêu thụ cho mỗi phép toán.

Chi phí exit head, truy cập bộ nhớ và overhead điều khiển phải được tính trong mô hình chi phí tương ứng, không tính trùng.

### 5.2 Persistent-State Bottleneck (PSB)

Thay vì lưu toàn bộ activation h_k, hệ thống nghiên cứu cách học một biểu diễn trạng thái nén:

$$
s_k=Q_b(g_k(h_k))
$$

Khôi phục activation:

$$
\hat h_k=r_k(s_k)
$$

Trong đó:

- g_k: bộ mã hóa trạng thái.
- Q_b: lượng tử hóa b-bit.
- r_k: bộ giải mã trạng thái.
- s_k: trạng thái được lưu vào NVM.

Việc nén trạng thái có thể làm thay đổi activation sau khôi phục. Do đó, ảnh hưởng lên độ chính xác cần được đánh giá thực nghiệm.

### 5.3 NVM writing cost

$$
E_k^{persist}
=
E_k^{enc}
+
S_k e_{NVM}^{w}
+
e_w^{ovh}
$$

Trong đó:

- S_k: tổng số byte cần ghi, bao gồm trạng thái và metadata checkpoint.
- e_NVM^w: năng lượng ghi trên mỗi byte.
- E_k^enc: năng lượng mã hóa trạng thái.
- e_w^ovh: overhead của thao tác ghi.

### 5.4 NVM restoration cost

$$
E_k^{restore}
=
S_k e_{NVM}^{r}
+
E_{boot}
+
E_k^{dec}
$$

Trong đó e_NVM^r là năng lượng đọc trên mỗi byte, E_boot là chi phí khởi động và E_k^dec là chi phí giải mã.

Chi phí boot chỉ được tính một lần cho mỗi lần khởi động.

Tổng chi phí lưu và khôi phục:

$$
E_k^{checkpoint}
=
E_k^{persist}
+
E_k^{restore}
$$

Các tham số chi phí cần được hiệu chỉnh bằng datasheet và số liệu công bố trước khi sử dụng để báo cáo kết quả.

## 6. Markov Decision Process (MDP)

### 6.1 State space

Trạng thái tại một thời điểm ra quyết định:

$$
z_t=
(\bar E_t,\bar P_{h,t},k,\bar c_k,\rho_t,d_t)
$$

Trong đó:

- E_bar: mức năng lượng đã rời rạc hóa.
- P_h_bar: mức công suất thu hoạch được ước lượng.
- k: block đang thực hiện.
- c_bar: độ tin cậy đã rời rạc hóa.
- rho_t: thông tin checkpoint hợp lệ gần nhất.
- d_t: thời gian còn lại trước deadline.

$$
d_t=\max(0,D_i-(t-t_i^{arrival}))
$$

### 6.2 Action space

$$
\mathcal A=
\{\mathrm{CONT},\mathrm{EXIT},
\mathrm{PERSIST},\mathrm{SKIP}\}
$$

| Action | Description |
|---|---|
| CONT | Tiếp tục thực hiện block k+1 nếu k < K |
| EXIT | Trả kết quả tại exit hiện tại |
| PERSIST | Lưu trạng thái và chờ đủ năng lượng để tiếp tục |
| SKIP | Bỏ mẫu hiện tại |

Policy:

$$
a_t=\pi(z_t)
$$

Policy chỉ sử dụng thông tin hiện tại và thông tin quá khứ, không được biết trước toàn bộ trace năng lượng test.

### 6.3 State transition

Năng lượng được cập nhật theo mô hình tụ điện ở Section 4.

Độ tin cậy tại block tiếp theo được mô hình hóa bằng phân phối thực nghiệm:

$$
p(\bar c_{k+1}\mid\bar c_k)
$$

Phân phối này được ước lượng trên tập validation.

Mô hình Markov của công suất thu hoạch được xây dựng từ development traces.

Nếu thiết bị mất điện, hệ thống chỉ được khôi phục từ checkpoint hợp lệ gần nhất. Nếu không có checkpoint hợp lệ, inference phải khởi động lại hoặc bị bỏ qua.

### 6.4 Reward function

$$
r_t=
\mathbb 1[a_t=\mathrm{EXIT}]
\mathbb 1[T_i\le D_i]
\widehat{\mathrm{Acc}}_k(c_k)
-
\mu E_{used,t}
$$

Trong đó:

- Acc_hat là xác suất dự đoán đúng được ước lượng từ confidence trên validation.
- E_used,t là năng lượng tiêu thụ của hành động.
- mu là trọng số chi phí năng lượng.

Phần thưởng cho dự đoán chỉ được tính khi EXIT, tránh tính nhiều lần cho cùng một mẫu.

## 7. Optimization objective

Cho N mẫu đầu vào trong một khoảng mô phỏng cố định.

Số dự đoán đúng hoàn thành trong deadline:

$$
N_{correct,D}
=
\sum_{i=1}^{N}
\mathbb 1[
\hat y_i=y_i
\land T_i\le D_i
\land i\text{ completed}
]
$$

Năng lượng thu hoạch khả dụng tại đầu ra bộ chuyển đổi, trước khi tính phần dư bị tràn:

$$
E_H=
\int_0^{T_{sim}}\eta P_h(t)\,dt
$$

Hàm mục tiêu chính:

$$
\max_{\pi}
\quad
J(\pi)=
\frac{\mathbb E[N_{correct,D}]}
{\mathbb E[E_H]}
$$

với E_H > 0.

Đơn vị báo cáo là số dự đoán đúng và đúng hạn trên mỗi mJ năng lượng thu hoạch.

Policy ban đầu được tối ưu trên một mạng đã xác định. Các giai đoạn nghiên cứu tiếp theo sẽ xem xét đồng tối ưu PSB, ETCT và policy.

Trong cùng một tập trace, thời gian mô phỏng và điều kiện năng lượng ban đầu phải giống nhau giữa các phương pháp.

## 8. Constraints

### C1. Energy feasibility

$$
0\le U(t)\le U_{max}
$$

Thiết bị chỉ thực hiện tính toán khi đang bật và có đủ năng lượng.

Một điều kiện an toàn bảo thủ cho hành động:

$$
E_{action}\le E(t)
$$

Điều kiện này không dựa vào năng lượng chưa chắc chắn sẽ thu được trong tương lai.

### C2. Memory capacity

$$
M_{weights}+M_{state}+M_{LUT}+M_{runtime}
\le M_{NVM}
$$

$$
M_{activation}^{peak}
+
M_{workspace}
\le M_{SRAM}
$$

### C3. Deadline

Một inference chỉ được tính là đúng hạn khi:

$$
T_i\le D_i
$$

Thời gian suy luận bao gồm thời gian tính toán, quyết định, lưu trạng thái, đọc và giải mã trạng thái, khởi động và chờ sạc.

### C4. Checkpoint consistency

Checkpoint mới chỉ được sử dụng sau khi quá trình ghi hoàn thành và được xác nhận hợp lệ.

Nếu mất điện giữa quá trình ghi, checkpoint chưa hoàn thành không được sử dụng để khôi phục.

Cơ chế đảm bảo tính nhất quán cụ thể cần được thiết kế và kiểm thử trong simulator.

## 9. Evaluation metrics

| Metric | Definition |
|---|---|
| Accuracy | Correct predictions / completed inferences |
| Expected accuracy | Correct predictions / all arriving samples |
| ICR | Completed inferences / all arriving samples |
| Correct-completions/mJ | Correct predictions completed within deadline / harvested energy in mJ |
| Energy per valid inference | Total consumed energy / number of valid completed inferences |
| NVM bytes per inference | Total bytes written to NVM / completed inferences |
| Persistence overhead | Energy spent on saving and restoring states / total consumed energy |
| Latency | Time from input arrival to prediction, including recharge |
| Deadline completion rate | Inferences completed within deadline / all arriving samples |
| Power failures | Total number of power interruptions |

Các trường hợp có mẫu số bằng 0 được báo cáo là N/A thay vì tạo giá trị không hợp lệ.

## 10. Assumptions

1. Nguồn năng lượng được biểu diễn bằng các power traces thật hoặc synthetic có khai báo nguồn rõ ràng.

2. Mất điện làm mất dữ liệu trong SRAM, nhưng các checkpoint hợp lệ trong NVM được giữ lại.

3. Chi phí tính toán, ghi, đọc và khởi động được mô hình hóa bằng các tham số năng lượng cần hiệu chỉnh.

4. Mạng DNN được chia thành K block và có các exit tại những ranh giới đã xác định.

5. Deadline được xác định trước cho từng tác vụ. Policy không biết trước toàn bộ trace test.

6. Nghiên cứu hiện tại chỉ thực hiện bằng simulation. Không tuyên bố đã triển khai hoặc đo trên phần cứng thật.

## 11. Numerical example

Các giá trị sau chỉ là giả định minh họa cho mô hình toán học, không phải số liệu đo thực nghiệm.

Cho:

| Parameter | Assumed value |
|---|---|
| C | 100 uF |
| V_on | 3.0 V |
| V_off | 2.0 V |
| V_max | 3.3 V |
| P_h | 100 uW |
| Conversion efficiency | 0.8 |
| Time interval | 0.1 s |
| Computation energy | 120 uJ |

Năng lượng khả dụng ban đầu:

$$
E_0=
\frac{1}{2}(100\times10^{-6})(3^2-2^2)
=
250\ \mu J
$$

Năng lượng thu thêm:

$$
E_h=0.8\times100\times0.1
=
8\ \mu J
$$

Năng lượng còn lại sau khi thực hiện block:

$$
E_1=250+8-120=138\ \mu J
$$

Điện áp tương ứng:

$$
V_1=
\sqrt{2^2+
\frac{2(138\times10^{-6})}
{100\times10^{-6}}}
=
2.6\ V
$$

Giả sử block tiếp theo cần 160 uJ. Theo điều kiện an toàn bảo thủ, CONT chưa khả thi vì:

$$
138<160
$$

Nếu trạng thái cần lưu có kích thước 100 byte, năng lượng ghi mỗi byte là 0.2 uJ, overhead ghi là 5 uJ và chi phí mã hóa là 5 uJ:

$$
E^{persist}
=
100\times0.2+5+5
=
30\ \mu J
$$

Sau khi lưu trạng thái:

$$
E_2=138-30=108\ \mu J
$$

Hệ thống có thể lựa chọn PERSIST, sau đó chờ tích lũy năng lượng.

Nếu exit hiện tại cần 10 uJ và có thể hoàn thành trước deadline, EXIT cũng là một hành động khả thi.

Policy phải lựa chọn giữa các hành động khả thi dựa trên confidence, deadline và chi phí năng lượng dự kiến.

## 12. Research scope and validation

Ba hướng nghiên cứu chính:

- PSB: giảm kích thước trạng thái cần lưu và chi phí NVM.
- ETCT: huấn luyện multi-exit theo phân phối năng lượng.
- MDP: tối ưu quyết định CONT / EXIT / PERSIST / SKIP.

Simulator cần được kiểm tra bằng:

- Unit test bảo toàn năng lượng.
- So sánh với nghiệm giải tích khi công suất không đổi.
- Đối chiếu xu hướng của ít nhất hai baseline đã công bố.
- Phân tích độ nhạy của các tham số năng lượng.

Các công trình Chinchilla (L1), SONIC/TAILS (L5), Stateful Neural Networks (L7), Zygarde (L13), Wu DAC 2020 (L14) và HarvNet (L15) được sử dụng làm tài liệu tham khảo cho việc xây dựng và đánh giá mô hình.

Research gap và tính mới của phương pháp vẫn cần được xác nhận qua novelty review.

## 13. AI assistance and verification status

Bản formulation được xây dựng với sự hỗ trợ của ChatGPT trong việc tổng hợp Roadmap, đề xuất ký hiệu, xây dựng công thức, kiểm tra tính nhất quán và trình bày Markdown.

Các bổ sung như deadline-aware state, điều kiện an toàn năng lượng và ví dụ số là đề xuất mô hình cần được sinh viên kiểm tra, hiểu rõ và trao đổi với cố vấn.

Không tuyên bố đây là công trình đã được kiểm chứng thực nghiệm hoặc được xây dựng hoàn toàn độc lập với AI.

**Status: Pending student verification and advisor review.**
