# S01 — Mathematical Problem Formulation

**Research topic:** State-Preserving Intermittent Deep Learning via Dynamic Early-Exit and Non-Volatile Memory Mapping on Micro-Watt IoT Nodes.

**Related issue:** #1

**Reference:** [Research Roadmap](BaoKhanh_Planning_12_Month_Research_Roadmap.md), Sections 3.1, 7.2, 7.3, 8.2, 8.3 and 13.1.

**Status:** Proposed formulation — pending student verification and advisor review.

## 1. Research problem

Xét một thiết bị IoT không pin thực hiện suy luận DNN bằng năng lượng thu hoạch từ môi trường. Năng lượng được lưu trong tụ điện. Khi điện áp xuống dưới ngưỡng hoạt động, thiết bị mất điện và dữ liệu trong SRAM bị mất, trong khi dữ liệu đã được lưu hợp lệ trong NVM vẫn được bảo toàn.

Các phương pháp checkpoint và intermittent inference như Chinchilla, SONIC/TAILS và Stateful Neural Networks cung cấp cơ sở cho việc bảo toàn tiến độ. Zygarde, Wu DAC 2020 và HarvNet cung cấp cơ sở cho việc lựa chọn độ sâu suy luận theo năng lượng, độ chính xác và thời gian.

Bài toán đặt ra là lựa chọn khi nào tiếp tục suy luận, dừng sớm, lưu trạng thái hoặc bỏ qua mẫu nhằm tối đa hóa số dự đoán đúng và đúng hạn trên mỗi đơn vị năng lượng thu hoạch.

Phương pháp đề xuất nghiên cứu sự kết hợp giữa:

- PSB: Learned Persistent-State Bottleneck.
- ETCT: Energy-Trace-Conditioned Training.
- MDP: Energy-aware runtime decision policy.

Đây là hướng nghiên cứu cần kiểm chứng novelty, không phải đóng góp đã được xác nhận.

Phạm vi: algorithm, model, simulation và experimental validation trên tài nguyên tính toán hiện có; không triển khai phần cứng thật.

## 2. Mathematical notation

| Symbol | Definition | Unit |
|---|---|---|
| $x_i,y_i$ | Input sample and ground-truth label | — |
| $\hat y_i$ | Predicted label | — |
| $K,k$ | Number of blocks and current completed block | — |
| $h_k$ | Intermediate activation | — |
| $s_k$ | Compressed persistent state | — |
| $S_k$ | Persistent state size including metadata | byte |
| $c_k$ | Calibrated prediction confidence | — |
| $C$ | Capacitance | F |
| $V(t)$ | Capacitor voltage | V |
| $V_{on},V_{off},V_{max}$ | Power-on, power-off and maximum voltage | V |
| $U(t)$ | Physical stored energy | J |
| $E(t)$ | Usable stored energy | J |
| $P_h(t)$ | Harvested electrical power | W |
| $P_{load}(t)$ | Device power consumption | W |
| $\eta$ | Power conversion efficiency | — |
| $D_i,T_i$ | Relative deadline and completion latency | s |
| $\pi$ | Runtime decision policy | — |

## 3. Multi-exit DNN and persistent state

Cho DNN gồm K block, mỗi block có một exit head:

$$
h_0=x_i,\qquad h_k=B_k(h_{k-1})
$$

Tại exit k, xác suất phân loại được biểu diễn bởi $p_k(j\mid h_k)$.

$$
\hat y_k=\arg\max_j p_k(j\mid h_k)
$$

$$
c_k=\max_j p_k(j\mid h_k)
$$

Confidence cần được hiệu chỉnh trên tập validation.

Khi thực hiện PERSIST, PSB tạo trạng thái nén:

$$
s_k=Q_b(g_k(h_k)),\qquad
\hat h_k=r_k(s_k)
$$

Trong đó $g_k$ là encoder, $Q_b$ là phép lượng tử hóa b-bit và $r_k$ là decoder.

Khi tiếp tục suy luận bình thường, hệ thống sử dụng $h_k$. Sau mất điện, hệ thống sử dụng trạng thái khôi phục $\hat h_k$.

Mục tiêu PSB là giảm số byte cần lưu trong khi hạn chế suy giảm accuracy. Chi phí encoder và decoder phải được tính vào mô hình năng lượng.

ETCT sử dụng xác suất dừng tại mỗi exit, ước lượng từ development energy traces, để điều chỉnh quá trình huấn luyện multi-exit. Test traces không được sử dụng để huấn luyện hoặc lựa chọn policy.

## 4. Energy harvesting and capacitor dynamics

Năng lượng thu hoạch sau bộ chuyển đổi trong khoảng thời gian $\Delta t$:

$$
E_h(t,\Delta t)=
\int_t^{t+\Delta t}\eta P_h(u)\,du
$$

Với trace ánh sáng EnHANTs có đơn vị $\mu W/cm^2$, công suất điện được mô hình hóa bằng diện tích pin quang điện và hiệu suất chuyển đổi:

$$
P_h(t)=10^{-6}\eta_{pv}A_{pv}G(t)
$$

Trong đó $A_{pv}$ có đơn vị $cm^2$, $G(t)$ có đơn vị $\mu W/cm^2$ và $P_h(t)$ có đơn vị W.

Các tham số chuyển đổi phải được hiệu chỉnh, không xem mật độ công suất ánh sáng trực tiếp là công suất điện.

Năng lượng vật lý của tụ:

$$
U(t)=\frac12 CV(t)^2
$$

Năng lượng khả dụng trước ngưỡng mất điện:

$$
E(t)=\max\left(0,U(t)-\frac12 CV_{off}^2\right)
$$

Theo Roadmap Section 7.2, dynamics của tụ được biểu diễn:

$$
\frac{dU}{dt}=\eta P_h(t)-P_{load}(t)
$$

Với bước mô phỏng $\Delta t$:

$$
U_{t+\Delta t}=
\operatorname{clip}
\left(
U_t+[\eta P_h(t)-P_{load}(t)]\Delta t,
0,U_{max}
\right)
$$

$$
U_{max}=\frac12 CV_{max}^2,\qquad
V(t)=\sqrt{\frac{2U(t)}{C}}
$$

Quy tắc hysteresis:

- Bật nguồn khi $V(t)\ge V_{on}$.
- Tắt nguồn khi $V(t)\le V_{off}$.
- Giữ trạng thái nguồn trước đó khi điện áp nằm giữa hai ngưỡng.

Khi mất điện, simulator dừng hành động tại thời điểm xảy ra brownout, xóa trạng thái volatile và giữ checkpoint NVM hợp lệ gần nhất.

Năng lượng vượt quá dung lượng tụ được ghi nhận là overflow, không được tính là năng lượng tiêu thụ.

## 5. Computation and NVM costs

Chi phí tính toán block k:

$$
E_k^{comp}=\sum_{o\in\mathcal O_k}n_oe_o
$$

Trong đó $n_o$ là số phép toán loại o và $e_o$ là năng lượng tiêu thụ cho mỗi phép toán.

Chi phí lưu trạng thái:

$$
E_k^{pers}
=
E_k^{enc}+S_ke_{NVM}^{w}+e_w^{ovh}
$$

Chi phí khôi phục trạng thái:

$$
E_k^{rest}
=
S_ke_{NVM}^{r}+E_k^{dec}
$$

Tổng chi phí của một lần lưu, khởi động và khôi phục thành công:

$$
E_k^{checkpoint}
=
E_k^{pers}+E_{boot}+E_k^{rest}
$$

Trong đó:

- $e_{NVM}^{w},e_{NVM}^{r}$: năng lượng ghi và đọc mỗi byte (J/byte).
- $E_k^{enc},E_k^{dec}$: năng lượng mã hóa và giải mã.
- $e_w^{ovh}$: overhead ghi và xác nhận checkpoint.
- $E_{boot}$: năng lượng khởi động.

Chi phí boot được tính một lần cho mỗi lần khởi động thực tế.

Chi phí exit head, điều khiển, truy cập bộ nhớ, chờ và các lần thực thi thất bại cũng được tính khi mô phỏng. Không tính trùng chi phí giữa mô hình block và mô hình công suất.

Các tham số phần cứng hiện ở trạng thái NEEDS VERIFICATION, sẽ được hiệu chỉnh bằng datasheet và literature theo Roadmap.

## 6. State and action space

Mô hình chính sách được xây dựng dưới dạng MDP xấp xỉ với trạng thái:

$$
z_t=(\bar E_t,\bar P_{h,t},k,\bar c_k,\rho_t,d_t,q_t)
$$

Trong đó:

- $\bar E_t$: năng lượng đã rời rạc hóa.
- $\bar P_{h,t}$: mức công suất thu hoạch được ước lượng.
- $k$: block đã hoàn thành gần nhất.
- $\bar c_k$: confidence đã rời rạc hóa.
- $\rho_t$: thông tin checkpoint hợp lệ, gồm vị trí block đã lưu.
- $d_t$: thời gian còn lại trước deadline.
- $q_t$: trạng thái nguồn ON/OFF.

Với thời điểm mẫu đến là $t_i^{arr}$:

$$
d_t=\max(0,D_i-(t-t_i^{arr}))
$$

Tập hành động:

$$
\mathcal A=
\{\mathrm{CONT},\mathrm{EXIT},
\mathrm{PERSIST},\mathrm{SKIP}\}
$$

| Action | State transition |
|---|---|
| CONT | Chạy block tiếp theo; nếu thành công, cập nhật k và confidence |
| EXIT | Trả dự đoán tại exit hiện tại và kết thúc mẫu |
| PERSIST | Mã hóa, ghi checkpoint hợp lệ, sau đó chờ điều kiện tiếp tục |
| SKIP | Bỏ mẫu hiện tại, không nhận thưởng dự đoán |

Policy:

$$
a_t=\pi(z_t)
$$

Sau mỗi hành động có thời lượng $\Delta t_a$:

$$
d_{t+1}=\max(0,d_t-\Delta t_a)
$$

Năng lượng được cập nhật theo Section 4. Nếu mất điện, tiến độ volatile bị mất và hệ thống chỉ được khôi phục từ checkpoint hợp lệ.

Khi thiết bị OFF, quá trình sạc và khởi động là chuyển trạng thái tự động, không phải quyết định suy luận.

Mô hình chuyển trạng thái năng lượng được ước lượng từ development traces; chuyển confidence được ước lượng từ validation data. Giả định Markov cần được kiểm chứng trong quá trình xây simulator.

## 7. Optimization objective

Cho N mẫu đến trong khoảng mô phỏng cố định $T_{sim}$. Đặt $b_i=1$ nếu inference i hoàn thành và $b_i=0$ nếu không hoàn thành.

Số dự đoán đúng và đúng hạn:

$$
N_{correct,D}=
\sum_{i=1}^{N}
b_i\,
\mathbb 1[\hat y_i=y_i]\,
\mathbb 1[T_i\le D_i]
$$

Mẫu bị bỏ qua hoặc không hoàn thành có đóng góp bằng 0.

Năng lượng thu hoạch sau bộ chuyển đổi, trước tổn thất overflow:

$$
E_H=\int_0^{T_{sim}}\eta P_h(t)\,dt
$$

Mục tiêu chính:

$$
\boxed{
\max_{\pi}
J(\pi)=
\frac{\mathbb E[N_{correct,D}]}
{\mathbb E[E_H]}
}
$$

với $\mathbb E[E_H]>0$.

Đơn vị báo cáo: correct-completions/mJ.

Trong giai đoạn đầu, policy được tối ưu trên một DNN cố định. PSB và ETCT sẽ được đồng tối ưu ở các giai đoạn nghiên cứu tiếp theo.

Theo định hướng Roadmap Section 7.3, reward thực dụng có thể sử dụng:

$$
r_t=
\mathbb 1[a_t=\mathrm{EXIT}]
\mathbb 1[\Delta t_a\le d_t]
\widehat{\mathrm{Acc}}_k(c_k)
-\mu E_{used,t}
$$

Trong đó $\widehat{\mathrm{Acc}}_k(c_k)$ được ước lượng từ validation data, không sử dụng ground-truth label của mẫu test khi ra quyết định.

$\mu\ge0$ là trọng số phạt năng lượng, có đơn vị $J^{-1}$.

Reward này là mục tiêu thay thế có phạt năng lượng, không mặc định tương đương với hàm tỷ số J.

Policy dự kiến được giải offline bằng value iteration trên không gian trạng thái rời rạc và triển khai dưới dạng LUT. Kích thước LUT phải được kiểm tra sau khi bổ sung các biến trạng thái.

## 8. Constraints

**Energy:**

$$
0\le U(t)\le U_{max}
$$

Điều kiện đủ bảo thủ để thực hiện một hành động mà không phụ thuộc năng lượng thu trong tương lai:

$$
E_a+E_{margin}\le E(t)
$$

Trong đó $E_{margin}>0$ là biên an toàn.

**Memory:**

$$
M_{weights}+M_{checkpoint}+M_{LUT}
+M_{runtime,NVM}\le M_{NVM}
$$

$$
M_{activation}^{peak}+M_{workspace}
+M_{runtime,SRAM}\le M_{SRAM}
$$

**Deadline:**

$$
T_i\le D_i
$$

Đây là điều kiện để một inference được tính là đúng hạn. Thời gian bao gồm suy luận, lưu/khôi phục, khởi động và chờ sạc.

**Checkpoint correctness:**

Checkpoint chỉ được khôi phục nếu toàn bộ dữ liệu và metadata đã được ghi và xác nhận hợp lệ. Mất điện trong quá trình ghi không được làm mất checkpoint hợp lệ trước đó.

Cơ chế commit checkpoint cần được thiết kế và kiểm thử; đây là yêu cầu của mô hình, chưa phải bảo đảm đã được triển khai.

## 9. Evaluation metrics

| Metric | Definition |
|---|---|
| Accuracy | Correct predictions / completed inferences |
| Expected accuracy | Correct predictions / all arriving samples |
| ICR | Completed inferences / all arriving samples |
| Deadline completion rate | On-time completions / all arriving samples |
| Correct-completions/mJ | Correct on-time predictions / harvested energy in mJ |
| Energy per valid inference | Total consumed energy / on-time completed inferences |
| NVM bytes/inference | Total NVM bytes written / completed inferences |
| Persistence overhead | Persistence and restoration energy / total consumed energy |
| Latency | Completion time minus arrival time |
| Power failures | Number of unplanned power interruptions |
| Memory footprint | NVM and peak SRAM usage |

Các trường hợp có mẫu số bằng 0 được ghi là N/A.

So sánh các phương pháp phải sử dụng cùng dữ liệu đầu vào, energy traces, thời gian mô phỏng, năng lượng ban đầu và giả định phần cứng.

## 10. Assumptions

1. Nguồn năng lượng được biểu diễn bằng real hoặc synthetic traces có nguồn gốc và đơn vị rõ ràng.

2. SRAM mất dữ liệu khi mất điện; NVM bảo toàn checkpoint đã được ghi hợp lệ.

3. Chi phí tính toán và truy cập NVM được mô hình hóa bằng tham số cần hiệu chỉnh.

4. Trong formulation ban đầu, mỗi thời điểm chỉ xử lý một inference; mẫu đến khi thiết bị bận hoặc tắt nguồn được ghi nhận là missed nếu không thể tiếp nhận.

5. Policy chỉ sử dụng thông tin năng lượng hiện tại và quá khứ, không biết trước test trace.

6. Nghiên cứu sử dụng simulation, chưa tuyên bố kết quả triển khai trên MCU thật.

## 11. Numerical example

Các giá trị dưới đây chỉ dùng minh họa, không phải số liệu thực nghiệm.

Giả sử thiết bị đã khởi động xong, có:

| Parameter | Assumed value |
|---|---|
| Capacitance C | 100 uF |
| Power-on voltage | 3.0 V |
| Power-off voltage | 2.0 V |
| Maximum voltage | 3.3 V |
| Harvested electrical power | 100 uW |
| Conversion efficiency | 0.8 |
| Block duration | 0.1 s |
| Block energy cost | 120 uJ |
| Relative deadline | 1.0 s |

Năng lượng khả dụng ban đầu:

$$
E_0=\frac12(100\times10^{-6})(3^2-2^2)
=250\ \mu J
$$

Năng lượng thu thêm trong 0.1 s:

$$
E_h=0.8\times100\times0.1
=8\ \mu J
$$

Năng lượng còn lại sau block:

$$
E_1=250+8-120=138\ \mu J
$$

Điện áp tương ứng:

$$
V_1=
\sqrt{2^2+\frac{2(138\times10^{-6})}
{100\times10^{-6}}}
=2.6\ V
$$

Giả sử block tiếp theo cần 160 uJ. CONT không đáp ứng điều kiện an toàn bảo thủ vì 138 uJ nhỏ hơn 160 uJ.

Giả sử trạng thái cần lưu gồm 100 byte, chi phí ghi 0.2 uJ/byte, encoder 5 uJ và overhead ghi 5 uJ:

$$
E^{pers}=100\times0.2+5+5
=30\ \mu J
$$

Bỏ qua năng lượng thu thêm trong quá trình ghi:

$$
E_2=138-30=108\ \mu J
$$

Nếu chọn PERSIST và chờ sạc đến ngưỡng bật nguồn, thời gian sạc tối thiểu theo mô hình đơn giản, bỏ qua standby power, là:

$$
T_{charge}=
\frac{250-108}{0.8\times100}
=1.775\ s
$$

Deadline còn lại sau block đầu tiên là:

$$
d=1.0-0.1=0.9\ s
$$

Do thời gian sạc lớn hơn deadline còn lại, PERSIST rồi tiếp tục không thể hoàn thành mẫu đúng hạn trong ví dụ này.

Giả sử EXIT cần thêm 10 uJ và 0.01 s, EXIT là hành động khả thi về năng lượng và deadline.

Tuy nhiên, policy còn phải xem xét confidence và ảnh hưởng đến các mẫu tiếp theo trước khi quyết định EXIT hay SKIP.

Ví dụ này minh họa sự cần thiết của việc xét đồng thời năng lượng, chi phí persistence, thời gian sạc và deadline.

## 12. References and validation plan

- [L1 — Chinchilla](https://www.usenix.org/conference/osdi18/presentation/maeng): Adaptive checkpointing and forward progress.
- [L5 — SONIC/TAILS](https://doi.org/10.1145/3297858.3304011): Intermittent DNN execution and state preservation costs.
- [L7 — Stateful Neural Networks](https://doi.org/10.1109/TCAD.2022.3197513): Embedded inference-progress information.
- [L13 — Zygarde](https://doi.org/10.1145/3411808): Deadline-aware intermittent inference.
- [L14 — Wu DAC 2020](https://doi.org/10.1109/DAC18072.2020.9218526): Energy-aware multi-exit inference.
- [L15 — HarvNet](https://doi.org/10.1145/3581791.3596845): Resource-constrained architecture and inference policy.
- [L24 — Early-Exit Survey](https://doi.org/10.1145/3698767): Multi-exit architecture, training and exit policies.

Theo Roadmap Section 7.2, simulator cần được kiểm tra bằng unit tests bảo toàn năng lượng, nghiệm giải tích với công suất không đổi, xu hướng của ít nhất hai baseline và phân tích độ nhạy tham số.

Chưa có kết quả thực nghiệm trong tài liệu này. Novelty và các giả thuyết PSB, ETCT, MDP vẫn cần được kiểm chứng.

## AI assistance disclosure

ChatGPT được sử dụng để hỗ trợ tổng hợp Roadmap và literature, xây dựng bản nháp formulation, đề xuất ký hiệu, kiểm tra phép tính và trình bày Markdown.

Các mở rộng về trạng thái nguồn, deadline, checkpoint metadata, reward và điều kiện năng lượng là đề xuất mô hình cần được sinh viên kiểm chứng và cố vấn đánh giá.

Không tuyên bố đã tái lập thí nghiệm hoặc hoàn thành simulator.

**Status: Pending student verification and advisor review.**
