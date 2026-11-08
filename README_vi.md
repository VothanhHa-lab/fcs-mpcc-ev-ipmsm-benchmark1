# AI-FCS-MPCC: tái lập lại kết quả mô phỏng trên động cơ cầu sau Tesla Model 3

Dự án Python tái lập kết quả cho bản thảo *"AI-Based Adaptive Weighting FCS-MPCC for PMSM Drives in Electric Vehicles"*. Khác bản thảo, mọi số liệu và hình ở đây đều sinh ra từ mô phỏng thật, không có giá trị nào điền tay.

> **Mức kiểm chứng:** chỉ là mô phỏng offline (MIL), **không phải** HIL hay thực nghiệm. Thông số động cơ lấy từ phép đo và phân tích công khai của MotorXP. Ld và Lq là giá trị **suy ra** tại điểm mô-men đỉnh (trạng thái bão hòa), không phải số đo (xem bảng thông số).

## 1. Chạy lại

```bash
pip install -r requirements.txt
python3 tests/test_sanity.py          # 41 kiểm tra vật lý / số học
python3 scripts/reproduce.py          # toàn bộ: 15615 lần chạy, ~135 s trên 2 nhân CPU (trong đó mục 6c: 14184 lần chạy, ~90 s)
python3 scripts/vsf_baselines.py      # chỉ riêng mục 6c (baseline PWM tần số thay đổi, mục 3.4); dùng lại dữ liệu mô phỏng chung
python3 scripts/reproduce.py --quick  # kiểm tra nhanh pipeline
python3 scripts/derive_tesla_params.py  # suy Ld, Lq, psi_f, J từ dữ liệu MotorXP
```

Đầu ra:
- `results/figures/`: hình, mỗi hình có bản PDF vector và PNG 300 dpi.
- `results/tables/`: bảng CSV. Tệp `*_all_seeds.csv` là dữ liệu từng lần chạy; tệp `Table*.csv` là bảng tổng hợp.
- `results/raw/`: log đầy đủ của seed 0.
- `results/training_*/`: tập lưới huấn luyện, trọng số mạng nơ-ron, báo cáo chia tập.
- `results/run_manifest.json`: thông tin từng lần chạy (seed, sai lệch tham số, Ts, trạng thái, thời gian) và phiên bản thư viện.

## 2. Mô hình

| Thành phần | Cách mô phỏng |
|---|---|
| Động cơ | IPMSM mô hình dq, tham số hằng (chưa có bảng tra bão hòa). Tích phân Euler với bước con h = 0,25 µs; đã kiểm tra hội tụ khi giảm h còn 0,125 µs |
| Nghịch lưu | 2 mức, **chuyển mạch thật** (không dùng mô hình trung bình). Dead-time từng pha, điện áp trong dead-time phụ thuộc chiều dòng |
| Tổn hao | Đồng: 1,5·Rs·(id²+iq²). Dẫn: V0·\|i\| + r_on·i². Chuyển mạch: E_ref·(V/V_ref)·(\|i\|/I_ref) mỗi lần chuyển mạch. **Sắt stator (răng + gông) có xét sóng hài PWM**: xem mục 2.1 (`src/aifcs/ironloss.py`). **Chưa có tổn hao sắt rotor/nam châm và tổn hao cơ** |
| Điều khiển | Lấy mẫu Ts, giữ ZOH, **trễ tính toán 1 chu kỳ**, có bù trễ bằng dự báo 2 bước cho DB và FCS. Nhiễu đo dòng σ = 2 A, nhiễu tốc độ σ = 0,5 rpm |
| Vòng ngoài | Cùng một bộ PI tốc độ cho mọi bộ điều khiển dòng. Bộ tạo dòng đặt gồm MTPA, **suy giảm từ thông theo elip điện áp** (biên 0,9·Vdc/√3) và giới hạn vòng tròn dòng. Giới hạn mô-men 430 N·m và công suất 211 kW. Bộ quan sát mô-men tải dùng chung |
| Cơ | Kịch bản bàn thử (S1, S4): tốc độ do dyno giữ. Kịch bản xe (S2, S3, S5, MC): J = rotor + quán tính xe quy đổi (2,27 kg·m²) |

### 2.1 Mô hình tổn hao sắt stator có sóng hài PWM (mới; trả lời phản biện M2)
Tính từ chính dạng sóng mô phỏng, theo Bertotti dạng miền thời gian cho mỗi kg thép, áp dụng riêng cho răng và gông:

- Từ thông móc vòng: ψd = Ld·id + ψf, ψq = Lq·iq (dòng thật và L thật của plant). Đạo hàm từ thông ở hệ tĩnh: dψ/dt = v − Rs·i, với v là **điện áp đóng cắt thật** (kể cả dead-time), được tích lũy **ở mọi bước con 0,25 µs** trong `core.simulate` (cột `FDP2`, `FDP15` của log mịn, trung bình trong mỗi khoảng 2,5 µs). Nhờ đó dB/dt của xung PWM được phân giải chính xác, không bị giới hạn bởi tần số lấy mẫu log 400 kHz (xem kiểm tra trong `tests/test_sanity.py`: sai khác giữa sai phân hữu hạn 2,5 µs và giá trị chính xác chỉ 4–8 %).
- Từ thông → B: B tại một vị trí φ của stator là K·(ψ·u_φ). Trung bình theo φ cho ⟨(dB/dt)²⟩ = K²|ψ̇|²/2 và ⟨|dB/dt|^1,5⟩ = K^1,5·c15·|ψ̇|^1,5 (c15 = 0,5564). Với từ thông quay hình sin, các biểu thức này trùng chính xác công thức Bertotti (có kiểm tra).
- Hệ số K từ hình học: số vòng nối tiếp mỗi pha Ns = 18 cuộn × 2 vòng / 3 nhánh song song = **12**; hệ số dây quấn k_w1 = 0,95 [ASM]; Φ_cực = ψ/(Ns·k_w1); B_gông = Φ_cực/(2·h_y·L·k_xếp); từ thông răng = (π/2)·(2p/Q)·Φ_cực. Với ψf = 0,0772 Wb: Φ_cực = 6,77 mWb, B khe hở cơ bản 1,01 T (hợp lý), **B răng không tải 1,66 T, B gông 1,48 T** (đạt mục tiêu 1,6–1,7 và 1,4–1,5 T).
- Thành phần: từ trễ cơ bản k_h·f·B²; từ trễ vòng nhỏ của sóng hài PWM Σ k_h·f_n·B_n² (FFT của gợn từ thông, dạng Parseval, α = 2); dòng xoáy cổ điển k_c/(2π²)·⟨(dB/dt)²⟩ và dư thừa k_e/C_e·⟨|dB/dt|^1,5⟩ (C_e = 8,763) trong miền thời gian. Với phần sóng hài của dòng xoáy và dư thừa có nhân hệ số che chắn hiệu ứng bề mặt của lá thép F(d/δ) (μr hiệu dụng 300 [ASM]; chỉ đáng kể trên khoảng 20 kHz). Hệ số chế tạo (build factor) 1,7 nhân toàn bộ.
- "Phần cơ bản" = tổn hao của từ thông quay cơ bản thuần túy; "phần sóng hài" = phần tăng thêm do gợn PWM.

**Thông số tổn hao sắt** (`configs/default.json`, khóa `iron`; file `results/tables/iron_model_parameters.csv`):

| Đại lượng | Giá trị | Nguồn |
|---|---|---|
| Đường kính ngoài / trong lá thép stator, chiều dài lõi, số rãnh | 225 / 151,3 mm; 134 mm; 54 | MXP |
| Hệ số xếp lá | 0,95 | ASM (theo yêu cầu) |
| Số vòng/cuộn, nhánh song song | 2; 3 | MXP (theo đề bài) |
| Số cuộn mỗi pha | 18 (hai lớp, một cuộn mỗi rãnh) | ASM |
| k_w1 | 0,95 (dải 0,93–0,96) | ASM |
| Bề rộng răng | 5,6 mm | **ASM**, chọn để B răng không tải ≈ 1,66 T |
| Chiều cao gông | 18 mm (chiều sâu rãnh còn lại 18,85 mm; bỏ qua đầu răng) | **ASM**, chọn để B gông không tải ≈ 1,48 T |
| Khối lượng răng / gông | 5,55 / 11,40 kg (thép 7650 kg/m³) | DER từ hình học ASM |
| Thép | NO 0,30 mm; σ = 2,0 MS/m | ASM |
| k_h (α = 2) | 0,01067 W·s/(kg·T²) | **ASM**, dải 0,007–0,015 |
| k_c | 3,87·10⁻⁵ W·s²/(kg·T²) | DER: π²σd²/(6ρ) với d = 0,30 mm |
| k_e | 3,8·10⁻⁴ W·s^1,5/(kg·T^1,5) | **ASM**, dải 2–8·10⁻⁴ |
| Hệ số chế tạo | 1,7 (dải 1,5–2) | ASM |
| μr hiệu dụng (che chắn) | 300 | ASM |

Không có datasheet cụ thể để trích dẫn. k_h và k_e được chọn sao cho P(1,5 T, 50 Hz) = 1,7 W/kg và P(1,0 T, 400 Hz) = 13,5 W/kg, cỡ của thép NO mỏng 0,30 mm (hai giá trị này là mức điển hình theo hiểu biết chung, **không** có nguồn trích dẫn). Mọi kết luận được kiểm tra bằng độ nhạy (mục 3.4).

### Thông số và nguồn
Ký hiệu nguồn: [MXP] whitepaper MotorXP về động cơ cầu sau Tesla Model 3 (2018); [EVS] evspecifications.com cho Model 3 RWD 2019; [DER] suy ra trong dự án; [ASM] giả thiết.

| Đại lượng | Giá trị | Nguồn |
|---|---|---|
| Rs | 4,75 mΩ (20 °C) | MXP |
| Số đôi cực p | 3 | MXP |
| ψf | 0,0772 Wb | DER từ hằng số back-EMF 42 Vpk,LL/krpm (MXP) |
| Ld / Lq | 81,1 / 113,1 µH | **DER** từ điểm 430 N·m, 1000 Arms, góc 53,5°, 94,61 Vrms, 192 kW (MXP). Là giá trị **bão hòa**; khi tải nhẹ, L thực lớn hơn |
| Imax | 1414 A đỉnh (1000 Arms) | MXP |
| J rotor | 0,048 kg·m² | DER từ kích thước lá thép rotor (MXP), coi rotor đặc |
| Vdc | 360 V | EVS |
| m, Cd, tỉ số truyền | 1611 kg, 0,23, 9 | EVS |
| Rw, A, Cr | 0,334 m, 2,22 m², 0,010 | ASM |
| Tổn hao nghịch lưu | E_ref = 30 mJ @ 360 V / 1 kA; r_on = 1,5 mΩ | **ASM**, cỡ độ lớn của nghịch lưu SiC; không lấy từ datasheet cụ thể |
| Mô-men / công suất đỉnh | 430 N·m / 211 kW | MXP / EVS |
| Chu trình | WLTC Class 3b: 1801 s, 23,27 km, đỉnh 131,3 km/h | gói `wltp` (JRC), tệp `data/wltc_class3b.csv` |
| Ts | 50 µs (như bản thảo) | — |

## 3. Kết quả chính (trung bình ± độ lệch chuẩn, 5 seed nhiễu; Monte Carlo 40 mẫu)

**Bộ điều khiển so sánh** (Ts = 50 µs cho tất cả):
- PI-SVPWM;
- Deadbeat-SVPWM;
- FCS-MPCC với trọng số (1,1), là baseline trong bản thảo;
- **FCS-MPCC với cặp trọng số cố định tốt nhất** (0,2; 5), lấy từ chính lưới tìm kiếm offline;
- AI-FCS-MPCC theo đúng quy trình của bản thảo (mạng 4-10-2, nhãn từ lưới λ, chia tập theo điều kiện vận hành).

| Chỉ tiêu | PI | DB | FCS (1,1) | FCS cố định tốt nhất | AI-FCS (bài) |
|---|---|---|---|---|---|
| S1: RMSE iq (A) | 26,0 | 26,0 | 37,4 | 29,2 | 29,5 |
| S1: vọt lố (A) | 2,2 | 1,2 | 63 | 41,9 | 41,1 |
| S4: T_rms ripple (N·m) @1500 rpm, 200 N·m | 0,99 | 1,06 | 10,2 | 7,79 | 7,59 |
| S4: THD dòng pha | 0,75 % | 0,71 % | 7,4 % | 11,9 % | 10,0 % |
| f_sw trung bình (kHz) | 20 | 20 | 1,7 | 2,4 | 2,1 |
| Monte Carlo: RMSE iq (A) | 5,67 | 5,52 | 26,6 | 17,5 | 18,0 |

**Kết luận rút ra từ dữ liệu:**
1. **Mạng nơ-ron không cho lợi ích có hệ thống.** Theo RMSE dòng i_q, AI-FCS kém hơn cặp trọng số cố định tốt nhất ở **cả 40/40** mẫu Monte Carlo (RMSE trung bình lớn hơn 0,53 A; Wilcoxon p ≈ 2·10⁻¹², đây là giá trị nhỏ nhất có thể đạt với 40 cặp). Theo gợn mô-men thì kết quả lẫn lộn: AI-FCS kém hơn ở 31/40 mẫu Monte Carlo, nhưng nhỉnh hơn 2,5 % ở S4 và 5–16 % trong phép thử độ nhạy theo L ở 300 N·m. Chưa thể quy phần nhỉnh hơn này cho tính thích nghi, vì chưa thử cặp trọng số cố định riêng cho từng điểm vận hành. Lưu ý thêm: cặp cố định tốt nhất (0,2; 5) nằm ở biên lưới tìm kiếm. Hình FigA cho thấy đầu ra của mạng gần như là một tỉ số λd/λq không đổi, khoảng 0,06–0,07. Kiểm tra `tests/test_sanity.py` xác nhận chỉ tỉ số này ảnh hưởng tới lựa chọn trạng thái chuyển mạch.
2. **Ở Ts = 50 µs, FCS-MPCC thua xa PI-SVPWM** trên động cơ Tesla. Nguyên nhân là L chỉ khoảng 0,1 mH, nên mỗi chu kỳ dòng thay đổi cỡ 100 A. FCS-MPCC khi đó chỉ chuyển mạch khoảng 1,5–2,5 kHz, trong khi PI-SVPWM chuyển mạch 20 kHz. FCS chỉ thắng nếu so với chính FCS dùng trọng số (1,1).
3. **Khi cùng tần số chuyển mạch** (Bảng 11, Hình 8), PI-SVPWM vẫn có gợn mô-men và THD thấp hơn FCS-MPCC một chút. Ví dụ, ở khoảng 10 kHz: 1,75 N·m / 1,1 % so với 1,98 N·m / 1,5 %.
4. **Khả năng chịu sai lệch tham số** (Bảng 9, Bảng 10): PI và DB ít nhạy với sai lệch tham số hơn các biến thể FCS. Biến thể FCS nhạy nhất với L −20%.
5. **Hướng cải tiến λ_sw** (phạt số lần chuyển mạch; Ts = 10 µs; thử trên các điểm vận hành không dùng khi huấn luyện):
   - Với giới hạn gợn tuyệt đối 3 % mô-men đỉnh, λ_sw tối ưu như nhau ở mọi điểm vận hành, nên lập lịch bằng mạng nơ-ron hay bảng tra là thừa. λ_sw cố định = 1000 giảm 14 % tổng tổn hao so với λ_sw = 0 (4,34 so với 5,04 kW), nhưng gợn tăng từ 2,1 lên 6,7 N·m.
   - Ở các điểm vận hành tải trung bình và cao này, **PI-SVPWM 3,6 kHz trội hơn**: 4,25 kW và 5,8 N·m. Trên WLTC thì ngược lại (mục 3.1), vì chu trình chủ yếu chạy ở tải nhẹ.
   - Với giới hạn tương đối (3 % mô-men đang vận hành), bảng tra cho λ_sw thay đổi theo điểm vận hành và giảm tổn hao 10 % so với λ_sw = 0. Mạng nơ-ron ngoại suy sai ở một điểm tốc độ cao.
6. **Bản đồ hiệu suất** (Hình 7) được mô phỏng trên toàn đường bao mô-men–tốc độ tới 9500 rpm, có suy giảm từ thông.

### 3.1 Năng lượng trên chu trình WLTC Class 3b (Bảng 14)
Tính tựa-tĩnh theo từng giây, dùng bản đồ tổn hao mô phỏng của từng bộ điều khiển.

| Bộ điều khiển | Tổn hao truyền động (Wh/km) | Gợn mô-men trung bình (N·m) | Gợn / mô-men yêu cầu |
|---|---|---|---|
| PI-SVPWM 10 kHz | 4,08 | 1,93 | 10,9 % |
| PI-SVPWM 3,6 kHz | 3,66 | 5,84 | 32,7 % |
| FCS-MPCC, λ_sw = 0 | 4,93 | 1,84 | 10,1 % |
| FCS-MPCC, λ_sw cố định = 1000 | **2,82** | 6,82 | 37,7 % |
| FCS-MPCC + bảng tra / mạng nơ-ron λ_sw | 2,82 / 2,83 | 6,82 / 6,81 | 37,7 % |

Phạt số lần chuyển mạch giảm tổn hao truyền động 31 % so với PI-SVPWM 10 kHz và 23 % so với PI 3,6 kHz trên chu trình. Đổi lại, gợn mô-men ở tải nhẹ lớn hơn nhiều. Lập lịch bằng mạng nơ-ron không tốt hơn λ_sw cố định.

Với luật giới hạn gợn tương đối, mạng nơ-ron ngoại suy sai ở một điểm vận hành tốc độ cao không dùng khi huấn luyện: gợn lên 33,6 N·m (Bảng 12, luật `rel`). Bảng tra không gặp lỗi này.

### 3.2 Độ nhạy theo mô hình tổn hao nghịch lưu (Bảng 14b, Bảng 14c)
Tổn hao truyền động trên WLTC, đơn vị Wh/km:

| Bộ điều khiển | Danh định | Esw ×0,5 | Esw ×2 | r_on ×0,5 | r_on ×2 |
|---|---|---|---|---|---|
| PI-SVPWM 10 kHz | 4,08 | 3,14 | 5,96 | 3,82 | 4,61 |
| PI-SVPWM 3,6 kHz | 3,66 | 3,25 | 4,48 | 3,32 | 4,34 |
| FCS-MPCC λ_sw = 1000 | 2,82 | 2,47 | 3,51 | 2,56 | 3,32 |

Thứ hạng giữ nguyên trong cả 5 trường hợp, và λ_sw cố định tối ưu vẫn là 1000. Mô hình tổn hao không ảnh hưởng tới động học nên có thể nhân hệ số trực tiếp mà không cần chạy lại.

### 3.3 Độ nhạy theo điện cảm, thay cho mô hình bão hòa (Bảng 15)
Nhân L (cả plant lẫn mô hình trong bộ điều khiển) với ×1; ×1,5; ×2, đại diện cho vùng ít bão hòa hơn khi tải nhẹ. Thử ở 100 và 300 N·m, 1500 rpm.
- Gợn của FCS-MPCC giảm khi L tăng. Tuy vậy, ở cùng tần số chuyển mạch, PI-SVPWM vẫn ngang hoặc tốt hơn.
- Ở 100 N·m, AI-FCS xấp xỉ cặp trọng số cố định. Ở 300 N·m, AI-FCS có gợn thấp hơn 5–16 % (Bảng 15).
- Kết luận ở mục 3 không thay đổi trong dải này.

### 3.4 So sánh công bằng với PWM tần số chuyển mạch thay đổi, cùng ràng buộc gợn mô-men (Bảng 16, Hình 9, 10)
**Lý do.** Ở mục 3.1, baseline PWM có tần số cố định (10 kHz, 3,6 kHz) nên so sánh với FCS-MPCC có λ_sw là chưa công bằng. Ở đây mỗi phương pháp có một "núm chỉnh" riêng và núm này được lập lịch **theo từng điểm vận hành** với cùng một ràng buộc gợn:

| Phương pháp | Bộ điều khiển dòng | Núm chỉnh được lập lịch |
|---|---|---|
| PI-SVPWM VSF | PI + SVPWM (mã 0) | tần số sóng mang ∈ {1; 1,5; 2; 2,5; 3,64; 5; 7,5; 10; 15; 20} kHz |
| PI-DPWM1 VSF | PI + DPWM1 (mã 6, mới) | như trên |
| PI-DPWM-I VSF | PI + DPWM đặt kẹp theo dòng pha (mã 7, mới) | như trên |
| FCS-MPCC (Ts = 10 µs) | FCS (mã 2) | λ_sw ∈ {0; 10; 30; 100; 300; 1000; 3000; 10000} |
| FCS-MPCC (Ts, λ_sw) | FCS (mã 2) | cặp (Ts ∈ {10; 20; 50} µs, λ_sw) |

- **DPWM1** (mã 6): kẹp pha có |v\*| lớn nhất vào thanh áp cùng dấu (kẹp 60° quanh đỉnh điện áp pha, vị trí theo **điện áp**, chuẩn DPWM1). Mỗi pha được kẹp 2×60°/360°, giảm số lần chuyển mạch còn khoảng 2/3 (đo được 0,675 so với SVPWM cùng sóng mang).
- **DPWM-I** (mã 7): so sánh hai phương án kẹp khả thi (pha có áp lớn nhất lên thanh +, pha có áp nhỏ nhất xuống thanh −) và chọn phương án có |i| pha lớn hơn (dòng đặt dq quay về góc PWM). Vì E_sw ∝ |i|, đây là vị trí kẹp tại đỉnh dòng pha. Khi dòng lệch pha với áp (MTPA, suy từ thông) thì khác DPWM1 theo điện áp.
- **VSF** không phải mã điều khiển riêng: chu kỳ sóng mang bằng chu kỳ điều khiển Ts và cố định trong một lần chạy. Tần số sóng mang được chọn **ngoại tuyến** thành bảng tra theo (tốc độ, mô-men), giống cách chọn λ_sw. Mỗi điểm bản đồ chạy ở sóng mang riêng. Mọi lần chạy động học đều dùng một sóng mang cố định theo điểm.
- **Luật băng thông PI** giữ nguyên cho mọi sóng mang: ω_c = 2π·min(800 Hz, f_c/12). Từ 9,6 kHz trở lên băng thông là 800 Hz. Ở 1; 2,5; 5 kHz băng thông là 83; 208; 417 Hz. Chi phí động học (Bảng 16i, đáp ứng nấc iq 200→600 A): thời gian xác lập 5 % là 14,0 ms (1 kHz), 3,3 ms (2,5 kHz), 2,2 ms (5 kHz), 1,0 ms (10 kHz), 0,33 ms (15 kHz), 0,50 ms (20 kHz); vọt lố 22–30 % ở 1–7,5 kHz. Bản đồ ở chế độ xác lập không phản ánh chi phí này.

**Ràng buộc và cách chọn.** Tại mỗi điểm lưới (9 tốc độ × 9 mô-men, 76 điểm khả thi), chọn cài đặt có tổn hao truyền động P_cu + P_cond + P_sw nhỏ nhất thoả T_rms ≤ giới hạn (trung bình 3 seed). Ba giới hạn chính: **abs3** = 3 % × 430 = 12,9 N·m; **abs1** = 1 % × 430 = 4,3 N·m; **rel5** = 5 % mô-men yêu cầu, sàn 3 N·m. Sàn 3 N·m được chọn vì ở sàn 2 N·m (cột `rel5f2` trong CSV) DPWM không có cài đặt khả thi tại 9 điểm, FCS tại 3 điểm. Cài đặt không đạt mô-men trung bình (sai lệch > max(10 %, 5 N·m), thường là mất ổn định ở sóng mang thấp và tốc độ cao, hoặc λ_sw quá lớn) bị loại bất kể gợn (Bảng 16h). Mỗi tổ hợp được chạy 3 seed; mô phỏng dài max(30 ms, 1 chu kỳ điện + 10 ms) để phủ đủ chuỗi kẹp DPWM ở tốc độ thấp. Tổng: 12312 lần chạy bản đồ (54 cài đặt), 0 lần không hữu hạn.

**Đánh giá WLTC.** (i) *Lập lịch lý tưởng*: tại mỗi giây của chu trình chọn cài đặt tổn hao nhỏ nhất thoả giới hạn trên bản đồ nội suy, tức không chịu sai số do lưới thô và đúng ràng buộc 100 % thời gian. (ii) *Bảng tra 9×9 (nút gần nhất)*: phương án triển khai được; cột "vượt" là tỉ lệ thời gian gợn lớn hơn giới hạn do lưới thô. Các số trong bảng cùng bản đồ mô phỏng, nên **không** trùng đúng Bảng 14 (khác thời lượng mô phỏng ở tốc độ thấp và số seed); các điểm tham chiếu vẫn khớp: PI 10 kHz 4,08 và FCS λ_sw = 1000 là 2,82 Wh/km (Bảng 16e).

Tổn hao truyền động WLTC (Wh/km), gợn mô-men trung bình theo thời gian (N·m). "So với PI" là so với PI-SVPWM VSF cùng giới hạn (Bảng 16):

| Giới hạn | Phương pháp | Lập lịch lý tưởng: tổn hao / gợn TB / so với PI-VSF | Bảng tra 9×9: tổn hao / % thời gian vượt | Một cài đặt cố định tốt nhất: tổn hao (cài đặt) |
|---|---|---|---|---|
| abs3 (12,9 N·m) | PI-SVPWM VSF | 3,18 / 4,37 / — | 3,20 / 0,0 % | 3,44 (5 kHz) |
| | PI-DPWM1 VSF | 3,00 / 6,45 / −5,7 % | 3,01 / 0,7 % | 3,18 (5 kHz) |
| | PI-DPWM-I VSF | 2,97 / 6,38 / −6,4 % | 2,99 / 1,0 % | 3,16 (5 kHz) |
| | FCS, λ_sw lập lịch (Ts = 10 µs) | **2,81** / 6,28 / −11,4 % | 2,83 / 0,0 % | 2,82 (λ_sw = 1000) |
| | FCS, lập lịch (Ts, λ_sw) | **2,67** / 7,34 / −16,0 % | 2,63 / 0,0 % | 2,82 (Ts = 10 µs, λ_sw = 1000) |
| abs1 (4,3 N·m) | PI-SVPWM VSF | 3,38 / 3,11 / — | 3,42 / 7,0 % | 4,08 (10 kHz) |
| | PI-DPWM1 VSF | 3,35 / 3,45 / −1,1 % | 3,37 / 7,8 % | 3,73 (15 kHz) |
| | PI-DPWM-I VSF | **3,31** / 3,41 / −2,0 % | 3,34 / 7,8 % | 3,67 (15 kHz) |
| | FCS, λ_sw lập lịch (Ts = 10 µs) | 3,59 / 3,07 / **+6,1 %** | 3,59 / 0,0 % | 3,59 (λ_sw = 300) |
| | FCS, lập lịch (Ts, λ_sw) | 3,33 / 3,57 / −1,5 % | 3,33 / 0,0 % | 3,59 (Ts = 10 µs, λ_sw = 300) |
| rel5 (5 %, sàn 3 N·m) | PI-SVPWM VSF | 3,57 / 2,53 / — | 3,60 / 17,3 % | 4,08 (10 kHz) |
| | PI-DPWM1 VSF | 3,63 / 2,53 / +1,7 % | 3,61 / 17,8 % | 4,21 (20 kHz) |
| | PI-DPWM-I VSF | **3,54** / 2,58 / −1,0 % | 3,52 / 18,0 % | 4,12 (20 kHz) |
| | FCS, λ_sw lập lịch (Ts = 10 µs) | 4,13 / 2,27 / **+15,5 %** | 4,20 / 10,5 % | 4,39 (λ_sw = 100) |
| | FCS, lập lịch (Ts, λ_sw) | 4,09 / 2,31 / +14,5 % | 4,17 / 10,5 % | 4,39 (Ts = 10 µs, λ_sw = 100) |

**Kết luận từ dữ liệu (Hình 9: Pareto tổn hao–gợn trên WLTC):**
1. **FCS-MPCC có λ_sw chỉ thắng khi giới hạn gợn lỏng.** Ở 3 % mô-men đỉnh, FCS λ_sw lập lịch giảm 11,4 % tổn hao so với PI-SVPWM VSF và 5,4 % so với DPWM-I VSF tốt nhất (2,81 so với 2,97). Một λ_sw cố định = 1000 đã đạt 2,82, nên lập lịch λ_sw cho thêm không đáng kể. Phần lợi này đi cùng gợn trung bình lớn hơn (khoảng 6,3 so với 4,4 N·m ở PI-VSF): FCS dùng hết ngân sách gợn.
2. **Khi giới hạn chặt, PWM tần số thay đổi trội hơn.** Ở 1 % (4,3 N·m), FCS (Ts = 10 µs) kém PI-VSF 6,1 % và kém DPWM-I VSF 8,3 % (3,59 so với 3,31). Ở giới hạn tương đối 5 % (sàn 3 N·m), FCS kém PI-VSF 15,5 %. Lập lịch cả Ts cho FCS thu hẹp khoảng cách nhưng chỉ ngang DPWM-I ở 1 % (3,33 so với 3,31) và vẫn kém 14,5 % ở giới hạn tương đối. Quét giới hạn tuyệt đối (Bảng 16f): FCS (Ts = 10 µs) thua PI-VSF và DPWM-I khi giới hạn ≤ 6 N·m (3,59 so với 3,22 và 3,12 ở 6 N·m) và thắng từ 8 N·m (2,81 so với 3,18 và 3,03); FCS (Ts, λ_sw) hoà DPWM-I ở 4,3 N·m và thắng từ 6 N·m (2,95 so với 3,12).
3. **DPWM chỉ mang lại lợi ích nhỏ.** DPWM giảm 2/3 số lần chuyển mạch nhưng gần gấp đôi gợn ở cùng sóng mang ở tải/tốc độ thấp (chỉ số điều chế nhỏ; khớp với tính toán volt-giây lý tưởng, 1,95 so với 1,96 ở 1500 rpm, 200 N·m). Kết quả ròng so với SVPWM VSF: −6 % ở 12,9 N·m, −1 đến −2 % ở 4,3 N·m, và −1 % đến +1,7 % ở giới hạn tương đối. DPWM-I (kẹp theo dòng) tốt hơn DPWM1 (kẹp theo áp) từ 0,8 đến 2,6 %.
4. **Baseline cố định 3,6 kHz / 10 kHz của mục 3.1 đúng là không công bằng**: PI tần số cố định tốt nhất thoả giới hạn ở mọi điểm tốn thêm +8 % (abs3, 5 kHz: 3,44 so với 3,18), +21 % (abs1, 10 kHz: 4,08 so với 3,38) và +14 % (rel5, 10 kHz: 4,08 so với 3,57) so với PI-VSF. Riêng PI 3,6 kHz (3,65 Wh/km) vi phạm giới hạn 12,9 N·m trong 4 % thời gian chu trình (Bảng 16e).
5. **Mở rộng tập sóng mang xuống 1 đến 2 kHz không đổi kết luận**: PI-SVPWM VSF ở 12,9 N·m chỉ giảm từ 3,20 (tập 2,5 đến 20 kHz) xuống 3,18 Wh/km (tập 1 đến 20 kHz), DPWM-I từ 2,98 xuống 2,97 (Bảng 16j). Các sóng mang ≤ 2,5 kHz không giữ được mô-men ở tốc độ cao (Bảng 16h: 11 đến 43 trong 76 điểm). Đây là hệ quả của trễ tính toán 1 chu kỳ và luật f_c/12, chưa dùng PWM đồng bộ hay bù trễ nâng cao.
6. **Kiểm chứng ngoài lưới** (Bảng 16g; 11 điểm vận hành lệch lưới, seed mới): bảng tra 9×9 chênh −0,2 đến +1,2 % tổn hao so với chọn tối ưu tại chính điểm đó, nhưng chỉ 82 đến 100 % số điểm thoả giới hạn (nút gần nhất ở lưới thô; cần lưới mịn hơn hoặc dự trữ biên).

**Độ nhạy theo mô hình tổn hao** (Bảng 16b, 16c; chọn lại lịch cho từng trường hợp; Wh/km, lập lịch lý tưởng):

| Giới hạn | Phương pháp | Danh định | Esw ×0,5 | Esw ×2 | r_on ×0,5 | r_on ×2 |
|---|---|---|---|---|---|---|
| abs3 | PI-SVPWM VSF | 3,18 | 2,74 | 3,94 | 2,90 | 3,74 |
| | PI-DPWM-I VSF | 2,97 | 2,64 | 3,58 | 2,69 | 3,54 |
| | FCS (Ts = 10 µs) | 2,82 | 2,47 | 3,51 | 2,56 | 3,32 |
| | FCS (Ts, λ_sw) | 2,67 | 2,38 | 3,21 | 2,42 | 3,17 |
| abs1 | PI-SVPWM VSF | 3,38 | 2,82 | 4,46 | 3,11 | 3,92 |
| | PI-DPWM-I VSF | 3,32 | 2,78 | 4,37 | 3,04 | 3,86 |
| | FCS (Ts = 10 µs) | 3,59 | 2,86 | 5,05 | 3,33 | 4,10 |
| | FCS (Ts, λ_sw) | 3,33 | 2,74 | 4,51 | 3,07 | 3,85 |
| rel5 | PI-SVPWM VSF | 3,57 | 2,90 | 4,91 | 3,30 | 4,11 |
| | PI-DPWM-I VSF | 3,54 | 2,87 | 4,86 | 3,27 | 4,07 |
| | FCS (Ts = 10 µs) | 4,13 | 3,14 | 6,11 | 3,87 | 4,64 |
| | FCS (Ts, λ_sw) | 4,09 | 3,12 | 6,03 | 3,83 | 4,61 |

Với FCS λ_sw ở Ts = 10 µs, thứ hạng giữ nguyên trong cả 5 trường hợp tổn hao: thấp hơn mọi PWM ở 12,9 N·m, cao hơn PI-VSF ở 4,3 N·m và ở giới hạn tương đối. FCS (Ts, λ_sw) ở 4,3 N·m không phân định được với PWM: so với PI-VSF là 3,33 so với 3,38; 2,74 so với 2,82; 4,51 so với 4,46; 3,07 so với 3,11; 3,85 so với 3,92 (đổi dấu ở Esw ×2), và so với DPWM-I VSF là 3,33 so với 3,32; 2,74 so với 2,78; 4,51 so với 4,37; 3,07 so với 3,04; 3,85 so với 3,86. Khi tổn hao chuyển mạch tăng (Esw ×2), FCS (Ts, λ_sw) hơn PI-VSF 18,4 % ở 12,9 N·m (3,21 so với 3,94, so với 16,0 % danh định), còn FCS (Ts = 10 µs) kém PI-VSF 13,2 % ở 4,3 N·m (5,05 so với 4,46, so với 6,1 % danh định).

**Hạn chế riêng của mục này.**
- Chi phí tính toán không nằm trong so sánh: FCS chạy ở Ts = 10 µs (100 kHz) còn PWM chỉ cần sóng mang 1 đến 20 kHz. Lịch FCS (Ts, λ_sw) dùng Ts = 20 µs ở 52 trong 76 nút (abs1) và Ts = 20 hoặc 50 µs ở 59 trong 76 nút (abs3), tức cần tần số điều khiển thấp hơn FCS Ts = 10 µs (Bảng 16k).
- Mô-men được đo bằng độ lệch chuẩn của mô-men điện từ trong cửa sổ xác lập; ràng buộc là theo từng điểm, còn trục hoành Hình 9 là trung bình theo thời gian.
- Tổn hao sắt, tổn hao cơ, bão hoà và điều chế đồng bộ chưa được mô hình hoá (xem mục 4). Mọi kết luận ở mục này nằm trong mô hình tổn hao chuyển mạch E_sw ∝ |i| của dự án.
- Bảng "một cài đặt cố định tốt nhất" chọn trong số cài đặt thoả giới hạn ở mọi điểm lưới, rồi lấy cài đặt có tổn hao WLTC nhỏ nhất; có thể vi phạm giới hạn giữa các nút lưới (cột `viol_time_frac` trong CSV).
- Cơ chế loại cài đặt không giữ được mô-men (sai lệch > max(10 %, 5 N·m)) làm giảm tập khả thi của FCS với λ_sw ≥ 3000 và Ts lớn.

### 3.5 Tổn hao sắt stator có sóng hài PWM (Bảng 11b/11c, 14d–14h, Hình 7, 11, 12)
**Phần cơ bản của tổn hao sắt giống nhau với mọi bộ điều khiển; khác biệt chỉ đến từ phần sóng hài.** Số liệu lấy từ `Table14d_WLTC_with_iron_loss.csv` (nominal):

| Bộ điều khiển | Cu + nghịch lưu (Wh/km) | Thêm sắt (Wh/km) | Tổng truyền động (Wh/km) | Sắt / tổng | Sóng hài / sắt | Hạng (không sắt → có sắt) |
|---|---|---|---|---|---|---|
| PI-SVPWM 10 kHz | 4,08 | 9,04 | 13,12 | 69 % | 34 % | 5 → 5 |
| PI-SVPWM 3,6 kHz | 3,66 | 9,18 | 12,83 | 71 % | 37 % | 4 → 4 |
| FCS-MPCC λ_sw = 0 | 4,93 | 9,52 | 14,45 | 66 % | 37 % | 6 → 6 |
| FCS-MPCC λ_sw cố định = 1000 | 2,82 | 9,42 | **12,24** | 77 % | 36 % | 1 → 1 |
| FCS-MPCC + bảng tra λ_sw | 2,82 | 9,42 | 12,24 | 77 % | 36 % | 1 → 1 |
| FCS-MPCC + mạng nơ-ron λ_sw | 2,83 | 9,43 | 12,26 | 77 % | 36 % | 3 → 3 |

(Hạng được xếp trên cả 6 bộ điều khiển; bảng tra và λ_sw cố định đồng hạng.)

- **Thứ hạng WLTC không đổi** ở mô hình danh định. Lợi thế của FCS λ_sw = 1000 so với PI 3,6 kHz co lại từ 23 % (chỉ Cu + nghịch lưu) còn **4,7 %**; so với PI 10 kHz còn 6,7 %. Lý do: tổn hao sắt cơ bản (6,0 Wh/km) lớn và như nhau; FCS có sắt sóng hài cao hơn PI từ 0,04 (so với PI 3,6 kHz) đến 0,36 Wh/km (so với PI 10 kHz), nhỏ hơn nhiều so với mức tiết kiệm tổn hao chuyển mạch.
- **Độ nhạy** (`Table14e`, `Table14f`): trong 16 trường hợp (k_h, k_c, k_e ×0,5/×2; hệ số chế tạo ×0,5/×2; bỏ che chắn bề mặt; chỉ tính phần phổ phân giải được; phần sóng hài ×0, ×0,25, ×0,5, ×2, ×3; tổ hợp bất lợi), **FCS λ_sw = 1000 luôn là bộ điều khiển tốn ít năng lượng nhất**. Thứ hạng thay đổi chỉ ở chỗ PI 10 kHz vượt PI 3,6 kHz khi phần sóng hài ×2 và ×3. Trường hợp bất lợi (k_c ×2, k_e ×2, không che chắn, sóng hài ×2): FCS hơn PI 3,6 kHz chỉ 0,8 % (26,46 so với 26,67 Wh/km).
- **Hệ số hòa vốn** (`Table14g`): FCS λ_sw = 1000 chỉ thua PI 10 kHz khi phần sóng hài của sắt stator lớn hơn **3,5 lần** danh định, và thua PI 3,6 kHz khi lớn hơn **14,8 lần**. Hệ số này cũng đóng vai trò chặn trên cho tổn hao sóng hài chưa mô hình hóa ở rotor/nam châm (nếu chúng tỉ lệ với phần sóng hài stator).
- λ_sw cố định tối ưu vẫn là 1000 khi có tổn hao sắt (`Table14h`; không huấn luyện lại mạng/bảng tra).
- **Tổn hao sóng hài hầu như không phụ thuộc f_sw** (Bảng 11b): dòng xoáy và tổn hao dư thừa do bước điện áp PWM phụ thuộc vào Vdc chứ không vào f_sw (⟨|dψ/dt|²⟩ gần như hằng, ≈ 14 000 V² ở 3000 rpm/30 N·m). Chỉ thành phần từ trễ vòng nhỏ giảm theo ~1/f_sw, nhưng nó chỉ chiếm 1–5 W trong ~400 W. Với PI, tổn hao sóng hài giảm đơn điệu 198 → 181 → 137 W khi tần số sóng mang tăng 2,5 → 10 → 40 kHz (chủ yếu nhờ che chắn bề mặt). Với FCS λ_sw = 0, tổn hao sóng hài **không** giảm đơn điệu theo f_sw ở Ts nhỏ (tăng ở Ts ≤ 5–7,5 µs do nhiễu cảm biến 2 A gây đảo vectơ). Giảm f_sw bằng λ_sw (Ts = 10 µs) làm sắt sóng hài đổi rất ít (201 → 203 W ở 3000 rpm/30 N·m), nên cơ chế "f_sw thấp → sắt nhiều" mà phản biện nêu **không xuất hiện đáng kể ở stator trong mô hình này**.
- **f_sw tối ưu theo tổng tổn hao** (`Table11c`): ở 3000 rpm/30 N·m có cực trị trong khoảng quét: PI 3,6 kHz (cả khi có và không có sắt), FCS λ_sw = 0 chuyển từ 3,0 lên 5,7 kHz khi có sắt, FCS (λ_sw quét ở Ts = 10 µs) 3,7 kHz (λ_sw = 1000). Ở 1500 rpm/200 N·m tổng tổn hao đơn điệu theo f_sw nên cực trị nằm ở biên thấp nhất của khoảng quét (1,25–1,8 kHz); khi ràng buộc gợn mô-men ≤ 12,9 N·m thì: PI 1,7 kHz, FCS λ_sw = 0 1,8 kHz, FCS họ λ_sw 3,3 kHz. Tổn hao sắt dịch cực trị rất ít vì tổn hao chuyển mạch (cỡ 10 W/kHz ở tải nhẹ, 60 W/kHz ở 200 N·m) lấn át độ dốc của tổn hao sắt theo f_sw.
- **Hiệu suất** (Hình 7): hàng trên là Cu + nghịch lưu (như trước), hàng dưới thêm sắt stator. Sắt hạ hiệu suất rõ nhất ở mô-men nhỏ, tốc độ cao.

**Hạn chế riêng của mô hình sắt (đọc kỹ):**
1. Toàn bộ gợn từ thông móc vòng ψ = L·i (gồm cả từ thông tản) được coi là đi qua sắt stator, nên phần sóng hài danh định là **chặn trên**. Thực tế một phần là từ thông tản rãnh/đầu răng. Cận dưới được thể hiện bằng các trường hợp "sóng hài ×0,25/×0,5". Phần sóng hài đang chiếm ~35 % tổn hao sắt, lớn hơn mức thường đo được, nên cần xem như chặn trên.
2. Không có kiểm chứng FEA hay đo. Hình học răng/gông, k_h, k_e, hệ số chế tạo, μr hiệu dụng là giả thiết [ASM].
3. B tuyến tính theo ψ nên ở tải lớn (≥ 300 N·m) B răng đạt tới 2,75 T, vượt bão hòa: tổn hao sắt ở các điểm đó bị đánh giá cao. Vùng WLTC chủ yếu tải nhẹ nên ít ảnh hưởng.
4. **Chưa có** tổn hao sắt rotor, tổn hao dòng xoáy nam châm do sóng hài, tổn hao rải/đầu cuộn, ảnh hưởng nhiệt độ. Tổn hao nam châm do gợn dòng có thể tăng khi f_sw giảm, làm lệch có lợi cho phương pháp f_sw thấp; hệ số hòa vốn ở trên cho biết dư địa.
5. Che chắn hiệu ứng bề mặt dựa trên FFT của log 400 kHz (cột `ws_fft_resolved` = phần ⟨|dψ/dt|²⟩ sóng hài mà FFT nhìn thấy). Ở sóng mang ≥ 40 kHz chỉ phân giải 70–90 %. Trong bản đồ tổn hao, FCS phân giải 0,87–1,18 và PI 3,6 kHz 0,79–1,03; riêng **PI 10 kHz ở 200–500 rpm chỉ 0,31–0,78** (xung PWM rất hẹp ở chỉ số điều chế thấp, phổ vượt 200 kHz). Ở các điểm đó phần che chắn bị đánh giá thấp, tức tổn hao sóng hài của PI 10 kHz bị đánh giá cao (có lợi cho FCS). Trường hợp "chỉ tính phần phổ phân giải được" (cận dưới) hạ tổn hao PI 10 kHz xuống 12,95 Wh/km; thứ hạng không đổi, FCS λ_sw = 1000 vẫn hơn PI 10 kHz 5,6 % và PI 3,6 kHz 4,3 % (`Table14e`, `Table14f`).
6. Kỳ vọng "ở không tải với PI sóng mang rất cao thì tỉ lệ sóng hài nhỏ" **chỉ đúng với thành phần từ trễ vòng nhỏ** (0,26 % tại 40 kHz). Với dòng xoáy/dư thừa thì không: ở không tải, 1000 rpm, 40 kHz, sóng hài vẫn chiếm 52 % tổn hao sắt (điện áp bước PWM không giảm theo f_sw).

### 3.6 Kết luận tích hợp: ràng buộc gợn chung + tổn hao sắt (Bảng 16c, cột "+ iron")
Đây là phép so sánh cuối cùng: mỗi phương pháp được lập lịch tối ưu theo điểm vận hành dưới cùng giới hạn gợn, và tổn hao gồm đồng, nghịch lưu và sắt (có sóng hài PWM). Đơn vị là Wh/km trên WLTC.

| Giới hạn gợn | PI-SVPWM VSF | PI-DPWM-I VSF | FCS λ_sw (Ts = 10 µs) | FCS (Ts, λ_sw) |
|---|---|---|---|---|
| 12,9 N·m (3 %) | 12,32 | 12,18 | 12,18 | **12,01** |
| 4,3 N·m (1 %) | 12,49 | **12,43** | 12,72 | 12,53 |
| 5 % mô-men yêu cầu (sàn 3 N·m) | 12,64 | **12,60** | 13,21 | 13,18 |

Khi đã có baseline PWM công bằng và có tổn hao sắt, lợi thế năng lượng của FCS-MPCC gần như biến mất. Ở giới hạn gợn lỏng, FCS chỉ hơn khoảng 1,4–2,6 %, và cần lấy mẫu 50–100 kHz. Ở giới hạn chặt, PWM/DPWM có tần số thay đổi tốt hơn 0,8–4,9 %. Kết quả này đứng vững khi nhân Esw với 0,5 và 2.

## 4. Phạm vi mô hình
- Động cơ dùng tham số dq hằng. Ảnh hưởng của bão hòa được đánh giá qua độ nhạy theo L (mục 3.3), chưa dùng bản đồ từ thông FEA.
- Tổn hao sắt stator có sóng hài PWM đã được mô hình hóa (mục 2.1, 3.4), nhưng **chưa có** tổn hao sắt rotor, nam châm, tổn hao cơ và hiệu suất hộp số, không có FEA. Các bảng 5–15 (không có chữ "iron") vẫn là Cu + nghịch lưu, giữ nguyên số liệu cũ. Trên WLTC, tổn hao khi phát và khi hãm tái sinh được coi là đối xứng.
- Thời gian tính toán được thể hiện bằng ước lượng số phép toán (Bảng 1), chưa đo trên DSP/FPGA.

## 5. Cấu trúc
```
configs/default.json          thông số + thẻ nguồn
src/aifcs/core.py             vòng kín (numba): plant, nghịch lưu, 8 bộ điều khiển (mã 6, 7: DPWM1 và DPWM-I)
src/aifcs/vsf.py              bảng tra sóng mang VSF, chọn cài đặt theo ràng buộc gợn, đánh giá WLTC (lý tưởng và bảng tra)
scripts/vsf_baselines.py      mục 6c: bản đồ, Bảng 16–16i, Hình 9–10 (reproduce.py gọi tệp này)
src/aifcs/params.py           cấu hình -> vector tham số, tải đường, MTPA
src/aifcs/scenarios.py        S1–S4
src/aifcs/metrics.py          định nghĩa chỉ tiêu (docstring)
src/aifcs/training.py         huấn luyện mạng theo bản thảo + bộ lập lịch λ_sw
src/aifcs/wltc.py             đánh giá năng lượng WLTC tựa-tĩnh
src/aifcs/ironloss.py         tổn hao sắt stator có sóng hài PWM (hàm iron_loss, combine, waveform_stats)
data/wltc_class3b.csv         vận tốc WLTC Class 3b, 1 Hz
src/aifcs/plotting.py         hình
scripts/reproduce.py          chạy toàn bộ
scripts/derive_tesla_params.py
tests/test_sanity.py
```

## Nguồn
- MotorXP, *Performance Analysis of the Tesla Model 3 Electric Motor using MotorXP-PM*: https://motorxp.com/wp-content/uploads/mxp_analysis_TeslaModel3.pdf
- Gói Python `wltp` 0.1.2a0 (JRC), `wltp/cycles/class3.py`: dữ liệu WLTC Class 3b
- evspecifications.com, Tesla Model 3 RWD (khối lượng, Cd, tỉ số truyền, điện áp pin): https://www.evspecifications.com/en/comparison/30a08841 ; https://www.evspecifications.com/en/model/bc5896
