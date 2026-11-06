# KẾ HOẠCH TỔNG THỂ KHÓA LUẬN TỐT NGHIỆP (KLTN)
## Đề tài: Nghiên cứu và Đánh giá Energy-Based Concept Bottleneck Model (ECBM) cho Tiên lượng Lâm sàng từ Ảnh X-quang Lồng ngực có Khả năng Giải thích

---

- **Nhóm thực hiện:** Huy – Khánh
- **Thời gian thực hiện:** 24 tuần (~6 tháng)
- **Học kỳ:** HK1 (2026 – 2027)
- **Đơn vị:** Khoa Khoa học & Kỹ thuật Máy tính – Đại học Bách Khoa TP.HCM (HCMUT)
- **Dữ liệu thực nghiệm kết hợp:**
  1. **MIMIC-CXR (v2.1.0):** 5.000 ca chụp X-quang lồng ngực chọn lọc sạch tư thế thẳng (AP/PA), lưu trữ tại `data/mimic_cxr/`.
  2. **MIMIC-IV (v3.1):** Hồ sơ lâm sàng bệnh viện Beth Israel Deaconess Medical Center, trích xuất nhãn kết cục từ `hosp/admissions.csv.gz`.
- **Không gian Khái niệm ($C$):** **Đủ 14 Khái niệm CheXpert Chuẩn** trích xuất từ ảnh X-quang lồng ngực.
- **Biến Chẩn đoán Mục tiêu ($Y$):** **Nguy cơ Tử vong trong Viện (`hospital_expire_flag`: 0/1)** liên kết từ MIMIC-IV.
- **Phiên bản kế hoạch:** v4.0 (Đồng bộ 100% với kiến trúc 14 Concepts CheXpert & Tiên lượng Tử vong MIMIC-IV của Thầy)

---

## MỤC LỤC
1. [Tổng quan Đề tài & Bối cảnh Khoa học](#1-tổng-quan-đề-tài--bối-cảnh-khoa-học)
2. [Câu hỏi Nghiên cứu (Research Questions - RQ)](#2-câu-hỏi-nghiên-cứu-research-questions---rq)
3. [Dữ liệu Thực nghiệm Kết hợp (MIMIC-CXR & MIMIC-IV)](#3-dữ-liệu-thực-nghiệm-kết-hợp-mimic-cxr--mimic-iv)
4. [Hệ thống Mô hình Thực nghiệm Đối chứng & Đề xuất](#4-hệ-thống-mô-hình-thực-nghiệm-đối-chứng--đề-xuất)
5. [Quy chuẩn Kỹ thuật & Thiết kế Thực nghiệm (Evaluation Protocol)](#5-quy-chuẩn-kỹ-thuật--thiết-kế-thực-nghiệm-evaluation-protocol)
6. [Kế hoạch Chi tiết 24 Tuần (Timeline & Milestones)](#6-kế-hoạch-chi-tiết-24-tuần-timeline--milestones)
7. [Phân công Nhiệm vụ Nhóm (Huy – Khánh)](#7-phân-công-nhiệm-vụ-nhóm-huy--khánh)
8. [Cấu trúc Chi tiết Báo cáo Khóa luận 6 Chương](#8-cấu-trúc-chi-tiết-báo-cáo-khóa-luận-6-chương)
9. [Quản trị Rủi ro & Phương án Dự phòng](#9-quản-trị-rủi-ro--phương-án-dự-phòng)
10. [Bảng Theo dõi Tiến độ Hiện tại (Project Status Checklist)](#10-bảng-theo-dõi-tiến-độ-hiện-tại-project-status-checklist)

---

## 1. TỔNG QUAN ĐỀ TÀI & BỐI CẢNH KHOA HỌC

### 1.1. Lý do chọn đề tài
- **Thách thức của Black-box AI trong Tiên lượng Lâm sàng:** Trong môi trường cấp cứu và hồi sức tích cực (ICU), việc dự đoán nguy cơ tử vong của bệnh nhân từ phim chụp X-quang lồng ngực là bài toán tối quan trọng giúp bác sĩ phân loại mức độ ưu tiên điều trị. Tuy nhiên, các mô hình học sâu truyền thống ($X \rightarrow Y$) đưa ra cảnh báo tử vong như một "hộp đen": bác sĩ không thể biết mô hình dựa vào dấu hiệu bệnh lý nào (bệnh nhân nguy kịch do Viêm phổi nặng, Phù phổi cấp tính hay Tràn khí màng phổi?), dẫn đến thiếu sự tin cậy trong các quyết định y khoa sống còn.
- **Concept Bottleneck Model (CBM - Koh et al., ICML 2020):** Đưa một tầng khái niệm lâm sàng trung gian ($C$) vào giữa ảnh ($X$) và kết cục lâm sàng ($Y$):
  $$\text{Ảnh X-quang } (X) \longrightarrow \text{14 Triệu chứng CheXpert } (C) \longrightarrow \text{Nguy cơ Tử vong } (Y)$$
  Bác sĩ có thể đọc hiểu logic: *"Mô hình dự báo nguy cơ tử vong cao vì phát hiện đồng thời Phù phổi và Đông đặc diện rộng"*.
- **Hạn chế của CBM truyền thống:**
  1. *Lỗi truyền thừa (Cascading error):* Nhận diện sai triệu chứng hình ảnh sẽ trực tiếp làm sai lệch mức độ cảnh báo nguy cơ.
  2. *Bỏ qua tương quan bệnh học đa tổn thương:* 14 triệu chứng X-quang có mối liên hệ sinh bệnh học chặt chẽ (ví dụ: Suy tim thường đi liền với Bóng tim to và Phù phổi). CBM chuẩn giả định các khái niệm độc lập có điều kiện, bỏ lỡ tương quan này.
  3. *Độ chính xác suy giảm:* CBM thường bị sụt giảm hiệu năng dự báo so với Black-box CNN.
- **Giải pháp - Energy-Based Concept Bottleneck Model (ECBM - Xu et al., ICLR 2024):**
  - Mô hình hóa phân bố đồng thời: $p(X, C, Y) \propto \exp(-E(X, C, Y))$.
  - Thống nhất hàm năng lượng:
    $$E(X, C, Y) = E_{img}(X, C) + E_{concept}(C, Y) + E_{global}(X, C, Y)$$
  - **Khả năng Can thiệp Khái niệm (Concept Intervention) mang giá trị lâm sàng cao:** Khi bác sĩ chẩn đoán hình ảnh đính chính hoặc xác nhận một số triệu chứng trên phim ($C_{obs}$), hệ thống tự động hiệu chỉnh xác suất của các triệu chứng còn lại ($C_{unobs}$) và tính toán lại nguy cơ tử vong chính xác: $p(Y | X, C_{obs})$.

### 1.2. Mục tiêu nghiên cứu tổng quát
Xây dựng, tối ưu hóa và đánh giá thực nghiệm toàn diện mô hình ECBM đối chứng với Black-box và các biến thể CBM trên bài toán: **Tiên lượng nguy cơ tử vong trong viện (`hospital_expire_flag`) từ ảnh X-quang lồng ngực (MIMIC-CXR) thông qua tầng giải thích 14 khái niệm bệnh học chuẩn CheXpert**, đồng thời liên kết xác thực với hồ sơ bệnh án điện tử **MIMIC-IV**.

---

## 2. CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS - RQ)

| Mã RQ | Câu hỏi Nghiên cứu | Giả thuyết Khoa học | Bằng chứng Thực nghiệm & Chỉ số Đánh giá |
|:---|:---|:---|:---|
| **RQ1** | **Hiệu năng Tiên lượng Tử vong (Mortality Prediction Performance):** ECBM đạt hiệu năng dự báo tử vong như thế nào so với Black-box Classifier và các baseline CBM chuẩn (Soft CBM, Hard CBM)? | ECBM bảo toàn được hiệu năng dự báo tử vong tiệm cận Black-box nhờ năng lượng toàn cục $E_{global}$, đồng thời vượt trội hơn Soft CBM và Hard CBM nhờ mô hình hóa năng lượng tương quan giữa 14 triệu chứng. | • Macro-F1 (metric chính)<br>• Balanced Accuracy<br>• ROC-AUC, PR-AUC<br>• Đánh giá qua 3 seeds (42, 123, 2026) |
| **RQ2** | **Chất lượng Phát hiện 14 Khái niệm CheXpert (Concept Quality):** ECBM dự đoán từng triệu chứng và toàn bộ hồ sơ 14 triệu chứng chính xác đến mức nào? | Ràng buộc năng lượng tương thích giữa hình ảnh và bệnh lý giúp nâng cao khả năng phát hiện các tổn thương X-quang khó nhận diện, đặc biệt ở các ca bệnh nặng. | • Per-concept Macro-F1 (trên cả 14 concepts)<br>• Mean Concept Accuracy<br>• Exact-Match Ratio ($C_{pred} \equiv C_{gt}$ toàn bộ 14 concepts) |
| **RQ3** | **Hiệu quả Can thiệp Triệu chứng (Concept Intervention):** Khi bác sĩ can thiệp hiệu chỉnh $m$ triệu chứng ($m = 0 \dots 14$), mức độ cảnh báo tử vong và các triệu chứng chưa can thiệp thay đổi ra sao? | Can thiệp bằng ground-truth triệu chứng giúp cải thiện đáng kể độ chính xác tiên lượng tử vong. Trong ECBM, việc sửa 1 tổn thương (như Phù phổi) sẽ tự động lan truyền để hiệu chỉnh các tổn thương liên đới còn lại. | • Đường cong can thiệp: $m \rightarrow \text{Macro-F1}$<br>• Area Under Intervention Curve (AUC-IC)<br>• $\Delta \text{Macro-F1} = F1(m=14) - F1(m=0)$<br>• Độ chính xác của các unobserved concepts |
| **RQ4** | **Phân tích Bất nhất Lâm sàng & Lỗi Suy luận (Inconsistency & Error Analysis):** Các ca tiên lượng sai liên hệ thế nào với hiện tượng bất nhất giữa tổn thương X-quang và kết cục bệnh nhân? | Một số bệnh nhân có tổn thương phổi nặng nhưng được cứu sống, hoặc tổn thương nhẹ nhưng tử vong do suy đa tạng ngoài phổi (inconsistency); ECBM xử lý tốt hơn nhờ phân bố xác suất mềm. | • Ma trận phân loại 4 góc lỗi (Triệu chứng Đúng/Sai × Tử vong Đúng/Sai)<br>• Thống kê các ca mâu thuẫn giữa 14 concepts và kết cục tử vong<br>• Case-study định tính trực quan trên ca bệnh thực tế |

---

## 3. DỮ LIỆU THỰC NGHIỆM KẾT HỢP (MIMIC-CXR & MIMIC-IV)

### 3.1. Thiết kế Ghép nối Dữ liệu (Data Linkage)
Đề tài khai thác sức mạnh kết hợp giữa hình ảnh học (MIMIC-CXR) và bệnh án điện tử (MIMIC-IV) thông qua mã định danh bệnh nhân duy nhất `subject_id`:
- **Nguồn ảnh & Triệu chứng:** `data/mimic_cxr/` (5.000 ca chọn lọc sạch tư thế thẳng AP/PA, mỗi đợt chụp duy nhất 1 ảnh).
- **Nguồn nhãn kết cục:** `Desktop/Project/mimic-iv-3.1/hosp/admissions.csv.gz` (lấy cột `hospital_expire_flag`).
- **Phân chia dữ liệu (Data Split - Seed 42):** 
  - Đảm bảo **Patient-disjoint tuyệt đối** (tất cả ảnh và thông tin của 1 bệnh nhân chỉ thuộc duy nhất 1 tập: Train, Val hoặc Test).
  - Quy mô: Train ~4.000 ca, Validation ~500 ca, Test ~500 ca.

### 3.2. Không gian 14 Khái niệm CheXpert Chuẩn ($C$)
Toàn bộ 14 cột nhãn trong `selected_chexpert_labels.csv` được sử dụng làm không gian khái niệm giải thích:
1. **No Finding:** Không phát hiện tổn thương bất thường
2. **Enlarged Cardiomediastinum:** Trung thất giãn rộng
3. **Cardiomegaly:** Bóng tim to
4. **Lung Opacity:** Mờ phế trường
5. **Lung Lesion:** Tổn thương khu trú trong phổi (khối u/nốt)
6. **Edema:** Phù phổi mô kẽ / phế nang
7. **Consolidation:** Đông đặc phổi
8. **Pneumonia:** Tổn thương dạng viêm phổi
9. **Atelectasis:** Xẹp phổi
10. **Pneumothorax:** Tràn khí màng phổi
11. **Pleural Effusion:** Tràn dịch màng phổi
12. **Pleural Other:** Tổn thương màng phổi khác
13. **Fracture:** Gãy xương (gãy xương sườn)
14. **Support Devices:** Thiết bị y tế hỗ trợ (ống nội khí quản, catheter, máy tạo nhịp)

*Không gian trạng thái:* $2^{14} = 16.384$ profiles.
$\rightarrow$ Trên GPU, việc tính toán ma trận phân bố xác suất cho 16.384 trạng thái chỉ tiêu tốn **~2 MB bộ nhớ**, cho phép thực hiện **Exact Enumeration Inference** cực nhanh và chính xác tuyệt đối mà không cần dùng thuật toán xấp xỉ MCMC.

### 3.3. Nhãn Chẩn đoán Đích ($Y$)
- **Biến mục tiêu:** `hospital_expire_flag`
  - `0`: Bệnh nhân xuất viện an toàn (Survived / Discharged)
  - `1`: Bệnh nhân tử vong trong thời gian nằm viện (In-hospital Deceased)

### 3.4. Phương pháp luận Data-Centric AI trên Không gian 14 Concepts
1. **Chính sách Nhãn Bất định (CheXpert Uncertainty Policy):**
   - *U-Zeros (Mặc định):* Gán các vị trí nhãn nghi ngờ (`-1.0`) và không đề cập (`missing`) thành $0$ (thận trọng, tránh dương tính giả).
   - *U-Ignore (Masked Loss):* Bỏ qua các vị trí concept không chắc chắn khi tính gradient, giúp mô hình không bị học nhiễu.
2. **Đo lường Trần Dữ liệu & Concept Inconsistency:**
   - Phân tích xem trong 16.384 tổ hợp triệu chứng X-quang, có bao nhiêu % ca có cùng bộ 14 triệu chứng nhưng kết cục sống/chết khác nhau (do các yếu tố bệnh nền ngoài phổi).
   - Xác lập trần chính xác lý thuyết khi dự báo sống còn chỉ dựa vào phim chụp lồng ngực.
3. **Xử lý Mất cân bằng Triệu chứng & Kết cục:**
   - Tỷ lệ tử vong trong viện thường chiếm khoảng 10–15% tổng số ca nhập viện $\rightarrow$ áp dụng Pos-weight và Focal Loss cho cả nhánh concept và nhánh chẩn đoán $Y$.

---

## 4. HỆ THỐNG MÔ HÌNH THỰC NGHIỆM ĐỐI CHỨNG & ĐỀ XUẤT

```
[Ảnh X-quang Ngực X]
   │
   ├──────────────────────────────► [Black-box CNN (EfficientNet-B0)] ───────────► Tử vong Y (0/1)
   │
   ├─► [Trích xuất 14 Triệu chứng] ─► [Soft CBM (SoftJointCBM)] ─────────────────► Tử vong Y (0/1)
   │          │          (Xác suất mềm 28D)
   │          │
   │          ├──────────────────► [Hard CBM (HardJointCBM)] ────────────────────► Tử vong Y (0/1)
   │          │          (Rời rạc hóa one-hot 28D)
   │          │
   │          └──────────────────► [Energy-Based CBM (CategoricalECBM)] ─────────► Tử vong Y (0/1)
   │                     (Hàm năng lượng E(x,c,y) qua 16.384 states)
```

1. **Majority Baseline (Baseline Đa số):** Luôn dự đoán bệnh nhân sống xuất viện (nhãn chiếm đa số). Xác lập ngưỡng sàn tối thiểu.
2. **Black-box Classifier (Mô hình Hộp đen):** Backbone `EfficientNet-B0` pretrained ImageNet kết nối trực tiếp đến Classification Head MLP(128) $\rightarrow Y$. Đóng vai trò mốc so sánh trần về hiệu năng dự báo trực tiếp không giải thích.
3. **Soft CBM (SoftJointCBM):** Huấn luyện đồng thời nhánh dự đoán 14 concepts và đưa vector xác suất mềm 28D vào head chẩn đoán $\rightarrow Y$.
4. **Hard CBM (HardJointCBM):** Rời rạc hóa 14 triệu chứng thành vector one-hot categorical 28D (chuẩn dermaXAI).
5. **Energy-Based CBM (CategoricalECBM - Mô hình Đề xuất Chính):**
   - Cấu trúc hàm năng lượng tổng: $E(x, c, y) = E_{img}(x, c) + E_{concept}(c, y) + E_{global}(x, c, y)$.
   - Huấn luyện qua Joint Negative Log-Likelihood trên không gian $16.384 \times 2 = 32.768$ cấu hình:
     $$\mathcal{L} = -\log p(c, y | x) = E(x, c, y) + \log \sum_{c' \in \{0,1\}^{14}, y' \in \{0,1\}} \exp(-E(x, c', y'))$$
   - **Inference chính xác (Exact Inference):** Biên độ hóa trên 16.384 trạng thái: $p(y|x) = \sum_{c \in \{0,1\}^{14}} p(c, y|x)$.
   - **Can thiệp Khái niệm (Concept Intervention):** Kẹp (clamp) các triệu chứng do bác sĩ xác nhận vào trạng thái cố định ($c_{obs}$) và suy luận phân bố tử vong có điều kiện: $p(y | x, c_{obs})$.

---

## 5. QUY CHUẨN KỸ THUẬT & THIẾT KẾ THỰC NGHIỆM

### 5.1. Tham số Huấn luyện Chuẩn hóa (Standard Hyperparameters)
- **Kiến trúc Backbone:** `EfficientNet-B0` (Pretrained weights: `IMAGENET1K_V1`).
- **Concept Head:** `Linear(in_dim, 14)` kết hợp Sigmoid / BCE Loss.
- **Diagnosis Head:** `Sequential(Linear(14, 128), ReLU(), Dropout(0.2), Linear(128, 2))`.
- **Kích thước ảnh đầu vào:** $224 \times 224$ pixels (và nhánh khảo sát $512 \times 512$).
- **Tiền xử lý & Augmentation:** Random Horizontal Flip (p=0.5), Random Rotation ($\pm 10^\circ$), Chuẩn hóa ImageNet.
- **Kích thước Batch (Batch Size):** 16.
- **Số lượng Epochs:** **30 epochs** (đủ để toàn bộ các nhánh hội tụ tối ưu).
- **Bộ tối ưu hóa (Optimizer):** `AdamW` với Weight Decay = $0.01$.
- **Tốc độ học (Learning Rate):** Backbone: $1 \times 10^{-4}$; Heads: $1 \times 10^{-3}$.
- **Lịch biểu học (Scheduler):** `CosineAnnealingLR` ($T_{max} = 30$, $\eta_{min} = 1 \times 10^{-6}$).
- **Bộ Random Seeds Cố định (3 Seeds):** `42`, `123`, `2026` (báo cáo Mean $\pm$ Standard Deviation).

### 5.2. Cơ chế Checkpoint & Lựa chọn Ngưỡng Phán quyết
- **Tiêu chí Lưu Checkpoint Tốt nhất:** Đánh giá trên tập **Validation** tại cuối mỗi epoch. Tự động lưu mô hình có **Validation Macro-F1** cao nhất (với ngưỡng mặc định $0.5$).
- **Điều chỉnh Ngưỡng Quyết định (Decision Threshold Tuning):** Quét ngưỡng $T \in [0.05, 0.95]$ trên xác suất Validation để cực đại hóa **Balanced Accuracy**, đóng băng $T^*$ áp dụng cho Test set.

### 5.3. Quy trình Đánh giá Can thiệp Triệu chứng (Intervention Protocol)
- Số lượng triệu chứng can thiệp $m$ chạy từ $0$ (chẩn đoán tự động hoàn toàn) đến $14$ (bác sĩ đính chính toàn bộ triệu chứng).
- Chiến lược chọn can thiệp:
  1. *Random Intervention:* Trung bình qua 5 lần chọn ngẫu nhiên tập con cỡ $m$.
  2. *Uncertainty-based Intervention:* Ưu tiên can thiệp các triệu chứng mà mô hình phân vân nhất (entropy cao nhất).
- Đo lường:
  - Đường cong can thiệp: $m \rightarrow \text{Macro-F1}$ ($m = 0 \dots 14$).
  - Chỉ số diện tích dưới đường cong: $\text{AUC-IC} = \int_0^1 F_1(r) \, dr$ với $r = m/14$.
  - Mức tăng chẩn đoán: $\Delta \text{Macro-F1} = F_1(m=14) - F_1(m=0)$.
  - **Độ chính xác của các triệu chứng chưa can thiệp (Unobserved Concepts Recovery):** Kiểm tra xem khi bác sĩ sửa 1 vài tổn thương chính thì mô hình có tự động sửa đúng các tổn thương còn lại hay không.

---

## 6. KẾ HOẠCH CHI TIẾT 24 TUẦN (TIMELINE & MILESTONES)

```
Tuần:  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
Phase 0: [===Lý thuyết===]
Phase 1:     [===Dữ liệu===]
Phase 2:         [=Baselines=]
Phase 3:             [===CBM Baselines===]
Phase 4:                     [======ECBM Proposed======]
Phase 5:                                     [====Thực nghiệm RQ1-2====]
Phase 6:                                                 [===Can thiệp RQ3===]
Phase 7:                                                       [==Phân tích RQ4==]
Phase 8:                                                   [======Viết luận văn======]
Phase 9:                                                                         [==Bảo vệ==]
```

### Chi tiết Từng Giai đoạn:

#### Phase 0: Nghiên cứu Lý thuyết & Tổng quan Tài liệu (Tuần 1 – 4)
- Đọc và hệ thống hóa lý thuyết từ các công trình cốt lõi: Koh et al. (ICML 2020), Xu et al. (ICLR 2024), CheXpert (AAAI 2019), MIMIC-CXR & MIMIC-IV.
- Xây dựng ma trận so sánh các phương pháp (Related-work Matrix).
- **Đầu ra:** Bản tổng hợp lý thuyết; công thức toán học hàm năng lượng $E(x, c, y)$ kết nối 14 concepts với kết cục tử vong.

#### Phase 1: Chuẩn bị Dữ liệu & Ghép nối Bất biến (Tuần 3 – 5)
- Ghép nối dữ liệu 5.000 ca ảnh MIMIC-CXR với bảng `admissions.csv.gz` của MIMIC-IV để trích xuất nhãn `hospital_expire_flag`.
- Lập trình class `MimicCxrDataset` trong `src/dataset.py` và tạo file cấu hình `label_mapping.json` cho 14 concepts và 1 nhãn tử vong.
- Kiểm tra tính phân tách bệnh nhân (Patient-disjoint splits) và sinh `selection_manifest.csv` hoàn chỉnh.
- **Đầu ra:** Pipeline nạp dữ liệu PyTorch sẵn sàng; báo cáo EDA phân bố nhãn tử vong và 14 concepts.

#### Phase 2: Xây dựng Baselines Đối chứng (Tuần 5 – 6)
- Cài đặt Majority Baseline.
- Cài đặt và huấn luyện Black-box EfficientNet-B0 (ảnh $X \rightarrow Y$ trực tiếp) qua 3 seeds (42, 123, 2026).
- **Đầu ra:** Script `experiments/run_blackbox_mimic.py`; báo cáo kết quả baseline.

#### Phase 3: Triển khai CBM Chuẩn - Soft CBM & Hard CBM (Tuần 6 – 9)
- Triển khai Soft CBM (SoftJointCBM) với 14 concepts.
- Triển khai Hard CBM (HardJointCBM) với 14 concepts.
- Chuẩn hóa module tính toán metrics: Per-concept Macro-F1, Exact Match, Balanced Accuracy, Decision Threshold Tuning.
- Chạy thực nghiệm 3 seeds cho Soft CBM và Hard CBM; lưu checkpoint và log kết quả.
- **Đầu ra:** Script `experiments/run_cbm_mimic.py`; hoàn tất hệ thống đối chứng CBM.

#### Phase 4: Hiện thực hóa Mô hình Đề xuất - Energy-Based CBM (Tuần 9 – 14)
- Thiết kế module hàm năng lượng `ecbm.py` cho 14 concepts nhị phân:
  - `ImageEnergyHead`: $E_{img}(x, c)$ từ feature map của backbone.
  - `ClassConceptEnergy`: $E_{concept}(c, y)$ qua ma trận tương thích $2 \times 16.384$.
  - `GlobalEnergyHead`: $E_{global}(x, c, y)$.
- Triển khai thuật toán **Exact Enumeration Inference** cực nhanh qua tensor operations trên 16.384 trạng thái.
- Triển khai cơ chế Clamping phục vụ can thiệp $p(y | x, c_{clamped})$.
- Viết Unit Tests kiểm tra tính bảo toàn phân bố xác suất: $\sum_{c, y} p(c, y|x) = 1.0$.
- **Đầu ra:** Script `experiments/run_ecbm_mimic.py`; kiểm thử hội tụ thành công.

#### Phase 5: Thực nghiệm Chính & Đánh giá Toàn diện (RQ1 & RQ2) (Tuần 14 – 17)
- Huấn luyện chính thức Energy-Based CBM (ECBM) qua 3 seeds (42, 123, 2026) với 30 epochs/run.
- Tự động chọn checkpoint theo Validation Macro-F1 cao nhất.
- Đánh giá trên Test set:
  - Bảng so sánh tiên lượng tử vong (RQ1): Macro-F1, Balanced Acc, ROC-AUC, PR-AUC giữa Black-box, Soft CBM, Hard CBM và ECBM.
  - Bảng so sánh chất lượng dự đoán 14 triệu chứng X-quang (RQ2): Per-concept Macro-F1, Mean Accuracy, Exact Match.
- **Đầu ra:** Bảng số liệu khoa học đầy đủ (Mean $\pm$ SD) phục vụ Chương 5 của Luận văn.

#### Phase 6: Thực nghiệm Can thiệp Triệu chứng (Concept Intervention - RQ3) (Tuần 17 – 19)
- Thực hiện can thiệp tuần tự từ $m=0$ đến $m=14$ triệu chứng.
- Đo lường mức tăng hiệu năng chẩn đoán $\Delta \text{Macro-F1}$ và diện tích dưới đường cong $\text{AUC-IC}$.
- Đo lường mức độ phục hồi của các triệu chứng chưa can thiệp (Unobserved Concepts Recovery).
- Vẽ đồ thị so sánh đường cong can thiệp giữa Soft CBM, Hard CBM và ECBM.
- **Đầu ra:** Biểu đồ đường cong can thiệp chất lượng cao; Bảng kết quả định lượng RQ3.

#### Phase 7: Phân tích Lỗi, Nghịch lý & Inconsistency (RQ4) (Tuần 19 – 20)
- Trích xuất và phân loại các ca dự đoán sai theo ma trận 4 góc lỗi.
- Đối chiếu với hiện tượng bất nhất lâm sàng giữa tổn thương X-quang và nguy cơ tử vong.
- **Đầu ra:** Mục phân tích chuyên sâu (Discussion & Qualitative Analysis) trong Luận văn; các ca lâm sàng điển hình.

#### Phase 8: Viết Báo cáo Khóa luận Tốt nghiệp (Tuần 18 – 22)
- Hoàn thiện bản thảo luận văn 6 chương chuẩn quy cách Trường/Khoa.
- Gửi bản thảo cho Cán bộ hướng dẫn (CBHD) duyệt và chỉnh sửa.
- **Đầu ra:** Cuốn Luận văn Tốt nghiệp hoàn chỉnh.

#### Phase 9: Nghiệm thu, Tái lập, Slide & Bảo vệ (Tuần 23 – 24)
- Đóng gói mã nguồn Reproducible trên GitHub, lưu trữ checkpoints.
- Thiết kế slide thuyết trình, soạn kịch bản và luyện tập bảo vệ thử nghiệm.
- **Đầu ra:** Slide bảo vệ chính thức, Bảo vệ thành công khóa luận tốt nghiệp.

---

## 7. PHÂN CÔNG NHIỆM VỤ NHÓM (HUY – KHÁNH)

| Hạng mục | Thành viên Phụ trách Chính | Thành viên Phối hợp / Review chéo | Sản phẩm Cụ thể Bàn giao |
|:---|:---|:---|:---|
| **1. Pipeline Dữ liệu & Tiền xử lý** | **Huy** | Khánh | • Script ghép nối MIMIC-CXR với `admissions.csv.gz`<br>• `src/dataset.py` (`MimicCxrDataset` 14 concepts + nhãn tử vong)<br>• Báo cáo EDA, manifest bất biến, mapping JSON |
| **2. Cơ sở Lý thuyết & Toán học** | **Khánh** | Huy | • Công thức toán học hàm năng lượng $E(x,c,y)$ cho 14 concepts<br>• Phân tích cơ chế Clamped Inference & Exact Marginalization<br>• Soạn thảo Chương 2 của luận văn |
| **3. Baselines Đối chứng** | **Huy** | Khánh | • Huấn luyện Black-box Classifier<br>• Viết script chạy `experiments/run_blackbox_mimic.py`<br>• Báo cáo kết quả baseline |
| **4. Baselines CBM (Soft & Hard CBM)** | **Khánh** | Huy | • Module `src/cbm.py` & `concept_loss.py`<br>• Huấn luyện Hard CBM trên 14 triệu chứng<br>• Kết quả đối chứng Soft CBM & Hard CBM trên 3 seeds |
| **5. Phát triển Mô hình ECBM** | **Khánh** | Huy | • Module `src/ecbm.py` & tối ưu Exact Inference qua 16.384 states<br>• Cơ chế can thiệp Clamped Inference<br>• Script huấn luyện `run_ecbm_mimic.py` |
| **6. Quản trị Thực nghiệm & Logging** | **Huy** | Khánh | • Pipeline tự động lưu metrics, tensorboard, checkpoint<br>• Thuật toán Decision Threshold Tuning trên Validation<br>• Tổng hợp bảng kết quả RQ1, RQ2 (Mean $\pm$ SD) |
| **7. Đánh giá Can thiệp (RQ3)** | **Khánh** | Huy | • Script can thiệp tự động $m=0 \dots 14$<br>• Tính chỉ số AUC-IC và unobserved concept recovery<br>• Vẽ biểu đồ so sánh đường cong can thiệp |
| **8. Phân tích Lỗi & Inconsistency (RQ4)** | **Huy** | Khánh | • Script trích xuất ca lỗi 4 góc trên ảnh X-quang<br>• Thống kê hiện tượng Inconsistency giữa 14 concepts và tử vong<br>• Trực quan hóa hình ảnh các ca lâm sàng điển hình |
| **9. Soạn thảo Luận văn Tốt nghiệp** | **Cả hai** | Cả hai | • Huy: Chương 1, 3, 4, một phần Chương 5<br>• Khánh: Chương 2, Phương pháp Chương 3, một phần Chương 5, 6<br>• Cùng rà soát và định dạng bản in cuối cùng |
| **10. Chuẩn bị Bảo vệ & Repository** | **Cả hai** | Cả hai | • Hoàn thiện mã nguồn chuẩn Reproducible trên GitHub<br>• Soạn Slide thuyết trình & Luyện tập phản biện |

---

## 8. CẤU TRÚC CHI TIẾT BÁO CÁO KHÓA LUẬN 6 CHƯƠNG

### CHƯƠNG 1: GIỚI THIỆU (INTRODUCTION)
- 1.1. Bối cảnh nghiên cứu và Đặt vấn đề (Sự bùng nổ của AI trong tiên lượng lâm sàng tại ICU và đòi hỏi cấp thiết về Tính minh bạch/Giải thích được).
- 1.2. Thách thức nghiên cứu (Hạn chế của mô hình hộp đen và các giới hạn cố hữu của CBM truyền thống khi áp dụng vào ảnh X-quang lồng ngực).
- 1.3. Mục tiêu và Phạm vi nghiên cứu (Mục tiêu tiên lượng tử vong trong viện dựa trên 14 khái niệm CheXpert kết nối giữa MIMIC-CXR và MIMIC-IV).
- 1.4. Các câu hỏi nghiên cứu cốt lõi (RQ1 – RQ4).
- 1.5. Đóng góp khoa học của đề tài.
- 1.6. Bố cục của cuốn luận văn.

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN (LITERATURE REVIEW)
- 2.1. Phân loại ảnh X-quang Lồng ngực bằng Mạng nơ-ron Tích chập (Backbone CNNs, EfficientNet).
- 2.2. Khả năng giải thích trong AI Y tế (Explainable AI - XAI): Post-hoc vs Ante-hoc/By-design.
- 2.3. Concept Bottleneck Models (CBM): Định nghĩa toán học, phân loại (Joint, Sequential, Standard), ưu điểm và hạn chế.
- 2.4. Khái niệm Can thiệp Triệu chứng (Concept Intervention): Tương tác người - máy trong hỗ trợ ra quyết định lâm sàng.
- 2.5. Mô hình dựa trên Năng lượng (Energy-Based Models - EBMs): Nền tảng xác suất và hàm năng lượng.
- 2.6. Energy-Based Concept Bottleneck Model (ECBM): Nguyên lý tích hợp năng lượng toàn cục, cơ chế can thiệp mềm/cứng.
- 2.7. Bệnh án điện tử và Các hệ thống gán nhãn X-quang tự động (CheXpert Labeler, MIMIC-IV).
- 2.8. Ma trận so sánh và Khoảng trống nghiên cứu (Research Gap).

### CHƯƠNG 3: DỮ LIỆU VÀ PHƯƠNG PHÁP ĐỀ XUẤT (DATA & METHODOLOGY)
- 3.1. Phân tích Dữ liệu Nghiên cứu:
  - 3.1.1. Quy trình chọn lọc 5.000 ca chụp X-quang sạch từ MIMIC-CXR.
  - 3.1.2. Định nghĩa 14 khái niệm CheXpert chuẩn và quy trình trích xuất nhãn tử vong từ MIMIC-IV `admissions.csv.gz`.
- 3.2. Tiền xử lý Dữ liệu và Phương pháp Tiếp cận Data-Centric AI:
  - 3.2.1. Đảm bảo tính phân tách bệnh nhân tuyệt đối (Patient-disjoint splits) và Manifest bất biến.
  - 3.2.2. Chiến lược xử lý nhãn bất định CheXpert (Chính sách U-Zeros vs U-Ignore Masked Loss).
  - 3.2.3. Giải quyết mất cân bằng phân bố ở tầng triệu chứng và kết cục tử vong.
- 3.3. Phương pháp Đề xuất: Kiến trúc Energy-Based Concept Bottleneck Model (ECBM):
  - 3.3.1. Định nghĩa cấu trúc hàm năng lượng $E(x, c, y)$ trên không gian 16.384 profiles.
  - 3.3.2. Thuật toán Suy luận Chính xác (Exact Enumeration Inference Algorithm qua 16.384 trạng thái).
  - 3.3.3. Cơ chế Can thiệp Khái niệm Kẹp Năng lượng (Clamped Energy Intervention).
  - 3.3.4. Hàm mất mát huấn luyện (Joint Negative Log-Likelihood Loss).
- 3.4. Các Mô hình Đối chứng (Majority Baseline, Black-box Classifier, Soft CBM, Hard CBM).

### CHƯƠNG 4: THIẾT KẾ THỰC NGHIỆM (EXPERIMENTAL SETUP)
- 4.1. Môi trường Thực nghiệm và Công cụ Triển khai (PyTorch, Phần cứng GPU).
- 4.2. Cấu hình Siêu tham số (Hyperparameters: Learning rate, Batch size, 30 Epochs, Cosine Annealing).
- 4.3. Tiêu chí Đánh giá và Chỉ số Đo lường (Metrics: Macro-F1, Balanced Acc, ROC-AUC, Exact Match, AUC-IC).
- 4.4. Quy trình Lựa chọn Checkpoint và Điều chỉnh Ngưỡng Quyết định (Decision Threshold Tuning trên Validation).
- 4.5. Thiết kế Quy trình Can thiệp Khái niệm Thực nghiệm (Random vs Uncertainty-based Intervention).

### CHƯƠNG 5: KẾT QUẢ VÀ BÀN LUẬN (RESULTS & DISCUSSION)
- 5.1. Kết quả Hiệu năng Tiên lượng Tử vong (Trả lời RQ1): Bảng tổng hợp so sánh tất cả các mô hình, so sánh cân bằng Trade-off.
- 5.2. Kết quả Dự đoán 14 Khái niệm CheXpert (Trả lời RQ2): Đánh giá chi tiết từng triệu chứng, phân tích các tổn thương nặng.
- 5.3. Kết quả Can thiệp Triệu chứng (Trả lời RQ3): Phân tích đường cong can thiệp (Intervention Curves) và chỉ số AUC-IC khi sửa $m = 0 \dots 14$ triệu chứng.
- 5.4. Hiệu ứng Lan truyền Tri thức khi Can thiệp: Đánh giá khả năng mô hình tự động hiệu chỉnh các triệu chứng chưa can thiệp (Unobserved Concepts Recovery).
- 5.5. Phân tích Chuyên sâu Data-Centric, Tính Bất nhất & Phân tích Lỗi (Trả lời RQ4):
  - Định lượng trần dữ liệu lý thuyết và hiện tượng bất nhất lâm sàng giữa tổn thương X-quang và nguy cơ tử vong.
  - Phân tích ma trận 4 góc lỗi chẩn đoán (Concept Đúng/Sai × Tử vong Đúng/Sai).
  - Đánh giá thực nghiệm Data-centric: Tác động của độ phân giải ảnh (224 vs 512) và chiến lược xử lý nhãn bất định CheXpert.
  - Nghiên cứu ca lâm sàng điển hình (Case Studies trực quan).
- 5.6. Bàn luận Tổng quát: Ý nghĩa trong việc hỗ trợ bác sĩ ICU cảnh báo sớm nguy cơ tử vong và khả năng ứng dụng thực tế.

### CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
- 6.1. Tóm tắt Đóng góp và Kết quả Chính của Khóa luận.
- 6.2. Hạn chế của Đề tài (Giới hạn phạm vi dữ liệu đơn trung tâm, can thiệp mô phỏng bằng ground-truth).
- 6.3. Hướng Nghiên cứu và Phát triển Tương lai (Tích hợp thêm chuỗi thời gian sinh hiệu động từ MIMIC-IV, kiểm chứng lâm sàng tại bệnh viện).

---

## 9. QUẢN TRỊ RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro Tiềm ẩn | Mức độ Ảnh hưởng | Xác suất | Giải pháp Giảm thiểu & Phương án Dự phòng Đã Thiết lập |
|:---|:---:|:---:|:---|
| **1. Mất cân bằng dữ liệu ở nhãn tử vong (`hospital_expire_flag`)** | Cao | Rất cao | • Sử dụng **Macro-F1** và **Balanced Accuracy** làm metric chính.<br>• Áp dụng Pos-weight trong BCE loss.<br>• Sử dụng Pretrained Backbone ImageNet để ổn định biểu diễn đặc trưng. |
| **2. Độ trễ tính toán khi duyệt 16.384 trạng thái** | Thấp | Thấp | • Trên GPU, tensor kích thước $(16, 16384)$ chỉ chiếm khoảng 2 MB bộ nhớ RAM, phép tính energy vectorization chạy trong vài mili-giây, hoàn toàn khả thi. |
| **3. Mô hình CBM bị sụt giảm hiệu năng so với Black-box** | Trung bình | Trung bình | • Báo cáo trung thực sự đánh đổi (Trade-off). Tận dụng thành phần $E_{global}$ trong ECBM để bù đắp dung lượng biểu diễn bị mất của bottleneck.<br>• Làm nổi bật ưu thế vượt trội của ECBM ở khâu **Can thiệp Khái niệm (Intervention)** mà Black-box không thể có được. |
| **4. Bệnh nhân X-quang không khớp với hồ sơ nhập viện MIMIC-IV** | Trung bình | Thấp | • Đã kiểm tra thực tế: Hơn 3.637 bệnh nhân trong 5.000 ca khớp hoàn toàn với `admissions.csv.gz`. Ta chỉ cần lọc các ca có đủ cả ảnh và nhãn tử vong là có ngay một tập dữ liệu chuẩn mực. |
| **5. Quá tải thời gian / Chậm tiến độ viết luận văn** | Cao | Trung bình | • Giảm số epoch từ 50 xuống **30 epochs** (tiết kiệm 40% thời gian chạy mà vẫn đảm bảo độ hội tụ).<br>• Khởi động viết luận văn song song từ Tuần 18 thay vì dồn vào cuối kỳ. |

---

## 10. BẢNG THEO DÕI TIẾN ĐỘ HIỆN TẠI (PROJECT STATUS CHECKLIST)

### Trạng thái Cập nhật đến Hiện tại:

| Hạng mục Công việc | Trạng thái | Chi tiết Thực tế Đạt được |
|:---|:---:|:---|
| **Nghiên cứu tài liệu & Đề cương chi tiết** | ✅ **Hoàn thành** | Đề cương 24 tuần, công thức ECBM và cấu trúc 6 chương chuyên biệt cho 14 Concepts + Tử vong MIMIC-IV đã chốt. |
| **Tải và lưu trữ dữ liệu MIMIC-CXR 5.000 ảnh** | ✅ **Hoàn thành** | Đã tải đủ 100% (**5.000 / 5.000 ảnh**, 0 lỗi) và lưu tại `data/mimic_cxr/`. |
| **Dữ liệu lâm sàng MIMIC-IV v3.1 & Ghép nối** | ✅ **Hoàn thành** | Đã tạo manifest bất biến `manifest_mortality.csv` (Train: 4.000, Val: 500, Test: 500, 0 rò rỉ bệnh nhân). |
| **Module Dataset & Transforms chuẩn** | ✅ **Hoàn thành** | `src/dataset.py` hỗ trợ LetterboxResize 224x224, nạp 14 khái niệm CheXpert và nhãn tử vong. |
| **Mô hình Black-box Baseline (3 Seeds)** | ✅ **Hoàn thành** | **Đã huấn luyện đủ 3 seeds (42, 123, 2026)** với 30 epochs/run: **Macro-F1 = 0.5729 ± 0.0091**, **Bal-Acc = 0.6496 ± 0.0072**, **ROC-AUC = 0.7499 ± 0.0224**. |
| **Mô hình CBM (Soft & Hard CBM)** | 🔄 **Đang thực nghiệm** | Codebase sẵn sàng 100% trên GitHub `LungXplain`, script đa seed `run_suite_mimic.py` đang chạy trên Kaggle GPU. |
| **Mô hình Energy-Based CBM (ECBM)** | 🔄 **Sẵn sàng huấn luyện** | Triển khai Exact Inference qua 16.384 trạng thái (~0.7ms/batch) và Clamped Energy Intervention. |
| **Thực nghiệm Can thiệp (Intervention) 14 concepts** | ⏳ **Sẵn sàng chạy** | Script `run_intervention_mimic.py` đã hoàn tất, chờ nạp checkpoint từ CBM/ECBM. |
| **Viết bản thảo Luận văn 6 chương** | ⏳ **Dự kiến Tuần 18** | Đã lập dàn ý chi tiết từng mục, sẵn sàng điền nội dung kết quả thực nghiệm. |

---

> [!TIP]
> **Hành động Ngay Tiếp theo:**
> 1. Viết script ghép nối nhanh để bổ sung cột `hospital_expire_flag` vào manifest từ `mimic-iv-3.1`.
> 2. Cập nhật `src/dataset.py` và `label_mapping.json` hỗ trợ đầy đủ 14 concepts CheXpert.
> 3. Chạy Smoke test một batch nhỏ để kiểm tra luồng dữ liệu trước khi huấn luyện mô hình.
