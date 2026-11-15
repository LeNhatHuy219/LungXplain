# Dataset v5 — chọn lại 5.000 ca cho đầy đủ hơn (protocol chốt trước khi chọn)

Ngày chốt: **14/11/2026**, trước khi chạy selection. Người dùng cho biết 5.000 ảnh
chỉ là một mẫu trích từ MIMIC, nên có thể chọn lại để dataset đầy đủ hơn. v3/v4 và mọi
run đã audit được giữ nguyên làm lịch sử; v5 nằm trong thư mục mới và mọi mô hình phải
huấn luyện lại trên v5.

## Lý do (từ kiểm tra độc lập ngày 14/11)

- 3.369/5.000 ca v3 không có xét nghiệm nào trước ảnh. Phân tích toàn pool 42.858 bệnh
  nhân cho thấy nguyên nhân chính là lab lấy ở cấp cứu trước khi nhập viện có `hadm_id`
  trống, bị quy tắc strict loại (`data/audits/pool_analysis_*`).
- Không có sinh hiệu; chỉ 46 ca có nhãn đọc phim của bác sĩ (5 ca trong test).

## Quyết định của người dùng

| Mục | Quy tắc v5 |
|---|---|
| Đơn vị | Một bệnh nhân, một lượt nhập viện, ảnh AP/PA đầu tiên của lượt đó (như v3) |
| Ghép ảnh–lượt | Ảnh trong `[min(edregtime, admittime), dischtime]`, chỉ thuộc đúng một lượt, trước `deathtime`; `hospital_expire_flag` khớp `deathtime` (như v3) |
| Liên kết lab | Cùng `subject_id` và (`hadm_id` đúng lượt **hoặc** `hadm_id` trống và lượt đích là lượt duy nhất của bệnh nhân chứa cả `charttime` và `storetime`) |
| Cửa sổ lab | `max(encounter start, t0 − 48 h) ≤ charttime ≤ storetime ≤ t0` |
| Chọn giá trị | Kết quả hợp lệ mới nhất theo (`charttime`, `storetime`); hòa mà mâu thuẫn → thiếu; không lùi về kết quả cũ; làm sạch số và đơn vị như v3 |
| Tiêu chí lab | **Bắt buộc đủ 11 lab thường quy** (bicarbonate, chloride, creatinine, glucose, potassium, sodium, BUN, hematocrit, hemoglobin, platelets, WBC); albumin là biến tuỳ chọn có cờ thiếu |
| Sinh hiệu | Triage MIMIC-IV-ED: nhiệt độ, nhịp tim, nhịp thở, SpO₂, huyết áp tâm thu/tâm trương, acuity. Lượt ED liên kết đúng `hadm_id`, `intime` trong `[encounter start, t0]`. Giá trị ngoài khoảng sinh lý → thiếu, có cờ thiếu; ca không qua ED có cờ thiếu. Không dùng chief complaint (văn bản) và pain |
| Cỡ mẫu | Đúng 5.000 bệnh nhân, 500 tử vong (10%); split 4.000/500/500, tử vong 400/50/50 |
| Nhãn bác sĩ | Mọi ca đủ điều kiện có nghiên cứu nằm trong `mimic-cxr-2.1.0-test-set-labeled.csv` được đưa vào **test** (chọn không dựa vào outcome). Nhãn này lưu file riêng, không phải đầu vào model |
| Chọn ca | Một lượt mỗi bệnh nhân. Nếu bệnh nhân có lượt mang nhãn bác sĩ thì ưu tiên lượt đó, còn lại theo thứ tự hash seed 42. Tử vong và sống được lấy theo thứ tự hash; không dùng model score, không dùng giá trị lab để chọn ngoài tiêu chí đủ 11 lab |
| Nhãn concept | Giữ raw CheXpert, mục tiêu U-Zeros kiểu v3 và mask/state kiểu v4 |

## Hệ quả cần ghi trong bài

- Quần thể là bệnh nhân **đã có xét nghiệm thường quy trước khi chụp phim**; không đại
  diện cho mọi bệnh nhân chụp X-quang. Báo cáo so sánh nhóm được giữ với pool loại.
- Tỷ lệ tử vong được làm giàu lên 10% (pool đủ lab khoảng 2,5%).
- Lab ED được ghép bằng thời gian khi `hadm_id` trống; quy tắc "duy nhất một lượt chứa
  cả hai thời điểm" ngăn ghép nhầm sang lượt khác.

## Kết quả selection hiện hành (15/11/2026, `data/mimic_cxr/cohort_v5r/selection_report.json`)

Selection sau khi sửa quy tắc lab mới nhất (mục kế tiếp). Tập chọn và split **trùng khớp
hoàn toàn** với bản dựng lại độc lập của `validate_dataset_v5.py` (pandas, quét lại
158.374.764 dòng labevents).

| Bước | Số lượng |
|---|---:|
| Ảnh AP/PA trong MIMIC-CXR | 243.334 |
| Lượt nhập viện có ảnh AP/PA trong encounter | 78.082 |
| Ảnh đầu tiên đủ điều kiện (loại 309 chồng lượt, 4 lệch outcome/deathtime, 3 không trước tử vong, 1 thiếu nhãn) | 77.765 |
| Lượt có đủ 11 lab thường quy | 34.227 |
| Bệnh nhân đủ điều kiện (một lượt mỗi người) | 24.629 (759 tử vong) |
| **Được chọn** | **5.000 (500 tử vong)** |

- Split 4.000/500/500, tử vong 400/50/50. 96 ca có nhãn bác sĩ, cả 96 trong test (6 tử vong).
- 11 lab bắt buộc: 0 ô thiếu. Albumin có ở 1.507 ca. Giá trị lab được chọn: 28.591 ghép
  đúng `hadm_id`, 27.916 ghép bằng thời gian khi `hadm_id` trống.
- AP 3.044 / PA 1.956; tử vong AP 445, PA 55. 760 bệnh nhân trùng với v3.
- 4.989 ảnh đã có trên máy (khớp checksum, phần lớn tải cho `cohort_v5`), 11 ảnh cần tải.

Lịch sử `cohort_v5` (14/11, có lỗi): 34.153 lượt đủ lab, 24.590 bệnh nhân (758 tử vong),
4.525 ảnh được tải; 12 bệnh nhân khác với `cohort_v5r`.

## Sửa lỗi selection (15/11/2026, trước mọi bước build và mô hình)

Khi viết validator độc lập, phát hiện selector lấy "kết quả mới nhất" bằng cách lấy
`charttime` lớn nhất và `storetime` lớn nhất **riêng rẽ** rồi đòi một xét nghiệm khớp
cả hai. Nếu một mẫu lấy sớm hơn có kết quả trả về muộn hơn mẫu lấy sau thì không dòng
nào khớp và lab đó bị coi là thiếu. Protocol yêu cầu thứ tự theo cặp
(`charttime`, `storetime`). Đo trên toàn pool: 252/443.529 nhóm (lượt × lab) bị mất oan;
lượt đủ 11 lab 34.153 → 34.227; bệnh nhân đủ điều kiện 24.590 → 24.629 (tử vong
758 → 759); 12/5.000 bệnh nhân được chọn thay đổi; 4.989 ca còn lại có giá trị lab
giống hệt. Lỗi không thể làm sai một giá trị đã chọn, chỉ gây thiếu oan.

Đã sửa bằng `latest_by_time` (có unit test) và chạy lại selection vào
`data/mimic_cxr/cohort_v5r/`. `cohort_v5/` được giữ làm lịch sử, ảnh của nó được tái
sử dụng sau khi kiểm checksum. Không dùng outcome, giá trị lab hay model nào để quyết
định việc sửa; chỉ đối chiếu với câu chữ protocol.

## Ảnh không lấy được từ bản phân phối (15/11/2026, ghi trước khi chọn lại)

Một ảnh của `cohort_v5r` (train, sống, không nhãn bác sĩ) có trong `SHA256SUMS` của
MIMIC-CXR-JPG 2.1.0 nhưng máy chủ PhysioNet trả **HTTP 404** ở hai lần chạy độc lập,
mỗi lần 4 lượt (cùng file cũng là ảnh duy nhất không tải được của `cohort_v5`).

Quy tắc: ảnh trả 404 lặp lại, sai checksum hoặc không giải mã được là **không có sẵn**;
ảnh đầu tiên của lượt nhập viện đó bị loại với trạng thái
`image_unavailable_in_distribution` (không lùi sang ảnh khác của lượt), và selection
chạy lại nguyên quy tắc cũ. Danh sách và bằng chứng nằm ở
`data/mimic_cxr/unavailable_images_v5.csv` (đầu vào cố định của selector và validator).
Không dùng outcome hay giá trị lab để quyết định. Kết quả ghi vào `cohort_v5r2`.

## Bổ sung protocol cho bước build (chốt 15/11/2026, trước khi tải MIMIC-IV-ED)

Ghi lại trước khi có bất kỳ giá trị nào của MIMIC-IV-ED trên máy, để ngưỡng làm sạch
không được chọn sau khi nhìn dữ liệu. Bước build nằm ở file mới
`experiments/finalize_dataset_v5.py` và từ chối chạy nếu script select khác với
`selector_sha256` ghi trong `selection_report.json` của cohort đầu vào (`cohort_v5r`).

| Mục | Quy tắc |
|---|---|
| Nguồn | MIMIC-IV-ED 2.2 `ed/edstays.csv.gz`, `ed/triage.csv.gz`, kiểm SHA-256 với `SHA256SUMS.txt` của PhysioNet. `vitalsign.csv.gz` được tải nhưng **không** dùng trong mô hình chính |
| Liên kết | `edstays.hadm_id` = `hadm_id` đích, cùng `subject_id`, `encounter start ≤ intime ≤ t0`. Nhiều lượt ED hợp lệ → lấy `intime` muộn nhất; hòa → thiếu. `triage.stay_id` phải duy nhất |
| Khoảng hợp lệ (gồm 2 đầu) | nhiệt độ 86–113 °F (MIMIC-IV-ED ghi °F; không tự đổi °C), nhịp tim 20–300/phút, nhịp thở 3–80/phút, SpO₂ 50–100 %, HA tâm thu 40–300 mmHg, HA tâm trương 10–200 mmHg, acuity là số nguyên 1–5 |
| Ngoài khoảng / không phải số | → thiếu, `*_missing=1`. Nếu tâm trương ≥ tâm thu thì cả hai → thiếu |
| Không có lượt ED liên kết | toàn bộ biến triage thiếu, cờ thiếu = 1 |
| Kiểm tra thời gian | báo số ca có `t0 − intime < 30 phút` (triage có thể được ghi sát giờ chụp); không loại ca |
| Concept | Dạng v3: cột concept = U-Zeros (`raw == 1`), `raw_`, `ignore_` (−1), `known_` (0/1), `n_findings`. Dạng v4: `nullable_` (chỉ 0/1 tường minh, còn lại trống), `label_mask_`, `state_` |
| Nhãn bác sĩ | File riêng `radiologist_labels_test.csv` cho các ca có nhãn; cột `Airspace Opacity` của nguồn được đối chiếu với `Lung Opacity`. Không phải đầu vào mô hình |
| Manifest phân tích độ nhạy | `manifest_temporal_sensitivity.csv` = chỉ giữ lab ghép đúng `hadm_id`; mọi ô lab ghép bằng thời gian bị để trống với cờ thiếu = 1 (ngược với v3, nơi bản chính là strict) |
| Biến mô hình | tuổi, cờ tuổi bị chặn, giới, 12 lab, 7 biến triage, cùng cờ thiếu của từng biến. Cờ thiếu của 11 lab bắt buộc luôn bằng 0 ở bản chính; vẫn giữ để cùng schema với bản độ nhạy |

## Trình tự

1. `experiments/build_dataset_v5.py select` → `data/mimic_cxr/cohort_v5r/` (manifest chọn
   ca, danh sách ảnh cần tải, báo cáo tổng hợp). Chạy sau khi tải xong danh sách của
   `cohort_v5/` để tái sử dụng ảnh đã tải; chỉ vài ảnh mới cần tải thêm.
2. Người dùng tải ảnh còn thiếu và MIMIC-IV-ED bằng tài khoản PhysioNet của mình.
3. `experiments/finalize_dataset_v5.py` → `data/mimic_cxr/multimodal_v5/`;
   `experiments/validate_dataset_v5.py --write` dựng lại từ nguồn bằng code riêng; chỉ
   dùng khi `passed=true`.
4. Huấn luyện lại v2 (Kaggle) và baseline cùng cohort trên v5.
