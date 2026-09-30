#!/usr/bin/env python
"""Build experiments/feature_ablation.xlsx: every feature test, its metric deltas and why.

Reads the finished runs of each ablation (local rot1-* runs in mlflow.db, the Kaggle
runs in experiments/kaggle-*/results/mlflow.db), writes the reference and variant
metric per model as inputs, and computes every delta and summary with formulas.
Read-only on the MLflow stores.

    python scripts/feature_sheet.py
"""
from __future__ import annotations

import os
import sqlite3
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spark_session import path  # noqa: E402

OUT = path("experiments", "feature_ablation.xlsx")
REG = [("linear_regression", "r2"), ("random_forest_regressor", "r2"), ("gbt_regressor", "r2")]
CLF = [(m, k) for m in ("linear_svc", "random_forest_classifier", "gbt_classifier")
       for k in ("areaUnderROC", "areaUnderPR")]

LOCAL = "mlflow.db"


def kg(folder):
    return f"experiments/{folder}/results/mlflow.db"


# id, task, change, columns, decision, reason, source folder, split note,
# db, reference experiment, variant experiment, variant arm (None = nopca), metric set
TESTS = [
    ("L1", "A, C", "Bỏ frequency encoding sân bay", "freq_origin, freq_dest", "Bỏ",
     "Có hay không đều hòa; tọa độ sân bay đã cho model biết sân bay nào",
     "rot1-predeparture_nofreq", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture", "rot1-predeparture_nofreq", None, "both"),
    ("L2", "A, C", "Thêm vòng quay tàu bay theo lịch", "leg_of_day, has_prev, turn_slack", "Giữ",
     "Biết trước từ lịch bay; model cây dùng (turn_slack 4 đến 7% importance)",
     "rot1-predeparture_sched", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture", "rot1-predeparture_sched", None, "both"),
    ("L3", "A, C", "Thêm độ trễ lúc đi của chuyến trước", "prev_dep_delay", "Bỏ (thay ở L5)",
     "Riêng nó có ích (R² +0.20), nhưng r 0.93 với prev_arr_delay: khi đã có prev_arr_delay thì thay bằng prev_air_gain hòa (L5), nên bỏ",
     "rot1-predeparture_inbound_dep", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_sched", "rot1-predeparture_inbound_dep", None, "both"),
    ("L4", "A, C", "Thêm độ trễ lúc đến của chuyến trước", "prev_arr_delay, inbound_overrun", "Giữ",
     "Tín hiệu mạnh nhất; inbound_overrun hạng nhất ở cả 4 model cây",
     "rot1-predeparture_inbound", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_inbound_dep", "rot1-predeparture_inbound", None, "both"),
    ("L5", "A, C", "Đổi prev_dep_delay thành prev_air_gain", "prev_dep_delay -> prev_air_gain",
     "Bỏ prev_dep_delay", "Hòa: prev_dep_delay không mang thêm gì khi đã có prev_arr_delay",
     "rot1-predeparture_inbound_gain", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_inbound", "rot1-predeparture_inbound_gain", None, "both"),
    ("L6", "A, C", "Bỏ has_prev và prev_dep_delay", "has_prev, prev_dep_delay", "Giữ has_prev",
     "Bỏ cả hai thì tụt; has_prev báo cho model các cột prev_* của chuyến đầu ngày là giá trị điền",
     "rot1-predeparture_inbound_lean", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_inbound", "rot1-predeparture_inbound_lean", None, "both"),
    ("L7", "A, C", "Thêm độ đông theo lịch (không có vòng quay)",
     "sched_origin_hour, sched_dest_hour, origin_day_ratio", "Bỏ",
     "r <= 0.04 với target; lệch nhỏ và không cùng chiều giữa các model", "rot1-predeparture_congestion", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture", "rot1-predeparture_congestion", None, "both"),
    ("L8", "A, C", "Thêm độ đông theo lịch (có vòng quay theo lịch)",
     "sched_origin_hour, sched_dest_hour, origin_day_ratio", "Bỏ", "Hòa",
     "rot1-predeparture_sched_congestion", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_sched", "rot1-predeparture_sched_congestion", None, "both"),
    ("L9", "A, C", "Thêm độ đông theo lịch (có đủ vòng quay)",
     "sched_origin_hour, sched_dest_hour, origin_day_ratio", "Bỏ", "Hòa",
     "rot1-predeparture_inbound_congestion", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_inbound", "rot1-predeparture_inbound_congestion", None, "both"),
    ("L10", "A, C", "Bỏ log_distance", "log_distance", "Bỏ", "Tụt rất nhẹ (R² trung bình −0.004), sát mức nhiễu; bỏ cho gọn",
     "rot1-predeparture_final_nodist", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_final", "rot1-predeparture_final_nodist", None, "both"),
    ("L11", "A, C", "Bỏ prev_air_gain", "prev_air_gain", "Bỏ", "Bỏ đi không đổi gì, không cùng chiều",
     "rot1-predeparture_final_nogain", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_final", "rot1-predeparture_final_nogain", None, "both"),
    ("L12", "A, C", "Chuyến đầu ngày: điền 0 thay median cho prev_*",
     "prev_arr_delay, inbound_overrun, prev_air_gain", "Giữ median",
     "Hòa; giữ cách cũ", "rot1-predeparture_final_zero", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_final", "rot1-predeparture_final_zero", None, "both"),
    ("L13", "A, C", "Bỏ nhóm cột trùng theo EDA",
     "log_sched_time, log_gc_distance, SCHEDULE_SPEED_MPH, ROUTE_DETOUR, SCHED_DEP_MIN, "
     "IS_WEEKEND, DAY, freq_route", "Bỏ",
     "Trùng nhau (r 0.94 đến 1.00) hoặc r < 0.01 với target. Bỏ cả nhóm thì tụt nhẹ (R² −0.008, AUC −0.006), đổi lại bỏ được UDF Haversine chậm; chưa test bỏ riêng từng cột",
     "rot1-predeparture_all", "1%, chia ngẫu nhiên, máy nhà",
     LOCAL, "rot1-predeparture_all", "rot1-predeparture", None, "both"),
    ("K1", "A, C", "Giờ và tháng sang sin/cos", "DEP_HOUR, MONTH -> hour_sin/cos, month_sin/cos",
     "Xem K7", "Hòa với one-hot; tách riêng giờ và tháng ở K7",
     "kaggle-cyclic-20260927", "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-cyclic-20260927"), "kg-v2", "kg-v2_cyclic", None, "both"),
    ("K2", "A, C", "Thêm lại ROUTE_DETOUR", "ROUTE_DETOUR", "Bỏ",
     "Lasso đưa hệ số về 0, cây không cải thiện", "kaggle-coords-detour-20260928",
     "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-coords-detour-20260928"), "kg-v2", "kg-v2_detour", None, "both"),
    ("K3", "A, C", "Bỏ tọa độ sân bay", "ORIGIN_LAT, ORIGIN_LON, DEST_LAT, DEST_LON", "Giữ",
     "Cây mất khả năng nhận ra sân bay, RF classifier tụt rõ", "kaggle-coords-detour-20260928",
     "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-coords-detour-20260928"), "kg-v2", "kg-v2_nocoords", None, "both"),
    ("K4", "A, C", "Bỏ thứ trong tuần", "DAY_OF_WEEK", "Bỏ (xác nhận ở K9)",
     "Không đóng góp đo được; riêng GBT regressor tụt 0.018 do CV đổi maxBins, không phải do cột", "kaggle-dow-month-20260928", "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-dow-month-20260928"), "kg-v2", "kg-v2_nodow", None, "both"),
    ("K5", "A, C", "Bỏ tháng", "MONTH", "Giữ", "Cả 7 model đều tụt nhẹ, cùng chiều",
     "kaggle-dow-month-20260928", "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-dow-month-20260928"), "kg-v2", "kg-v2_nomonth", None, "both"),
    ("K6", "A, C", "Thứ trong tuần sang sin/cos", "DAY_OF_WEEK -> dow_sin, dow_cos", "Bỏ cả hai",
     "Mã hóa kiểu nào cũng hòa", "kaggle-dow-cyclic-20260928", "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-dow-cyclic-20260928"), "kg-v2", "kg-v2_dowcyc", None, "both"),
    ("K7a", "A, C", "Chỉ giờ sang sin/cos", "DEP_HOUR -> hour_sin, hour_cos", "Dùng sin/cos",
     "Hòa ở cây, LinearSVC tốt hơn; 2 cột thay 24", "kaggle-hour-month-cyclic-20260928",
     "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-hour-month-cyclic-20260928"), "kg-v2", "kg-v2_hourcyc", None, "both"),
    ("K7b", "A, C", "Chỉ tháng sang sin/cos", "MONTH -> 2 cặp sin/cos", "Giữ one-hot",
     "Hòa nhưng nghiêng về tệ hơn (GBT regressor −0.018 là do CV đổi maxBins)", "kaggle-hour-month-cyclic-20260928",
     "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-hour-month-cyclic-20260928"), "kg-v2", "kg-v2_monthcyc", None, "both"),
    ("K8a", "A, C", "Bỏ giờ khởi hành one-hot", "DEP_HOUR", "Giữ (dạng sin/cos)",
     "Model tuyến tính tụt; cây không cần", "kaggle-sched-time-20260928",
     "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-sched-time-20260928"), "kg-v2", "kg-v2_nodephour", None, "both"),
    ("K8b", "A, C", "Bỏ giờ hạ cánh", "SCHED_ARR_MIN", "Giữ", "Cây tụt; model tuyến tính không đổi (GBT regressor −0.016 một phần do CV đổi maxBins)",
     "kaggle-sched-time-20260928", "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-sched-time-20260928"), "kg-v2", "kg-v2_noarr", None, "both"),
    ("K8c", "A, C", "Bỏ cả hai cột giờ", "DEP_HOUR, SCHED_ARR_MIN", "Giữ",
     "Mọi model tụt, mạnh nhất trong các test", "kaggle-sched-time-20260928",
     "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-sched-time-20260928"), "kg-v2", "kg-v2_notime", None, "both"),
    ("K8d", "A, C", "Thêm phút khởi hành", "SCHED_DEP_MIN", "Bỏ",
     "Không được gì; r 0.998 với DEP_HOUR", "kaggle-sched-time-20260928",
     "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-sched-time-20260928"), "kg-v2", "kg-v2_depmin", None, "both"),
    ("K9a", "A, C", "Bỏ thứ trong tuần, seed 42", "DAY_OF_WEEK", "Bỏ", "Hòa, lặp lại 3 seed",
     "kaggle-ladder-s42-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-ladder-s42-20260929"), "kg-v2", "kg-v2_nodow", None, "both"),
    ("K9b", "A, C", "Bỏ thứ trong tuần, seed 1", "DAY_OF_WEEK", "Bỏ", "Hòa",
     "kaggle-noise-s1-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-noise-s1-20260929"), "kg-v2_s1", "kg-v2_nodow_s1", None, "both"),
    ("K9c", "A, C", "Bỏ thứ trong tuần, seed 2", "DAY_OF_WEEK", "Bỏ", "Hòa",
     "kaggle-noise-s2-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-noise-s2-20260929"), "kg-v2_s2", "kg-v2_nodow_s2", None, "both"),
    ("K10a", "A, C", "Giờ sang sin/cos, seed 42", "DEP_HOUR -> hour_sin, hour_cos", "Dùng sin/cos",
     "Cây hòa, LinearSVC tốt hơn ở cả 3 seed", "kaggle-ladder-s42-20260929",
     "Kaggle 1%, chia theo ngày",
     kg("kaggle-ladder-s42-20260929"), "kg-v2", "kg-v2_hourcyc", None, "both"),
    ("K10b", "A, C", "Giờ sang sin/cos, seed 1", "DEP_HOUR -> hour_sin, hour_cos", "Dùng sin/cos",
     "", "kaggle-noise-s1-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-noise-s1-20260929"), "kg-v2_s1", "kg-v2_hourcyc_s1", None, "both"),
    ("K10c", "A, C", "Giờ sang sin/cos, seed 2", "DEP_HOUR -> hour_sin, hour_cos", "Dùng sin/cos",
     "", "kaggle-noise-s2-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-noise-s2-20260929"), "kg-v2_s2", "kg-v2_hourcyc_s2", None, "both"),
    ("K11", "A, C", "Bỏ 2 cột biết khi chuyến trước đã hạ cánh (chỉ còn lịch bay)",
     "prev_arr_delay, inbound_overrun", "Giữ",
     "Phần lớn điểm đến từ đây: AUC 0.82 xuống 0.68", "kaggle-ladder-s42-20260929",
     "Kaggle 1%, chia theo ngày",
     kg("kaggle-ladder-s42-20260929"), "kg-v2", "kg-v2_sched", None, "both"),
    ("K12a", "A, C", "PCA trên toàn vector (k 51)", "arm PCA", "Không dùng",
     "Thua PCA chỉ trên cột số ở 6/7 model", "kaggle-pca-20260927", "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-pca-20260927"), "kg-v2", "kg-v2_pca_all", "pca", "both"),
    ("K12b", "A, C", "PCA chỉ trên cột số (k 8)", "arm PCA", "Dùng cho arm PCA",
     "Gần không PCA cho model tuyến tính; cây vẫn tốt nhất khi không PCA",
     "kaggle-pca-20260927", "Kaggle 1%, chia ngẫu nhiên",
     kg("kaggle-pca-20260927"), "kg-v2", "kg-v2_pca_numeric", "pca", "both"),
    ("G1", "B", "Bỏ tốc độ theo lịch", "sched_mph", "Giữ", "Mọi model tụt, model tuyến tính tụt mạnh",
     "kaggle-check-gain-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-check-gain-20260929"), "kg-gain", "kg-gain_nomph", None, "reg"),
    ("G2", "B", "Thêm độ đông ở sân bay đích", "sched_dest_hour", "Giữ",
     "Model tuyến tính tăng rõ, cây hòa", "kaggle-check-gain-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-check-gain-20260929"), "kg-gain", "kg-gain_cong", None, "reg"),
    ("G3", "B", "Thay log_distance, sched_mph bằng dạng thô",
     "DISTANCE, SCHEDULED_TIME thay log_distance, sched_mph", "Không dùng", "Hòa",
     "kaggle-check-gain2-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-check-gain2-20260929"), "kg-gain_cong", "kg-gain_phys", None, "reg"),
    ("G4", "B", "Thêm DISTANCE, SCHEDULED_TIME", "DISTANCE, SCHEDULED_TIME", "Giữ",
     "Mọi model tăng; target phụ thuộc vào hiệu giữa chúng (độ dư lịch)",
     "kaggle-check-gain2-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-check-gain2-20260929"), "kg-gain_cong", "kg-gain_plus", None, "reg"),
    ("G5", "B", "Bỏ DISTANCE", "DISTANCE", "Giữ",
     "GBT y hệt (log đơn điệu); model tuyến tính tụt vì thời gian bay tuyến tính theo quãng đường",
     "kaggle-check-gain3-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-check-gain3-20260929"), "kg-gain_plus", "kg-gain_nodist", None, "reg"),
    ("G6", "B", "Bỏ log_distance, sched_mph", "log_distance, sched_mph", "Giữ", "Cây tụt",
     "kaggle-check-gain3-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-check-gain3-20260929"), "kg-gain_plus", "kg-gain_nolog", None, "reg"),
    ("G7", "B", "Chặn trần DEPARTURE_DELAY ở 120 phút", "DEPARTURE_DELAY", "Không cần",
     "Hòa: với target gain cột này gần như không mang thông tin (r 0.009)",
     "kaggle-check-cap-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-check-cap-20260929"), "kg-gain_plus", "kg-gain_cap120", None, "reg"),
    ("G8", "B", "Bỏ DEPARTURE_DELAY", "DEPARTURE_DELAY", "Bỏ",
     "Hòa; gain phẳng theo nó (trung vị −6 phút ở mọi mức trễ). Vẫn dùng qua phép cộng arrival = DEPARTURE_DELAY + gain",
     "kaggle-check-cap-20260929", "Kaggle 1%, chia theo ngày",
     kg("kaggle-check-cap-20260929"), "kg-gain_plus", "kg-gain_nodep", None, "reg"),
]

EDA = [
    ("DEPARTURE_DELAY, TAXI_OUT, WHEELS_OFF", "A, C", "Cấm",
     "Chỉ biết sau khi máy bay rời cổng; code tự chặn (PRE_DEPARTURE_FORBIDDEN)"),
    ("te_origin, te_dest, te_route", "A, B, C", "Cấm", "Target encoding, tính từ nhãn"),
    ("ARRIVAL_TIME, ELAPSED_TIME, AIR_TIME, WHEELS_ON, TAXI_IN, 5 cột *_DELAY", "A, B, C", "Cấm",
     "Chỉ biết sau khi hạ cánh; xóa từ bước chuẩn bị dữ liệu (flight_schema.LEAKY_COLUMNS)"),
    ("SCHEDULED_TIME, log_sched_time", "A, C", "Bỏ", "r 0.97 đến 0.98 với quãng đường; không có ablation riêng"),
    ("GC_DISTANCE_MI, log_gc_distance", "A, B, C", "Bỏ", "r 1.00 với log_distance; cần UDF Python chậm"),
    ("IS_WEEKEND", "A, C", "Bỏ", "Suy ra từ DAY_OF_WEEK"),
    ("DAY (ngày trong tháng)", "A, C", "Bỏ", "r < 0.01, không có quy luật"),
    ("AIRLINE", "A, B, C", "Giữ", "Tỷ lệ trễ nặng từ 4.1% (Hawaiian) đến 19.3% (Spirit); chưa có ablation bỏ riêng"),
]


def load(db, exp):
    con = sqlite3.connect(f"file:{path(db)}?mode=ro", uri=True)
    out = {}
    for name, rid in con.execute(
            "select r.name, r.run_uuid from runs r join experiments e using(experiment_id) "
            "where e.name=? and r.status='FINISHED' order by r.start_time", (exp,)):
        m = dict(con.execute("select key, value from latest_metrics where run_uuid=?", (rid,)))
        if m:
            out[name] = m          # newest finished run per name wins
    return out


def main():
    thin = Side(style="thin", color="C9CDD6")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="1F3A6B")
    head_font = Font(name="Arial", bold=True, color="FFFFFF")
    base = Font(name="Arial")
    blue = Font(name="Arial", color="0000FF")
    wrap = Alignment(wrap_text=True, vertical="top")

    wb = Workbook()
    ws = wb.active
    ws.title = "Tổng hợp"
    det = wb.create_sheet("Chi tiết")
    eda = wb.create_sheet("Bỏ theo EDA")
    note = wb.create_sheet("Chú thích")

    # ---- detail rows (inputs) ----
    dh = ["ID", "Nguồn", "Bộ tham chiếu", "Bộ thử", "Model", "Metric",
          "Tham chiếu", "Thử", "Δ (thử − tham chiếu)"]
    det.append(dh)
    missing = []
    r = 2
    for (tid, task, change, cols, dec, why, folder, split, db, ref, var, arm, mset) in TESTS:
        a, b = load(db, ref), load(db, var)
        metrics = REG + (CLF if mset == "both" else [])
        for model, key in metrics:
            ra = a.get(f"{model}__nopca", {}).get(key)
            rb = b.get(f"{model}__{arm or 'nopca'}", {}).get(key)
            if ra is None or rb is None:
                missing.append((tid, model, key))
                continue
            det.append([tid, folder, ref, var, model, key, round(ra, 6), round(rb, 6), None])
            det.cell(r, 9).value = f"=H{r}-G{r}"
            for c in (7, 8):
                det.cell(r, c).font = blue
            r += 1
    last = r - 1

    # ---- summary ----
    sh = ["ID", "Bài", "Thay đổi (so với bộ tham chiếu)", "Cột liên quan", "Quyết định",
          "Δ R² trung bình (regressor)", "Δ AUC trung bình (classifier)",
          "Δ AUC-PR trung bình (classifier)", "Δ nhỏ nhất", "Δ lớn nhất",
          "Lý do", "Folder test (experiments/)", "Mẫu và cách chia"]
    ws.append(sh)
    rng = f"'Chi tiết'!$A$2:$A${last}"
    met = f"'Chi tiết'!$F$2:$F${last}"
    dlt = f"'Chi tiết'!$I$2:$I${last}"
    for i, (tid, task, change, cols, dec, why, folder, split, *_rest) in enumerate(TESTS, start=2):
        ws.append([tid, task, change, cols, dec, None, None, None, None, None, why, folder, split])
        ws.cell(i, 6).value = f'=IFERROR(AVERAGEIFS({dlt},{rng},A{i},{met},"r2"),"")'
        ws.cell(i, 7).value = f'=IFERROR(AVERAGEIFS({dlt},{rng},A{i},{met},"areaUnderROC"),"")'
        ws.cell(i, 8).value = f'=IFERROR(AVERAGEIFS({dlt},{rng},A{i},{met},"areaUnderPR"),"")'
        ws.cell(i, 9).value = f"=_xlfn.MINIFS({dlt},{rng},A{i})"
        ws.cell(i, 10).value = f"=_xlfn.MAXIFS({dlt},{rng},A{i})"

    # ---- EDA ----
    eda.append(["Cột", "Bài", "Quyết định", "Căn cứ"])
    for row in EDA:
        eda.append(list(row))

    # ---- notes ----
    notes = [
        ["Chú thích"],
        ["Δ = điểm của bộ thử trừ điểm của bộ tham chiếu, tính cho từng model trên cùng mẫu, cùng cách chia."],
        ["R², AUC, AUC-PR: Δ dương là bộ thử tốt hơn, âm là tệ hơn."],
        ["Mức nhiễu: chênh lệch giữa hai bộ trên cùng seed dao động khoảng ±0.003 (đo bằng 3 seed, run H trong LOG.md). "
         "Chênh dưới khoảng 0.005 và không cùng chiều qua các model thì coi là hòa, và chọn bộ đơn giản hơn."],
        ["Mẫu: 1% dữ liệu (khoảng 57 nghìn chuyến). Kết quả Kaggle chia ngẫu nhiên trước 28/09, chia theo ngày từ 29/09; "
         "chỉ so sánh trong cùng một dòng, không so mức điểm giữa các dòng."],
        ["Model: regressor = LinearRegression, RandomForest, GBT; classifier = LinearSVC, RandomForest, GBT. "
         "Nhánh không PCA, trừ K12 (so nhánh PCA với nhánh không PCA)."],
        ["Bài: A = hồi quy trước khi bay, B = hồi quy sau cất cánh (label_gain), C = phân loại trễ trên 30 phút."],
        ["Chữ xanh trong sheet Chi tiết là số lấy từ MLflow; cột Δ và các cột trung bình ở sheet Tổng hợp là công thức."],
        ["Chi tiết từng lượt test: experiments/LOG.md. Bảng tổng: experiments/FEATURE_DECISIONS.md."],
    ]
    for n in notes:
        note.append(n)

    # ---- styling ----
    for sheet, widths in ((ws, [7, 7, 38, 34, 16, 14, 14, 14, 11, 11, 52, 32, 26]),
                          (det, [7, 32, 26, 26, 26, 14, 12, 12, 16]),
                          (eda, [48, 10, 12, 70]), (note, [130])):
        for i, w in enumerate(widths, start=1):
            sheet.column_dimensions[get_column_letter(i)].width = w
        for row in sheet.iter_rows():
            for c in row:
                if c.font != blue:
                    c.font = base
                c.alignment = wrap
                if sheet is not note:
                    c.border = border
        if sheet is not note:
            for c in sheet[1]:
                c.font = head_font
                c.fill = head_fill
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    note["A1"].font = Font(name="Arial", bold=True, size=12)
    for rr in range(2, ws.max_row + 1):
        for c in range(6, 11):
            ws.cell(rr, c).number_format = "+0.0000;-0.0000;0.0000"
    for rr in range(2, last + 1):
        for c in (7, 8):
            det.cell(rr, c).number_format = "0.0000"
        det.cell(rr, 9).number_format = "+0.0000;-0.0000;0.0000"

    wb.save(OUT)
    print(f"wrote {OUT}: {len(TESTS)} tests, {last - 1} detail rows")
    if missing:
        print("missing (not in the store):", missing)


if __name__ == "__main__":
    main()
