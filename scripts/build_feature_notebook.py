#!/usr/bin/env python
"""Write and execute notebooks/feature_distributions.ipynb.

The distribution of every feature the three tasks use, computed on the full curated
table exactly as the pipeline derives them, with a summary table that flags what to
look at (missing values, impossible negatives, zero-inflation, long tails).

    python scripts/build_feature_notebook.py
"""
from __future__ import annotations

import os

import nbformat
from nbclient import NotebookClient

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "notebooks", "feature_distributions.ipynb")

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md("""# Phân phối của mọi feature

Notebook này vẽ phân phối của **mọi feature mà 3 bài đang dùng**, tính trên toàn bộ bảng
`flights_curated` (5,704,000 chuyến), theo đúng cách pipeline tính chúng:

- **Bài A** (hồi quy trước khi bay) và **bài C** (phân loại): `predeparture_v3`
- **Bài B** (hồi quy sau cất cánh, target `label_gain`): `wheelsoff_gain_nodep`

Bảng tóm tắt ở mục 2 tự gắn cờ những chỗ cần xem: giá trị thiếu, số âm ở cột không thể âm,
cột gần như toàn số 0, đuôi dài. Tính lại bằng `python scripts/build_feature_notebook.py`."""),
    md("## 1. Tính các feature"),
    code("""import os, sys
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt

REPO = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, REPO)
from model_explain import _style, BLUE, INK, INK_2, SURFACE
%matplotlib inline

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
plt.rcParams["figure.dpi"] = 110

cols = ["MONTH", "DAY", "TAIL_NUMBER", "FLIGHT_NUMBER", "AIRLINE", "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT", "SCHED_DEP_MIN", "SCHED_ARR_MIN", "ORIGIN_LAT", "ORIGIN_LON",
        "DEST_LAT", "DEST_LON", "DISTANCE", "SCHEDULED_TIME", "TAXI_OUT", "DEPARTURE_DELAY",
        "label_delay", "label_severe"]
d = ds.dataset(os.path.join(REPO, "data", "parquet", "flights_curated")).to_table(columns=cols).to_pandas()
print(f"{len(d):,} chuyến")"""),
    code("""# Vòng quay tàu bay: chuyến trước cùng TAIL_NUMBER trong ngày, chỉ tính khi nối đúng sân bay
# (giống mllib_pipeline.rotation_table).
d = d.sort_values(["TAIL_NUMBER", "MONTH", "DAY", "SCHED_DEP_MIN", "FLIGHT_NUMBER"]).reset_index(drop=True)
g = d.groupby(["TAIL_NUMBER", "MONTH", "DAY"], sort=False)
d["leg_of_day"] = g.cumcount() + 1.0
linked = g["DESTINATION_AIRPORT"].shift() == d["ORIGIN_AIRPORT"]
d["has_prev"] = linked.astype(float)
d["turn_slack"] = np.where(linked, d["SCHED_DEP_MIN"] - g["SCHED_ARR_MIN"].shift(), np.nan)
d["prev_arr_delay"] = np.where(linked, g["label_delay"].shift(), np.nan)
d["inbound_overrun"] = np.maximum(d["prev_arr_delay"] - d["turn_slack"], 0)

# Giờ dạng vòng tròn, quãng đường, tốc độ theo lịch, target của bài B.
d["hour_sin"] = np.sin(2 * np.pi * d["SCHED_DEP_MIN"] / 1440)
d["hour_cos"] = np.cos(2 * np.pi * d["SCHED_DEP_MIN"] / 1440)
d["log_distance"] = np.log1p(d["DISTANCE"])
d["sched_mph"] = d["DISTANCE"] / (d["SCHEDULED_TIME"] / 60)
d["label_gain"] = d["label_delay"] - d["DEPARTURE_DELAY"]

# sched_dest_hour: số chuyến dự kiến hạ cánh ở sân bay đích trong giờ đó, đếm từ lịch gốc
# (kể cả chuyến bị hủy), mã sân bay tháng 10 đã sửa bằng data/dot_to_iata.csv.
raw = ds.dataset(os.path.join(REPO, "data", "parquet", "flights_raw"), partitioning="hive").to_table(
    columns=["MONTH", "DAY", "DESTINATION_AIRPORT", "SCHEDULED_ARRIVAL"]).to_pandas()
code_map = pd.read_csv(os.path.join(REPO, "data", "dot_to_iata.csv"), dtype=str)
raw["DESTINATION_AIRPORT"] = raw["DESTINATION_AIRPORT"].replace(
    dict(zip(code_map["dot_code"], code_map["iata_code"])))
raw["MONTH"] = raw["MONTH"].astype(int)
raw["_h"] = (pd.to_numeric(raw["SCHEDULED_ARRIVAL"]) // 100 % 24).astype(int)
cnt = raw.groupby(["DESTINATION_AIRPORT", "MONTH", "DAY", "_h"]).size().rename("_n").reset_index()
d["_h"] = (d["SCHED_ARR_MIN"] // 60).astype(int)
d = d.merge(cnt, on=["DESTINATION_AIRPORT", "MONTH", "DAY", "_h"], how="left")
d["sched_dest_hour"] = np.log1p(d["_n"])
del raw, cnt

A_C = ["SCHED_ARR_MIN", "hour_sin", "hour_cos", "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON",
       "leg_of_day", "has_prev", "turn_slack", "prev_arr_delay", "inbound_overrun"]
B = ["TAXI_OUT", "log_distance", "sched_mph", "DISTANCE", "SCHEDULED_TIME", "sched_dest_hour",
     "SCHED_ARR_MIN", "hour_sin", "hour_cos", "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON"]
TARGETS = ["label_delay", "label_severe", "label_gain"]
print("A và C:", len(A_C), "cột số; B:", len(B), "cột số")"""),
    md("""## 2. Bảng tóm tắt, có gắn cờ

Cột `cờ` được tính tự động:
- **thiếu**: hơn 1% ô trống (pipeline điền median của tập train)
- **âm**: có giá trị âm ở cột mà theo nghĩa không thể âm
- **toàn 0**: hơn 80% là 0
- **đuôi dài**: giá trị lớn nhất gấp hơn 5 lần phân vị 99
- **hằng số**: độ lệch chuẩn gần 0"""),
    code("""NONNEG = {"leg_of_day", "turn_slack", "inbound_overrun", "TAXI_OUT", "DISTANCE", "SCHEDULED_TIME",
          "sched_mph", "log_distance", "sched_dest_hour", "SCHED_ARR_MIN", "has_prev"}

def summary(cols):
    rows = []
    for c in cols:
        x = d[c].astype(float)
        v = x.dropna()
        q = v.quantile([0.01, 0.5, 0.99])
        flags = []
        if x.isna().mean() > 0.01: flags.append("thiếu")
        if c in NONNEG and (v < 0).any(): flags.append(f"âm ({(v < 0).mean():.2%})")
        if (v == 0).mean() > 0.8: flags.append("toàn 0")
        if q[0.99] > 0 and v.max() > 5 * q[0.99]: flags.append("đuôi dài")
        if v.std() < 1e-9: flags.append("hằng số")
        rows.append({"feature": c, "thiếu": f"{x.isna().mean():.1%}", "bằng 0": f"{(v == 0).mean():.1%}",
                     "min": v.min(), "p1": q[0.01], "trung vị": q[0.5], "p99": q[0.99], "max": v.max(),
                     "TB": v.mean(), "lệch chuẩn": v.std(), "độ lệch (skew)": v.skew(),
                     "cờ": ", ".join(flags)})
    return pd.DataFrame(rows).set_index("feature").round(3)

print("Bài A và C")
display(summary(A_C))
print("Bài B (chỉ các cột khác bài A và C)")
display(summary([c for c in B if c not in A_C]))
print("Target")
display(summary(TARGETS))"""),
    md("""## 3. Histogram từng cột số

Mỗi hình vẽ phần giữa phân vị 0.1 và 99.9. Phần nằm ngoài không bị dồn về mép (sẽ tạo đỉnh
giả) mà được bỏ khỏi hình; tiêu đề ghi giá trị lớn nhất và tỷ lệ bị bỏ. Cột có đuôi dài dùng
trục đếm thang log. Đường nét đứt là trung vị."""),
    code("""LOGY = {"turn_slack", "prev_arr_delay", "inbound_overrun", "TAXI_OUT", "sched_mph", "leg_of_day",
        "label_delay", "label_gain"}

def grid(cols, title):
    n = len(cols); ncol = 3; nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.9 * nrow))
    for ax, c in zip(axes.flat, cols):
        v = d[c].dropna().astype(float)
        lo, hi = v.quantile(0.001), v.quantile(0.999)
        if c in ("has_prev", "label_severe"):
            ax.bar([0, 1], [(v == 0).sum(), (v == 1).sum()], color=BLUE, width=0.6)
            ax.set_xticks([0, 1])
        else:
            inside = v[(v >= lo) & (v <= hi)]
            ax.hist(inside, bins=60, color=BLUE, alpha=0.9)
            ax.axvline(v.median(), color=INK_2, linestyle="--", linewidth=1)
        if c in LOGY:
            ax.set_yscale("log")
        miss = d[c].isna().mean()
        cut = 1 - ((v >= lo) & (v <= hi)).mean()
        notes = []
        if c in LOGY:
            notes.append(f"max {v.max():,.0f}, ngoài hình {cut:.1%}")
        if miss > 0.01:
            notes.append(f"thiếu {miss:.1%}")
        _style(ax, "", "số chuyến" + (" (log)" if c in LOGY else ""),
               c)
        if notes:
            ax.text(0.99, 0.97, "; ".join(notes), transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=INK_2)
        ax.ticklabel_format(axis="y", style="plain") if c not in LOGY else None
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}")) if c not in LOGY else None
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    plt.show()

grid(A_C, "Bài A và C: 12 cột số của predeparture_v3")"""),
    code("""grid([c for c in B if c not in A_C], "Bài B: các cột số riêng của wheelsoff_gain_nodep")"""),
    code("""grid(TARGETS, "Target của 3 bài")"""),
    md("## 4. Cột phân loại (one-hot)"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
a = d["AIRLINE"].value_counts()
axes[0].bar(a.index, a.values, color=BLUE)
_style(axes[0], "", "số chuyến", "AIRLINE (14 hãng)")
m = d["MONTH"].value_counts().sort_index()
axes[1].bar(m.index.astype(str), m.values, color=BLUE)
_style(axes[1], "", "số chuyến", "MONTH")
fig.tight_layout(); plt.show()
print("Hãng ít chuyến nhất:", a.idxmin(), f"{a.min():,}", f"({a.min() / len(d):.1%})")"""),
    md("## 5. Soi kỹ những cột bị gắn cờ"),
    code("""s = d["turn_slack"]
print(f"turn_slack âm: {(s < 0).sum():,} chuyến ({(s < 0).mean():.2%} số chuyến có chuyến trước)")
print("  phân vị âm:", s[s < 0].quantile([0.01, 0.5, 0.99]).round(0).to_dict())
print(f"turn_slack trên 12 tiếng: {(s > 720).sum():,}")
sp = d["sched_mph"]
print(f"\\nsched_mph trên 550 mph: {(sp > 550).sum():,}; dưới 60 mph: {(sp < 60).sum():,}")
display(d.loc[sp > 550, ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DISTANCE", "SCHEDULED_TIME", "sched_mph"]].head(12))
print(f"\\nsched_dest_hour thiếu (không khớp lịch gốc): {d['sched_dest_hour'].isna().sum():,}")
print(f"inbound_overrun > 300 phút: {(d['inbound_overrun'] > 300).sum():,}")"""),
    code("""# label_gain cực trị: bù lại hơn 1 tiếng trên không, hoặc mất thêm hơn 3 tiếng.
lg = d["label_gain"]
print(f"label_gain < -60: {(lg < -60).sum():,}   < -120: {(lg < -120).sum():,}   > 180: {(lg > 180).sum():,}")
show = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_TIME", "DISTANCE",
        "DEPARTURE_DELAY", "label_delay", "TAXI_OUT", "label_gain"]
display(d.loc[lg < -120, show].sort_values("label_gain").head(10))
# Tốc độ theo lịch dưới 60 mph: quãng đường rất ngắn so với thời gian lịch.
display(d.loc[d["sched_mph"] < 60, show[:5] + ["sched_mph"]].head(10))"""),
]

cells.append(md("""## 6. Bài B: vì sao bỏ `DEPARTURE_DELAY` mà giữ `TAXI_OUT`

`label_gain` là số phút bù lại (âm) hoặc mất thêm (dương) sau khi rời cổng. Hình trái và hình
phải dùng chung trục dọc. Đường đậm là trung vị; dải đậm chứa 50% số chuyến ở giữa (25% đến
75%), dải nhạt chứa 80% (10% đến 90%).

- **`TAXI_OUT` (trái):** cả dải đi lên. Lăn lâu hơn thì gain *điển hình* lớn hơn, khoảng một
  phút theo mỗi phút lăn. Cột này dịch chuyển giá trị kỳ vọng, nên model dùng được.
- **`DEPARTURE_DELAY` (phải):** trung vị nằm ngang ở khoảng −6 phút dù rời cổng sớm hay trễ
  10 tiếng; chỉ có dải là phình ra. Cột này không đổi giá trị kỳ vọng, chỉ đổi độ bất định,
  nên một model đoán một con số (tối ưu sai số bình phương) không dùng được nó."""))
cells.append(code("""def fan(ax, x, y, edges, xlabel, title, min_n=300):
    b = pd.cut(x, edges)
    g = y.groupby(b, observed=True)
    q = g.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    n = g.size()
    q = q[n >= min_n]
    mids = [(iv.left + iv.right) / 2 for iv in q.index]
    ax.fill_between(mids, q[0.1], q[0.9], color=BLUE, alpha=0.18, linewidth=0, label="10% đến 90% số chuyến")
    ax.fill_between(mids, q[0.25], q[0.75], color=BLUE, alpha=0.40, linewidth=0, label="25% đến 75%")
    ax.plot(mids, q[0.5], color=BLUE, linewidth=2, marker="o", markersize=4, label="trung vị")
    ax.axhline(0, color=INK_2, linewidth=1, linestyle="--")
    _style(ax, xlabel, "label_gain (phút)", title)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
fan(axes[0], d["TAXI_OUT"], d["label_gain"],
    [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 180],
    "TAXI_OUT (phút)", "Theo TAXI_OUT: cả dải đi lên")
fan(axes[1], d["DEPARTURE_DELAY"], d["label_gain"],
    [-30, -10, 0, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600],
    "DEPARTURE_DELAY (phút)", "Theo DEPARTURE_DELAY: trung vị nằm ngang, dải phình ra")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); plt.show()"""))
cells.append(code("""bands = [("rời cổng sớm hoặc đúng giờ", d["DEPARTURE_DELAY"] <= 0),
         ("trễ 30 đến 60 phút", d["DEPARTURE_DELAY"].between(31, 60)),
         ("trễ 3 đến 5 tiếng", d["DEPARTURE_DELAY"].between(181, 300))]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True, sharey=True)
for ax, (name, mask) in zip(axes, bands):
    v = d.loc[mask, "label_gain"]
    ax.hist(v[(v >= -60) & (v <= 80)], bins=70, density=True, color=BLUE, alpha=0.9)
    ax.axvline(v.median(), color=INK_2, linestyle="--", linewidth=1)
    ax.text(0.98, 0.95, f"{len(v):,} chuyến\\ntrung vị {v.median():.0f} phút\\nđộ lệch chuẩn {v.std():.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK_2)
    _style(ax, "label_gain (phút)", "tỷ lệ", name)
fig.suptitle("Cùng tâm, khác độ rộng: gain ở ba mức trễ lúc rời cổng", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(); plt.show()"""))

cells.append(md("""## 7. Kết luận: có vấn đề gì không

**Không có lỗi dữ liệu nào đáng sửa.** Những gì bị gắn cờ đều là thiết kế, sự kiện có thật,
hoặc lỗi lẻ tẻ quá ít để ảnh hưởng model.

| cột | bị gắn cờ | đánh giá |
|---|---|---|
| `turn_slack`, `prev_arr_delay`, `inbound_overrun` | thiếu 24.6% | **thiết kế**: chuyến đầu ngày không có chuyến trước. Pipeline điền median của train (lần lượt 46, −5, 0 phút) và `has_prev` = 0 báo cho model đó là giá trị điền |
| `turn_slack` | âm ở 0.72% (40,910 chuyến), thấp nhất −523 | **có thật**: theo lịch chuyến trước đến sau giờ chuyến này đi, tức hãng đổi máy bay. Giữ nguyên, `inbound_overrun` khi đó lớn hơn `prev_arr_delay`, đúng thực tế |
| `turn_slack` | trên 12 tiếng: 7,193 chuyến | máy bay đỗ cả ngày giữa hai chuyến, hợp lý; quá ít để ảnh hưởng |
| `inbound_overrun` | 93.8% bằng 0 | **thiết kế**: phần lớn chuyến trước về kịp. Đây là dạng bản lề, model cây cắt được |
| `prev_arr_delay`, `inbound_overrun`, `TAXI_OUT`, các target | đuôi dài (tối đa 1,360 đến 1,971 phút) | **có thật** (bão, dừng cất cánh). Không cắt IQR; trần chỉ đặt ở điểm bão hòa hoặc điểm gãy đo được (đang test cap60, cap300) |
| `sched_mph` | 12 chuyến trên 550 mph | **lỗi dữ liệu**: 10 của Alaska, 1 của ExpressJet (RST đến CLE 578 mph); HNL đến ATL 550 mph là thật. 12 trên 5.7 triệu, không đáng xử lý |
| `sched_mph` | 56 chuyến dưới 60 mph | **có thật**: MKE đến ORD 67 dặm, lịch 68 phút, chủ yếu là lăn và chờ |
| `label_gain` | thấp nhất −201 phút | **1 lỗi dữ liệu**: VX SAN đến SFO ghi lịch 285 phút (bình thường khoảng 90). Chỉ 278 chuyến bù hơn 1 tiếng |
| tọa độ | cụm ở vĩ độ 13 đến 21 và 60 đến 71, kinh độ −176 | **có thật**: Guam, Hawaii, Alaska |
| `hour_sin`, `hour_cos` | dồn về ±1 | tính chất của hàm sin, không phải lỗi |
| `AIRLINE`, `MONTH` | VX ít nhất (1.1%), tháng 2 ít nhất | có thật (hãng nhỏ, tháng 2 có 28 ngày) |
"""))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": os.path.join(REPO, "notebooks")}}).execute()
nbformat.write(nb, OUT)
print("wrote", OUT)
