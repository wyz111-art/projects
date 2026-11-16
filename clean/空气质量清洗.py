#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国城市空气质量（江苏13市）数据处理与清洗管道
=================================================

输入: 每天一个宽表 CSV(china_cities_YYYYMMDD.csv)，形如:
    date,hour,type,北京,天津,...,南京,...,宿迁
其中 type 含 15 种指标，带 `_24h` 后缀者为 24h 滑动平均浓度。
城市按列排放。

输出(写入 output/ 目录):
    1. jiangsu_daily_YYYYMMDD.csv   —— 天级(每日)聚合, 宽表, 一城市一行/一天
    2. jiangsu_daily_all.csv        —— 所有天拼接后的天级表
    3. jiangsu_monthly.csv          —— 月度聚合(城市×月)
    4. jiangsu_yearly.csv           —— 年度聚合(城市×年)

聚合口径说明:
    - 天级: 对每个带 `_24h` 后缀的指标, 取当天 hour==23 的那行值作为该指标
      的"当日日均值"(这也是用户指定的口径)。瞬时指标取当日逐小时均值。
    - 月/年: 在日级结果上按城市+月/年求均值。

数据清洗规则(逐条可开关):
    1. 宽表->长表(melt): 把每个城市展开为一行, 便于处理。
    2. 缺测: 整行为空的数值视为 NaN, 不参与聚合。
    3. 负值/零值异常: 浓度值 <=0 记为缺失(CO 除外可单独配置)。
    4. 去重: 同 (date, hour, type, city) 出现多行时保留最后一条。
    5. 日级取值缺失兜底: 若某城市当天 hour==23 的某 24h 指标缺失,
       退化为该指标当天所有小时的均值(非 24h 指标退化为当日小时均值)。
"""

import glob
import os
import argparse
import numpy as np
import pandas as pd

# ----------------------------- 常量配置 ----------------------------- #
JIANGSU_CITIES = [
    "南京", "苏州", "无锡", "常州", "徐州",
    "南通", "连云港", "淮安", "盐城", "扬州",
    "镇江", "泰州", "宿迁",
]

# 数据集里实际存在的 type(仅作说明; 实际聚合只用 DAILY_24H_TYPES + 瞬时指标)。

ALL_TYPES = [
    "PM2.5", "PM2.5_24h", "PM10", "PM10_24h",
    "SO2", "SO2_24h", "NO2", "NO2_24h",
    "O3", "O3_24h", "O3_8h", "O3_8h_24h", "CO", "CO_24h",
]

# 日值(带 _24h)指标 —— 天级聚合时取 hour==23 的那行
DAILY_24H_TYPES = ["PM2.5_24h", "PM10_24h", "SO2_24h", "NO2_24h", "O3_24h", "O3_8h_24h", "CO_24h"]
# 其他瞬时指标(PM2.5/PM10/SO2/NO2/O3/O3_8h/CO)保留当日小时均值作参考列


def load_one_file(path: str) -> pd.DataFrame:
    """读取单个宽表文件 -> 返回长表(已去重)。"""
    raw = pd.read_csv(path, dtype={"date": str})
    meta_cols = ["date", "hour", "type"]
    city_cols = [c for c in raw.columns if c not in meta_cols]

    # 1) 宽表 -> 长表
    long = raw.melt(id_vars=meta_cols, value_vars=city_cols,
                    var_name="city", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")  # 无法转数 -> NaN

    # 2) 去重: 同 (date,hour,type,city) 保留最后一条
    long = long.drop_duplicates(subset=["date", "hour", "type", "city"], keep="last")

    return long.reset_index(drop=True)


def clean_dataframe(long: pd.DataFrame, drop_nonpositive: bool = True) -> pd.DataFrame:
    """数据清洗: 异常值 -> NaN。"""
    df = long.copy()

    # 3) 负值/零值 -> NaN(CO 浓度 mg/m3 可为小数, 但仍应 >0)
    if drop_nonpositive:
        mask = (df["value"] <= 0) | (~np.isfinite(df["value"]))
        df.loc[mask, "value"] = np.nan

    return df


def daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """天级聚合: 每个 (date, city) 一行, 各指标一列。

    口径:
      - _24h 指标: 取当天 hour==23 的值(缺失则退化为当天该指标均值)
      - 瞬时指标: 取当天均值(参考列)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    rows = []
    for (date, city), grp in df.groupby(["date", "city"]):
        rec = {"date": date, "city": city}

        # --- 各 _24h 指标: 取 hour==23 ---
        for t in DAILY_24H_TYPES:
            sub = grp.loc[grp["type"] == t, :]
            if sub.empty:
                rec[t] = np.nan
                continue
            # 优先 hour==23; 缺失则退化为当日该指标均值。统一保留两位小数。
            h23 = sub.loc[sub["hour"] == 23, "value"]
            if h23.notna().any():
                rec[t] = round(float(h23.dropna().iloc[0]), 2)
            else:
                rec[t] = round(float(sub["value"].mean()), 2)

        # --- 瞬时指标均值(参考), 保留两位小数 ---
        for t in ["PM2.5", "PM10", "SO2", "NO2", "O3", "O3_8h", "CO"]:
            sub = grp.loc[grp["type"] == t, "value"]
            rec[t] = round(sub.mean(), 2) if not sub.empty else np.nan

        rows.append(rec)

    daily = pd.DataFrame(rows)
    # 列顺序: 日期 城市 24h指标 瞬时指标
    order = ["date", "city"] + DAILY_24H_TYPES + ["PM2.5", "PM10", "SO2", "NO2", "O3", "O3_8h", "CO"]
    daily = daily[order].sort_values(["date", "city"]).reset_index(drop=True)
    return daily


def monthly_aggregate(daily: pd.DataFrame) -> pd.DataFrame:
    """月度聚合: 城市 × 年月, 各指标均值(可配置 agg)。"""
    df = daily.copy()
    df["month"] = df["date"].dt.to_period("M")
    cols = [c for c in df.columns if c not in ("date", "city")]
    m = df.groupby(["city", "month"], as_index=False)[cols].mean().round(2)
    m = m.sort_values(["month", "city"]).reset_index(drop=True)
    return m


def yearly_aggregate(daily: pd.DataFrame) -> pd.DataFrame:
    """年度聚合: 城市 × 年, 各指标均值。"""
    df = daily.copy()
    df["year"] = df["date"].dt.year
    cols = [c for c in df.columns if c not in ("date", "city")]
    y = df.groupby(["city", "year"], as_index=False)[cols].mean().round(2)
    y = y.sort_values(["year", "city"]).reset_index(drop=True)
    return y


def process_folder(input_dir: str, output_dir: str, only_jiangsu: bool = True):
    """处理一个文件夹下的所有每日 CSV(拼接后统一清洗聚合)。"""
    files = sorted(glob.glob(os.path.join(input_dir, "china_cities_*.csv")))
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 下未找到 china_cities_*.csv")

    frames = []
    for f in files:
        print(f"[read] {os.path.basename(f)}")
        frames.append(load_one_file(f))
    long = pd.concat(frames, ignore_index=True)

    # 仅保留江苏城市
    if only_jiangsu:
        long = long[long["city"].isin(JIANGSU_CITIES)]
        long = long.reset_index(drop=True)

    long = clean_dataframe(long)

    # 天级
    daily = daily_aggregate(long)
    daily.to_csv(os.path.join(output_dir, "jiangsu_daily_all.csv"),
                 index=False, encoding="utf-8-sig")

    # 月级
    monthly_aggregate(daily).to_csv(os.path.join(output_dir, "jiangsu_monthly.csv"),
                                    index=False, encoding="utf-8-sig")

    # 年(季)级
    yearly_aggregate(daily).to_csv(os.path.join(output_dir, "jiangsu_yearly.csv"),
                                   index=False, encoding="utf-8-sig")

    print(f"\n[done] 共处理 {len(files)} 个文件")
    print(f"  天级表: {os.path.join(output_dir, 'jiangsu_daily_all.csv')}")
    print(f"  月级表: {os.path.join(output_dir, 'jiangsu_monthly.csv')}")
    print(f"  年(季)级表: {os.path.join(output_dir, 'jiangsu_yearly.csv')}")
    return daily


def process_one_file(path: str, output_dir: str, only_jiangsu: bool = True) -> pd.DataFrame:
    """处理单个文件(用于测试/单日)。"""
    long = load_one_file(path)
    if only_jiangsu:
        long = long[long["city"].isin(JIANGSU_CITIES)].reset_index(drop=True)
    long = clean_dataframe(long)

    daily = daily_aggregate(long)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"jiangsu_daily_{base}.csv")
    daily.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[done] 单日表 -> {out_path}")
    return daily


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="江苏空气质量数据清洗与天/月/年聚合")
    ap.add_argument("input", nargs="?", default=SCRIPT_DIR,
                    help="输入: 单个CSV或包含 china_cities_*.csv 的文件夹(默认=脚本所在目录)")
    ap.add_argument("-o", "--output", default=os.path.join(SCRIPT_DIR, "output"),
                    help="输出目录(默认=脚本目录下 output/)")
    ap.add_argument("--all-cities", action="store_true",
                    help="默认仅处理江苏13市; 加此参数保留全部城市")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    in_path = args.input
    if os.path.isdir(in_path):
        process_folder(in_path, args.output, only_jiangsu=not args.all_cities)
    else:
        process_one_file(in_path, args.output, only_jiangsu=not args.all_cities)
