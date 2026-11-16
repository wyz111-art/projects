# -*- coding: utf-8 -*-
"""ERA5 江苏13市气象数据：按空气质量表格式输出年度/日度/月度 CSV。

放到项目 scripts 目录，运行：python scripts/clean_era5_like_air_quality.py
扫描 data/raw/era5 下ZIP、NetCDF、GRIB，不改动原始文件。
输出到 data/cleaned/era5/江苏气象{日度,月度,年度}数据/，各年份独立。
旧的 data/cleaned/era5 仅在所有原始文件成功处理并验证后替换。
依赖：numpy pandas xarray netCDF4 shapely requests cfgrib eccodes
"""
from __future__ import annotations

import argparse
import calendar
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import Point, shape

# Official-prefecture adcodes for Jiangsu's 13 prefecture-level cities.
CITIES = {
    "320100": "南京市", "320200": "无锡市", "320300": "徐州市",
    "320400": "常州市", "320500": "苏州市", "320600": "南通市",
    "320700": "连云港市", "320800": "淮安市", "320900": "盐城市",
    "321000": "扬州市", "321100": "镇江市", "321200": "泰州市",
    "321300": "宿迁市",
}
BOUNDARY_URL = "https://geo.datav.aliyun.com/areas_v3/bound/320000_full.json"
VARS = ("t2m", "u10", "v10", "tp")


def read_boundaries(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("首次运行：下载江苏省13市行政边界……", flush=True)
        try:
            r = requests.get(BOUNDARY_URL, timeout=45)
            r.raise_for_status()
            geo = r.json()
            # Validate before saving a downloaded file.
            validate_boundaries(geo)
            path.write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(
                "行政边界下载失败。请用浏览器打开：\n"
                f"{BOUNDARY_URL}\n"
                f"将网页保存为：{path}\n"
                "然后重新运行脚本。原清洗文件未删除。\n"
                f"网络错误：{exc}"
            ) from exc
    try:
        geo = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"读取行政边界失败：{path}：{exc}") from exc
    return validate_boundaries(geo)


def validate_boundaries(geo: dict):
    if geo.get("type") != "FeatureCollection":
        raise ValueError("行政区文件不是GeoJSON FeatureCollection")
    out = {}
    for feat in geo.get("features", []):
        props = feat.get("properties") or {}
        code = str(props.get("adcode", ""))
        if code in CITIES:
            geom = shape(feat["geometry"])
            if geom.is_empty:
                raise ValueError(f"行政区{code}几何为空")
            out[code] = geom
    if set(out) != set(CITIES):
        missing = [CITIES[c] for c in CITIES if c not in out]
        raise ValueError(f"行政区边界不是江苏完整的13市数据；缺少：{missing}")
    return out


# Recognize common GRIB extensions; when ZIP members have no suffix, use magic bytes.
GRIB_EXTENSIONS = {".grib", ".grb", ".grib2", ".grb2"}


def is_grib(path: Path) -> bool:
    if path.suffix.lower() in GRIB_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            return f.read(4) == b"GRIB"
    except OSError:
        return False


def normalize_array(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Normalize both NetCDF and cfgrib variables to (valid UTC time, latitude, longitude)."""
    da = ds[var].squeeze(drop=True)
    replacements = {}
    if "lat" in da.dims:
        replacements["lat"] = "latitude"
    if "lon" in da.dims:
        replacements["lon"] = "longitude"
    if "valid_time" in da.dims:
        replacements["valid_time"] = "time"
    if replacements:
        da = da.rename(replacements)
    if not {"latitude", "longitude"}.issubset(da.dims):
        raise ValueError(f"{var}缺少经纬度维度: {da.dims}")
    other_dims = [d for d in da.dims if d not in ("latitude", "longitude")]
    if not other_dims or not set(other_dims).issubset({"time", "step"}):
        raise ValueError(f"{var}未知时间维度: {da.dims}")

    # GRIB forecast 'time' is initialization time. Its 'valid_time' is the
    # meteorological observation/forecast VALID time, which we must use.
    if "valid_time" in da.coords:
        valid = da.coords["valid_time"]
        if not set(valid.dims).issubset(set(other_dims)):
            raise ValueError(f"{var} GRIB valid_time维度异常: {valid.dims}")
        valid = valid.broadcast_like(da)
        first = [d for d in other_dims]
        stamps = np.asarray(valid.transpose(*first, "latitude", "longitude").values)
        # Because valid time does not vary spatially, get the first grid cell.
        stamps = stamps[..., 0, 0].reshape(-1)
    else:
        if other_dims != ["time"] or "time" not in da.coords:
            raise ValueError(f"{var}无法找到有效时间，不能把GRIB起报时间当成数据时间")
        stamps = np.asarray(da["time"].values).reshape(-1)

    data = np.asarray(da.transpose(*other_dims, "latitude", "longitude").values)
    data = data.reshape(-1, da.sizes["latitude"], da.sizes["longitude"])
    timestamps = pd.to_datetime(stamps, utc=True, errors="coerce")
    if pd.isna(timestamps).any():
        raise ValueError(f"{var}时间字段有无效值")
    naive = timestamps.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    order = np.argsort(naive, kind="stable")
    naive = naive[order]
    data = data[order]
    # Some GRIB streams contain overlapping forecast validity hours;
    # keep only one for each valid hour instead of double-counting rainfall.
    unique = ~pd.Index(naive).duplicated(keep="first")
    naive = naive[unique]
    data = data[unique]
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={"time": naive,
                "latitude": da["latitude"].values,
                "longitude": da["longitude"].values},
        name=var,
    )


def load_one_batch(paths: list[Path]):
    opened = []
    try:
        for path in paths:
            if is_grib(path):
                try:
                    import cfgrib
                except ImportError as exc:
                    raise RuntimeError(
                        "检测到GRIB，但缺少读取依赖。请先运行："
                        "python -m pip install cfgrib eccodes；然后 python -m cfgrib selfcheck"
                    ) from exc
                try:
                    groups = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
                except Exception as exc:
                    raise RuntimeError(f"GRIB解码失败({path.name})，请检查cfgrib/ecCodes安装：{exc}") from exc
                opened.extend(groups)
            else:
                opened.append(xr.open_dataset(path))
        arrays = {}
        for var in VARS:
            # 一个目录中可能有多个文件、多个自然月；旧版只取第一份，会漏月份。
            pieces = [normalize_array(ds, var) for ds in opened if var in ds.data_vars]
            if not pieces:
                continue
            if len(pieces) == 1:
                combined = pieces[0]
            else:
                combined = xr.concat(pieces, dim="time", join="outer", coords="minimal", compat="override")
                combined = combined.sortby("time")
                stamps = pd.Index(combined["time"].values)
                combined = combined.isel(time=np.flatnonzero(~stamps.duplicated(keep="first")))
            arrays[var] = combined
        missing = [v for v in VARS if v not in arrays]
        if missing:
            raise ValueError(f"缺少变量：{','.join(missing)}（必须同一批包含4项变量）")
        merged = xr.merge([arrays[v].to_dataset(name=v) for v in VARS], join="outer")
        times = pd.to_datetime(merged.time.values, utc=True, errors="coerce")
        if pd.isna(times).any():
            raise ValueError("时间字段包含无效值")
        lat = merged.latitude.values.astype("float64")
        lon = merged.longitude.values.astype("float64")
        vals = {v: np.asarray(merged[v].values, dtype="float64") for v in VARS}
        return times, lat, lon, vals
    finally:
        for ds in opened:
            ds.close()


def jobs_under(raw: Path):
    zips = sorted(raw.rglob("*.zip"))
    # 不因同目录存在 ZIP 就忽略所有 NC：否则可能漏掉同目录的其他月份。
    nc_files = [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in {".nc", ".nc4"}]
    grib_files = [p for p in raw.rglob("*") if p.is_file() and
                  p.suffix.lower() not in {".zip", ".nc", ".nc4"} and is_grib(p)]
    dirs = sorted({p.parent for p in nc_files + grib_files})
    return [("zip", x) for x in zips] + [("folder", x) for x in dirs]


def mask_by_city(lat, lon, geoms):
    lons, lats = np.meshgrid(lon, lat)
    all_points = [Point(x, y) for x, y in zip(lons.ravel(), lats.ravel())]
    masks, counts = {}, {}
    for code, geom in geoms.items():
        m = np.array([geom.covers(p) for p in all_points], dtype=bool).reshape(lats.shape)
        masks[code] = m
        counts[code] = int(m.sum())
    return masks, counts


def spatial_mean(a: np.ndarray, mask: np.ndarray):
    sub = a[:, mask]
    if sub.shape[1] == 0:
        return np.full(a.shape[0], np.nan)
    count = np.isfinite(sub).sum(axis=1)
    total = np.nansum(sub, axis=1)
    out = np.full(a.shape[0], np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def extract_city_hours(nc_paths, geoms, cache, city_count_records, warnings):
    times, lat, lon, vals = load_one_batch(nc_paths)
    grid_key = (tuple(lat.tolist()), tuple(lon.tolist()))
    if grid_key not in cache:
        cache[grid_key] = mask_by_city(lat, lon, geoms)
    masks, counts = cache[grid_key]
    for code, count in counts.items():
        city_count_records[code].add(count)
    if any(n == 0 for n in counts.values()):
        names = [CITIES[c] for c, n in counts.items() if n == 0]
        raise ValueError(f"所下载范围无法覆盖13个城市的网格中心：{names}，请检查下载范围与边界")
    records = []
    speed = np.hypot(vals["u10"], vals["v10"])
    invalid_temp = 0
    invalid_wind = 0
    invalid_precip = 0
    for code, city in CITIES.items():
        mask = masks[code]
        temp = spatial_mean(vals["t2m"], mask) - 273.15
        wind = spatial_mean(speed, mask)
        rain = spatial_mean(vals["tp"], mask) * 1000.0
        bad_temp = np.isfinite(temp) & ((temp < -90) | (temp > 60))
        bad_wind = np.isfinite(wind) & (wind > 100)
        bad_rain = np.isfinite(rain) & (rain < 0)
        invalid_temp += int(bad_temp.sum())
        invalid_wind += int(bad_wind.sum())
        invalid_precip += int(bad_rain.sum())
        temp[bad_temp] = np.nan
        wind[bad_wind] = np.nan
        rain[bad_rain] = np.nan
        records.append(pd.DataFrame({
            "city_code": code,
            "city": city,
            "time_utc": times,
            "temperature_c": temp,
            "wind_speed_ms": wind,
            "precipitation_mm": rain,
            "grid_points": counts[code],
        }))
    warnings["invalid_temperature_city_hours"] += invalid_temp
    warnings["invalid_wind_city_hours"] += invalid_wind
    warnings["negative_precipitation_city_hours"] += invalid_precip
    return pd.concat(records, ignore_index=True)


def load_job(kind, path, geoms, cache, city_count_records, warnings):
    if kind == "folder":
        paths = sorted(p for p in path.iterdir() if p.is_file() and
                       (p.suffix.lower() in {".nc", ".nc4"} or is_grib(p)))
        return extract_city_hours(paths, geoms, cache, city_count_records, warnings)
    with zipfile.ZipFile(path) as z:
        failed = z.testzip()
        if failed:
            raise ValueError(f"压缩包校验失败：{failed}")
        usable = []
        for member in z.infolist():
            if member.is_dir():
                continue
            ext = Path(member.filename).suffix.lower()
            if ext in {".nc", ".nc4"} or ext in GRIB_EXTENSIONS:
                usable.append(member)
            else:
                with z.open(member) as stream:
                    if stream.read(4) == b"GRIB":
                        usable.append(member)
        if not usable:
            raise ValueError("ZIP中没有NetCDF/GRIB气象数据")
        with tempfile.TemporaryDirectory(prefix="era5_unpack_") as tmp:
            paths = []
            for index, member in enumerate(usable):
                # Always use a fixed safe basename; never extract arbitrary ZIP paths.
                ext = Path(member.filename).suffix.lower()
                if ext not in ({".nc", ".nc4"} | GRIB_EXTENSIONS):
                    ext = ".grib"
                dest = Path(tmp) / f"{index:03d}{ext}"
                with z.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                paths.append(dest)
            return extract_city_hours(paths, geoms, cache, city_count_records, warnings)


def city_daily(hourly: pd.DataFrame):
    # Validity time: instantaneous temperature/wind belong to their local timestamp.
    # ERA5 hourly precipitation belongs to the hour ENDING at its validity time.
    # Thus 00:00 precipitation belongs to the previous calendar day, not today.
    hourly = hourly.copy()
    hourly["time_beijing"] = hourly["time_utc"].dt.tz_convert("Asia/Shanghai")
    hourly["date_weather"] = hourly["time_beijing"].dt.strftime("%Y-%m-%d")
    hourly["date_precip"] = (hourly["time_beijing"] - pd.Timedelta(nanoseconds=1)).dt.strftime("%Y-%m-%d")
    g1 = hourly.groupby(["city_code", "city", "date_weather"], as_index=False).agg(
        temp_c=("temperature_c", "mean"),
        wind_ms=("wind_speed_ms", "mean"),
        temp_hours=("temperature_c", "count"),
        wind_hours=("wind_speed_ms", "count"),
        grid_points=("grid_points", "max"),
    ).rename(columns={"date_weather": "date"})
    g2 = hourly.groupby(["city_code", "city", "date_precip"], as_index=False).agg(
        precip_mm=("precipitation_mm", "sum"),
        precip_hours=("precipitation_mm", "count"),
    ).rename(columns={"date_precip": "date"})
    out = g1.merge(g2, on=["city_code", "city", "date"], how="outer")
    for count_col in ("temp_hours", "wind_hours", "precip_hours"):
        out[count_col] = out[count_col].fillna(0).astype(int)
    # Incomplete days must not be silently treated as full-day results.
    out.loc[out.temp_hours != 24, "temp_c"] = np.nan
    out.loc[out.wind_hours != 24, "wind_ms"] = np.nan
    out.loc[out.precip_hours != 24, "precip_mm"] = np.nan
    out["complete_day"] = (
        (out.temp_hours == 24) & (out.wind_hours == 24) & (out.precip_hours == 24)
    )
    out["year"] = out["date"].str[:4].astype(int)
    out["month"] = out["date"].str[:7]
    out = out.rename(columns={
        "temp_c": "日平均气温_℃", "wind_ms": "日平均风速_m每s",
        "precip_mm": "日累计降水_mm", "temp_hours": "气温有效小时数",
        "wind_hours": "风速有效小时数", "precip_hours": "降水有效小时数",
        "complete_day": "全天数据完整", "grid_points": "使用网格数",
        "city_code": "城市代码", "city": "城市", "date": "日期",
        "year": "年份", "month": "月份",
    })
    order = ["日期", "年份", "月份", "城市代码", "城市", "日平均气温_℃", "日平均风速_m每s",
             "日累计降水_mm", "气温有效小时数", "风速有效小时数", "降水有效小时数", "全天数据完整", "使用网格数"]
    return out[order].sort_values(["年份", "城市代码", "日期"]).reset_index(drop=True)


def city_monthly(daily: pd.DataFrame):
    # Incomplete days have NaN metrics and cannot silently contribute to totals.
    groups = daily.groupby(["年份", "月份", "城市代码", "城市"], as_index=False).agg(**{
        "月平均气温_℃": ("日平均气温_℃", "mean"),
        "月平均风速_m每s": ("日平均风速_m每s", "mean"),
        "已完整日降水合计_mm": ("日累计降水_mm", "sum"),
        "完整天数": ("全天数据完整", "sum"),
        "记录天数": ("日期", "nunique"),
    })
    groups["完整天数"] = groups["完整天数"].astype(int)
    groups["该月总天数"] = groups["月份"].apply(lambda x: calendar.monthrange(int(x[:4]), int(x[-2:]))[1])
    groups["整月完整"] = groups["完整天数"] == groups["该月总天数"]
    groups.loc[groups["完整天数"] == 0, ["已完整日降水合计_mm"]] = np.nan
    # Keep preliminary partial-month stats explicitly labeled; leave the true
    # full-month fields blank unless every day has 24 valid hours.
    groups["已完整日平均气温_℃"] = groups["月平均气温_℃"]
    groups["已完整日平均风速_m每s"] = groups["月平均风速_m每s"]
    groups["月累计降水_mm"] = groups["已完整日降水合计_mm"].where(groups["整月完整"])
    groups.loc[~groups["整月完整"], ["月平均气温_℃", "月平均风速_m每s"]] = np.nan
    return groups.sort_values(["年份", "城市代码", "月份"]).reset_index(drop=True)


def format_daily(daily: pd.DataFrame):
    # 与空气质量表保持相同的前两列：date、city；每日期下连续13个城市。
    out = daily.rename(columns={
        "日期": "date", "城市": "city", "城市代码": "city_code",
        "日平均气温_℃": "temp_mean_C", "日平均风速_m每s": "wind_mean_ms",
        "日累计降水_mm": "precip_mm", "气温有效小时数": "temp_hours",
        "风速有效小时数": "wind_hours", "降水有效小时数": "precip_hours",
        "全天数据完整": "complete_day", "使用网格数": "grid_points",
    }).copy()
    out["city"] = out["city"].str.removesuffix("市")
    cols = ["date", "city", "temp_mean_C", "wind_mean_ms", "precip_mm",
            "temp_hours", "wind_hours", "precip_hours", "complete_day",
            "city_code", "grid_points"]
    return out[cols].sort_values(["date", "city_code"], kind="stable").reset_index(drop=True)


def format_monthly(monthly: pd.DataFrame):
    out = monthly.rename(columns={
        "月份": "date", "城市": "city", "城市代码": "city_code",
        "月平均气温_℃": "temp_mean_C", "月平均风速_m每s": "wind_mean_ms",
        "月累计降水_mm": "precip_mm", "完整天数": "complete_days",
        "该月总天数": "expected_days", "整月完整": "complete_period",
        "已完整日平均气温_℃": "observed_temp_mean_C",
        "已完整日平均风速_m每s": "observed_wind_mean_ms",
        "已完整日降水合计_mm": "observed_precip_mm",
    }).copy()
    out["city"] = out["city"].str.removesuffix("市")
    cols = ["date", "city", "temp_mean_C", "wind_mean_ms", "precip_mm",
            "complete_days", "expected_days", "complete_period",
            "observed_temp_mean_C", "observed_wind_mean_ms", "observed_precip_mm", "city_code"]
    return out[cols].sort_values(["date", "city_code"], kind="stable").reset_index(drop=True)


def format_annual(daily: pd.DataFrame):
    """只有全年365/366天全部完整，才填写真正的全年统计值。"""
    df = daily.copy()
    df["date"] = df["年份"].astype(str)
    all_years = sorted(df["年份"].unique())
    annual = df.groupby(["年份", "城市代码", "城市"], as_index=False).agg(
        observed_temp_mean_C=("日平均气温_℃", "mean"),
        observed_wind_mean_ms=("日平均风速_m每s", "mean"),
        observed_precip_mm=("日累计降水_mm", "sum"),
        complete_days=("全天数据完整", "sum"),
    )
    annual.loc[annual["complete_days"] == 0, ["observed_temp_mean_C", "observed_wind_mean_ms", "observed_precip_mm"]] = np.nan
    annual["expected_days"] = annual["年份"].apply(
        lambda year: 366 if calendar.isleap(int(year)) else 365)
    annual["complete_period"] = annual["complete_days"] == annual["expected_days"]
    for v, ob in [("temp_mean_C", "observed_temp_mean_C"),
                  ("wind_mean_ms", "observed_wind_mean_ms"),
                  ("precip_mm", "observed_precip_mm")]:
        annual[v] = annual[ob].where(annual["complete_period"])
    annual = annual.rename(columns={"年份": "date", "城市": "city", "城市代码": "city_code"})
    annual["city"] = annual["city"].str.removesuffix("市")
    cols = ["date", "city", "temp_mean_C", "wind_mean_ms", "precip_mm",
            "complete_days", "expected_days", "complete_period",
            "observed_temp_mean_C", "observed_wind_mean_ms", "observed_precip_mm", "city_code"]
    annual["date"] = annual["date"].astype(str)
    return annual[cols].sort_values(["date", "city_code"], kind="stable").reset_index(drop=True)


def save_result(output: Path, daily: pd.DataFrame, monthly: pd.DataFrame,
                annual: pd.DataFrame, report: dict):
    output.mkdir(parents=True, exist_ok=True)
    daily = format_daily(daily)
    monthly = format_monthly(monthly)
    annual = annual.copy()
    groups = [
        ("江苏气象日度数据", "江苏气象日度", daily),
        ("江苏气象月度数据", "江苏气象月度", monthly),
        ("江苏气象年度数据", "江苏气象年度", annual),
    ]
    years = sorted({int(d[:4]) for d in daily["date"].unique()})
    report["years_written"] = years
    report["daily_rows"] = len(daily)
    report["monthly_rows"] = len(monthly)
    report["annual_rows"] = len(annual)
    report["complete_city_days"] = int(daily["complete_day"].sum())
    report["incomplete_city_days"] = int((~daily["complete_day"]).sum())
    report["outputs"] = {}
    report["time_and_aggregation"] = (
        "ERA5原始时间UTC，日/月/年按北京时间UTC+8划分。城市值是行政区内0.25度网格中心等权平均；"
        "日温度/风速为24小时均值；日降水属于结束时刻前1小时，按结束时间减1纳秒归属。"
        "日度缺小时则指标留空；月度和年度若不完整，主要指标留空，observed_*仅代表已有完整日期的临时结果。"
    )
    for folder_name, prefix, frame in groups:
        folder = output / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        report["outputs"][folder_name] = {}
        for year in years:
            part = frame[frame["date"].str[:4] == str(year)]
            if part.empty:
                continue
            filename = f"{prefix}_{year}.csv"
            part.to_csv(folder / filename, index=False, encoding="utf-8-sig", float_format="%.4f")
            report["outputs"][folder_name][str(year)] = {"rows": int(len(part)), "file": filename}
    (output / "清洗质量报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def replace_old_outputs(staged_output: Path, final_output: Path):
    """Only delete prior CLEANED ERA5 output after new work is ready."""
    final_output.parent.mkdir(parents=True, exist_ok=True)
    old_backup = final_output.parent / "_era5_previous_autoremove"
    if old_backup.exists():
        raise RuntimeError(f"检测到上次残留的备份目录，请先人工检查：{old_backup}，为安全起见不覆盖")
    had_old = final_output.exists()
    if had_old:
        final_output.rename(old_backup)
    try:
        staged_output.rename(final_output)
    except Exception:
        if had_old and old_backup.exists() and not final_output.exists():
            old_backup.rename(final_output)
        raise
    if had_old:
        shutil.rmtree(old_backup)


def run():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="重建 ERA5 江苏13市日度、月度、年度气象CSV")
    parser.add_argument("--raw", type=Path, default=root / "data/raw/era5")
    parser.add_argument("--out", type=Path, default=root / "data/cleaned/era5")
    parser.add_argument("--boundary", type=Path, default=root / "data/reference/jiangsu_13cities.geojson")
    args = parser.parse_args()
    raw, out, boundary = args.raw.resolve(), args.out.resolve(), args.boundary.resolve()
    if not raw.is_dir():
        raise SystemExit(f"找不到原始数据目录：{raw}")
    # Avoid danger of deleting a raw folder via a mistaken --out argument.
    if out == raw or out in raw.parents or raw in out.parents:
        raise SystemExit("安全检查：清洗结果目录与原始数据目录不能互相包含")
    if out.name.lower() != "era5" or out.parent.name.lower() != "cleaned":
        raise SystemExit("安全检查：清洗结果必须放在 data/cleaned/era5 结构中，禁止误删其他目录")
    geoms = read_boundaries(boundary)
    jobs = jobs_under(raw)
    if not jobs:
        raise SystemExit("原始文件夹没有 ZIP 或 .nc 文件；旧清洗结果保持不变")
    print(f"检测到 {len(jobs)} 个批次（ZIP/NetCDF/GRIB）；全部处理成功后才替换旧的 data/cleaned/era5", flush=True)
    cache, frames = {}, []
    counts = {x: set() for x in CITIES}
    flags = {"invalid_temperature_city_hours": 0, "invalid_wind_city_hours": 0, "negative_precipitation_city_hours": 0}
    status = {"processed_sources": [], "skipped_sources": [], "boundary_source": BOUNDARY_URL,
              "boundary_file": str(boundary), "gridpoint_counts_per_city_per_grid": {}}
    for kind, path in jobs:
        print(f"读取：{path}", flush=True)
        try:
            city_hours = load_job(kind, path, geoms, cache, counts, flags)
            frames.append(city_hours)
            status["processed_sources"].append(str(path))
            print(f"  完成：{city_hours['time_utc'].nunique()}小时，13个城市", flush=True)
        except Exception as exc:
            status["skipped_sources"].append({"path": str(path), "reason": str(exc)})
            print(f"  跳过：{exc}", flush=True)
    if status["skipped_sources"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        diagnostic = out.parent / "era5_需要检查的原始文件.json"
        diagnostic.write_text(json.dumps(status["skipped_sources"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n有{len(status['skipped_sources'])}个原始数据批次没能正确处理。", flush=True)
        print(f"原因已写入：{diagnostic}", flush=True)
        raise SystemExit("为防止年份或月份丢失，暂不覆盖旧清洗结果。解决报错后再运行。")
    if not frames:
        raise SystemExit("没有可用的气象数据批次，旧清洗结果没有被删除。")
    hourly = pd.concat(frames, ignore_index=True)
    n_before = len(hourly)
    # Duplicate hours from overlapping ZIPs or monthly/half-year files are counted once.
    hourly = hourly.sort_values(["city_code", "time_utc"]).drop_duplicates(["city_code", "time_utc"], keep="first")
    status["duplicate_city_hour_records_removed"] = n_before - len(hourly)
    status["invalid_value_counts"] = flags
    status["gridpoint_counts_per_city_per_grid"] = {CITIES[c]: sorted(v) for c, v in counts.items()}
    daily = city_daily(hourly)
    monthly = city_monthly(daily)
    annual = format_annual(daily)
    if daily.empty or set(daily["城市"].unique()) != set(CITIES.values()):
        raise SystemExit("未能覆盖13个城市，旧清洗结果保持不变。")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_era5_staging_", dir=out.parent) as tmp:
        staged = Path(tmp) / "era5"
        save_result(staged, daily, monthly, annual, status)
        replace_old_outputs(staged, out)
    print("\n清洗并统计成功！旧的ERA5清洗文件已被新版替换（原始raw数据未动）。")
    print(f"三个文件夹：江苏气象日度数据、江苏气象月度数据、江苏气象年度数据：{out}")
    print(f"城市-日期记录数：{len(daily):,}；完整日：{int(daily['全天数据完整'].sum()):,}")
    print(f"跳过批次数：{len(status['skipped_sources'])}；跨批次重复城市小时：{status['duplicate_city_hour_records_removed']:,}")
    if status["skipped_sources"]:
        print("请查看清洗质量报告.json中的skipped_sources")


if __name__ == "__main__":
    run()
