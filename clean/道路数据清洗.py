from pathlib import Path
import geopandas as gpd
import numpy as np

gpkg_path = Path(r"C:\Users\a0475\Documents\重投影图层版.gpkg")  # 按你实际路径改

roads = gpd.read_file(gpkg_path, layer="重投影图层版热力图")

roads_only = roads[roads["highway"].notna()].copy()
print("道路要素数:", len(roads_only))
print(roads_only["highway"].value_counts().head(15))


# 1) 排除非机动车/步行道路
exclude = ["footway", "path", "cycleway", "track", "steps",
           "pedestrian", "bridleway", "corridor", "elevator"]
roads_mv = roads_only[~roads_only["highway"].isin(exclude)].copy()
print("机动车道路要素数:", len(roads_mv))

# 2) 按等级分组
def road_class(h):
    if h in ("motorway", "trunk", "motorway_link", "trunk_link"):
        return "高速/快速路"
    if h in ("primary", "secondary", "primary_link", "secondary_link"):
        return "主干道"
    if h in ("tertiary", "unclassified", "residential", "tertiary_link"):
        return "次干/支路"
    if h == "service":
        return "服务性道路"
    return "其他"

roads_mv["road_class"] = roads_mv["highway"].map(road_class)

# 3) 算长度（CRS 已是米制，直接算）
roads_mv["length_km"] = roads_mv.geometry.length / 1000
print(roads_mv.groupby("road_class")["length_km"].agg(["count", "sum"]).round(1))


geo_path = Path(r"C:\Users\a0475\Downloads\江苏省.geojson")  # 按你实际路径改
jiangsu = gpd.read_file(geo_path).to_crs("EPSG:32650")

# 先按江苏外框矩形粗筛，大幅减少 clip 的计算量
minx, miny, maxx, maxy = jiangsu.total_bounds
roads_cut = roads_mv.cx[minx:maxx, miny:maxy].copy()

# 再精确裁剪
roads_js = gpd.clip(roads_cut, jiangsu)

roads_js["length_km"] = roads_js.geometry.length / 1000
print("江苏机动车道路:", len(roads_js))
print(roads_js.groupby("road_class")["length_km"].agg(["count", "sum"]).round(1))


print("要素数:", len(jiangsu))
print("CRS:", jiangsu.crs)
print("范围:", jiangsu.total_bounds)
print("字段:", list(jiangsu.columns))
if "name" in jiangsu.columns:
    print(jiangsu["name"].head(20).tolist())
if "adcode" in jiangsu.columns:
    print(jiangsu["adcode"].astype(str).head(20).tolist())


roads_js["length_km"] = roads_js.geometry.length / 1000
print(roads_js["length_km"].describe(percentiles=[.5, .9, .99, .999]))
print("总长(km):", roads_js["length_km"].sum())
print(roads_js.groupby("road_class")["length_km"].agg(["count", "sum"]).round(1))


import pandas as pd

# 确保长度字段存在（米制 CRS，直接算）
if "length_km" not in roads_js.columns:
    roads_js["length_km"] = roads_js.geometry.length / 1000

print("=" * 60)
print("① 高速/快速路内部拆分：motorway / trunk / _link 各占多少")
print("=" * 60)
hw = roads_js[roads_js["road_class"] == "高速/快速路"]
print(hw.groupby("highway")["length_km"].agg(["count", "sum"]).round(1))
print("高速/快速路合计(km):", round(hw["length_km"].sum(), 1))

print("\n" + "=" * 60)
print("② 全数据中最长的 10 条线")
print("=" * 60)
top10 = roads_js.nlargest(10, "length_km")[["osm_id", "name", "highway", "length_km"]]
print(top10.to_string(index=False))

print("\n" + "=" * 60)
print("③ 重复 osm_id 检查")
print("=" * 60)
dup_count = roads_js["osm_id"].duplicated().sum()
print("重复 osm_id 条数:", dup_count)
print("唯一 osm_id 数:", roads_js["osm_id"].nunique(), "/ 总条数:", len(roads_js))
if dup_count > 0:
    # 看重复最多的前几个 osm_id 及其 highway 类型
    dup_top = (roads_js[roads_js["osm_id"].duplicated(keep=False)]
               .groupby("osm_id")
               .agg(出现次数=("osm_id", "size"),
                    道路类型=("highway", lambda s: ",".join(sorted(set(s)))),
                    总长_km=("length_km", "sum"))
               .sort_values("出现次数", ascending=False)
               .head(10))
    print("\n重复最多的 osm_id（前10）:")
    print(dup_top.to_string())

jiangsu_prov = jiangsu.dissolve()   # 13个市合并成省多边形

print("\n" + "=" * 60)
print("④ roads_js 范围 vs 江苏边界范围")
print("=" * 60)
print("roads_js 范围:", [round(v, 1) for v in roads_js.total_bounds])
print("江苏边界范围:", [round(v, 1) for v in jiangsu_prov.total_bounds])

rb, jb = roads_js.total_bounds, jiangsu_prov.total_bounds
tolerance = 1000  # 允许 1 km 误差（投影/边界精度）
within = (rb[0] >= jb[0] - tolerance and rb[1] >= jb[1] - tolerance and
          rb[2] <= jb[2] + tolerance and rb[3] <= jb[3] + tolerance)
print("roads_js 是否落在江苏边界内(±1km):", within)
if not within:
    print("⚠️ 范围超出，可能 clip 用的不是江苏边界，或打印的是裁剪前的变量")


# 1) 排除非现状道路
exclude_status = ["proposed", "construction", "abandoned", "disused", "planned", "razed"]
roads_clean = roads_js[~roads_js["highway"].isin(exclude_status)].copy()

# 2) "其他"里的 living_street / road 并入次干支路，其余保留为"其他"
roads_clean.loc[roads_clean["highway"].isin(["living_street", "road"]), "road_class"] = "次干/支路"

# 3) 重算长度并汇总
roads_clean["length_km"] = roads_clean.geometry.length / 1000
print("清理后要素数:", len(roads_clean))
print(roads_clean.groupby("road_class")["length_km"].agg(["count", "sum"]).round(1))


out_path = Path(r"C:\Users\a0475\Documents\temp\用py\jiangsu_roads_clean.gpkg")

# 1) 整体图层
roads_clean.to_file(out_path, layer="roads_all", driver="GPKG")

# 2) 按道路等级拆成子图层（可选，但后续按等级分析很方便）
for cls, gdf in roads_clean.groupby("road_class"):
    layer_name = {
        "高速/快速路": "roads_motorway",
        "主干道": "roads_arterial",
        "次干/支路": "roads_local",
        "服务性道路": "roads_service",
        "其他": "roads_other",
    }.get(cls, "roads_other")
    gdf.to_file(out_path, layer=layer_name, driver="GPKG")

print("写入完成")