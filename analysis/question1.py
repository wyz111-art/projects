import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm

# ==================================================================
# 一、路径 & 常量
# ==================================================================
YB_FOLDER = "annual/annual"
LAND_2021 = "江苏城市土地数据2021.xlsx"
LAND_2025 = "江苏城市土地数据2025.csv"
AIR_FILE  = "jiangsu_yearly.csv"

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

YEARBOOK_FILE  = os.path.join(OUT_DIR, "统计年鉴_合并.csv")
AIR_FILE_CLEAN = os.path.join(OUT_DIR, "空气质量_总表.csv")
LAND_FILE      = os.path.join(OUT_DIR, "土地_合并.csv")

START_YEAR, END_YEAR = 2014, 2024
N_CLUSTERS   = 3
CLUSTER_SEED = 42

# 第二层用到的土地利用列
COMMON_COLS = ['Cropland (%)', 'Urban_Built-up (%)', 'Water (%)',
               'Grassland (%)', 'Other (%)']
LABELS = {'Cropland (%)': '耕地', 'Urban_Built-up (%)': '建设用地',
          'Water (%)': '水域', 'Grassland (%)': '草地', 'Other (%)': '其他'}


# ==================================================================
# 二、工具函数
# ==================================================================
def smart_read(path):
    """按扩展名读 xlsx/xls/csv，csv 自动尝试编码。"""
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    for enc in ['utf-8-sig', 'gbk', 'utf-8']:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别 {path} 的编码")


def find_col(df, pattern):
    """模糊匹配列名（支持正则）。"""
    for c in df.columns:
        if re.search(pattern, str(c)):
            return c
    return None


def read_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    return pd.read_excel(path) if path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(path)


def setup_chinese_font():
    """物理路径注册 + 名称匹配 + sns.set_style(rc=...)。"""
    candidate_paths = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\simsun.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    registered = []
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                registered.append(fm.FontProperties(fname=p).get_name())
            except Exception:
                pass
    if not registered:
        zh = ['Microsoft YaHei', 'SimHei', 'SimSun', 'PingFang SC', 'Heiti SC',
              'STHeiti', 'Noto Sans CJK SC', 'Source Han Sans SC',
              'WenQuanYi Zen Hei', 'Arial Unicode MS']
        avail = {f.name for f in fm.fontManager.ttflist}
        registered = [n for n in zh if n in avail]
    if not registered:
        print("[警告] 未找到中文字体，可能乱码。")
        registered = ['sans-serif']
    else:
        print(f"[字体] 使用中文字体: {registered[0]}")
    sns.set_style("whitegrid", rc={
        "font.sans-serif": registered, "font.family": "sans-serif",
        "axes.unicode_minus": False,
    })


def save_close(fname):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()
    print(f"  -> {fname}")


setup_chinese_font()


# ==================================================================
# 三、数据准备（合并 + 清洗，全局只做一次）
# ==================================================================
print("\n=== 数据准备 ===")

# ---- 3.1 统计年鉴合并 ----
yb_files = sorted(glob.glob(os.path.join(YB_FOLDER, "*.csv")))
if not yb_files:
    print(f"!! 在 {YB_FOLDER} 里没找到CSV")
else:
    frames = []
    for f in yb_files:
        d = smart_read(f)
        city = os.path.basename(f).replace('.csv', '').replace('市', '').strip()
        cc = find_col(d, '城市')
        if cc:
            d[cc] = d[cc].fillna(city).replace('', city)
        else:
            d['城市'] = city
        d.columns = [str(c).strip() for c in d.columns]
        frames.append(d)
    yb_all = pd.concat(frames, ignore_index=True)
    yb_all.to_csv(YEARBOOK_FILE, index=False, encoding='utf-8-sig')
    print(f"[统计年鉴] {yb_all.shape} 行, 城市 {sorted(yb_all['城市'].astype(str).unique())}")

# ---- 3.2 土地合并 ----
land_frames = []
for path, yr in [(LAND_2021, 2021), (LAND_2025, 2025)]:
    if os.path.exists(path):
        d = smart_read(path)
        d.columns = [str(c).strip() for c in d.columns]
        d['year'] = yr
        land_frames.append(d)
if land_frames:
    land_all = pd.concat(land_frames, ignore_index=True)
    land_all.to_csv(LAND_FILE, index=False, encoding='utf-8-sig')
    print(f"[土地] {land_all.shape} 行")

# ---- 3.3 空气质量 ----
if os.path.exists(AIR_FILE):
    air_raw = smart_read(AIR_FILE)
    air_raw.to_csv(AIR_FILE_CLEAN, index=False, encoding='utf-8-sig')
    print(f"[空气] {air_raw.shape} 行")


# ==================================================================
# 四、构建分析用表（一次清洗，两个主题共享）
# ==================================================================
print("\n=== 构建分析表 ===")

# ---- 4.1 面板：年鉴 + 空气 (2014-2024) ----
yb = read_table(YEARBOOK_FILE)
air = read_table(AIR_FILE_CLEAN)

col_city  = find_col(yb, r'城市')
col_year  = find_col(yb, r'年份')
col_sec   = find_col(yb, r'第二产业.*占')
col_gdp   = find_col(yb, r'人均地区生产|人均GDP')
col_green = find_col(yb, r'绿化覆盖')
col_park  = find_col(yb, r'公园绿地')
for name, c in [('城市', col_city), ('年份', col_year), ('二产占比', col_sec),
                ('人均GDP', col_gdp), ('绿化覆盖率', col_green), ('公园绿地', col_park)]:
    if c is None:
        raise ValueError(f"统计年鉴里没找到列: {name}")

yb_clean = pd.DataFrame({
    'city':       yb[col_city].astype(str).str.replace('市', '', regex=False).str.strip(),
    'year':       pd.to_numeric(yb[col_year], errors='coerce'),
    'sec_ratio':  pd.to_numeric(yb[col_sec],  errors='coerce'),
    'gdp_percap': pd.to_numeric(yb[col_gdp],  errors='coerce'),
    'green':      pd.to_numeric(yb[col_green],errors='coerce'),
    'park':       pd.to_numeric(yb[col_park], errors='coerce'),
})
yb_clean = yb_clean[(yb_clean['year'] >= START_YEAR) & (yb_clean['year'] <= END_YEAR)]

col_a_city = find_col(air, r'city|城市')
col_a_year = find_col(air, r'year|年份')
col_pm_daily = col_pm_intl = None
for c in air.columns:
    if re.search(r'PM\s*2\.5', str(c), re.I):
        if re.search(r'_24|日均|daily', str(c), re.I):
            col_pm_daily = col_pm_daily or c
        else:
            col_pm_intl = col_pm_intl or c
if col_pm_daily is None or col_pm_intl is None:
    raise ValueError("空气质量表里找不到两套 PM2.5 列")

air_clean = pd.DataFrame({
    'city':     air[col_a_city].astype(str).str.replace('市', '', regex=False).str.strip(),
    'year':     pd.to_numeric(air[col_a_year], errors='coerce'),
    'pm_daily': pd.to_numeric(air[col_pm_daily], errors='coerce'),
    'pm_intl':  pd.to_numeric(air[col_pm_intl],  errors='coerce'),
})

df = (yb_clean.merge(air_clean, on=['city', 'year'], how='inner')
      .sort_values(['city', 'year']).reset_index(drop=True))
df['ln_gdp'] = np.log(df['gdp_percap'])
print(f"[面板] {df.shape}, 城市 {df['city'].nunique()} × 年份 {df['year'].nunique()}")
miss = df.isna().sum()
if miss.any():
    print("缺失值:\n", miss[miss > 0])
n0 = len(df)
df = df.dropna(subset=['sec_ratio', 'gdp_percap', 'green', 'park', 'pm_daily', 'pm_intl'])
print(f"删除缺失: {n0} -> {len(df)} 行")
df.to_csv(os.path.join(OUT_DIR, 'panel_2014-2024_clean.csv'), index=False, encoding='utf-8-sig')

# ---- 4.2 土地 Δ 表 (2021 vs 2025) ----
land = pd.read_csv(LAND_FILE, encoding='utf-8-sig')
air2 = pd.read_csv(AIR_FILE_CLEAN, encoding='utf-8-sig')
land['City'] = land['City'].astype(str).str.replace('市', '', regex=False).str.strip()
air2['city'] = air2['city'].astype(str).str.replace('市', '', regex=False).str.strip()
print(f"[土地] 年份 {sorted(land['year'].unique())}, 城市数 {land['City'].nunique()}")

pm_cols = ['city', 'year'] + [c for c in ['PM2.5', 'PM2.5_24h'] if c in air2.columns]
air_land = air2[air2['year'].isin([2021, 2025])][pm_cols]

land_wide = land.pivot(index='City', columns='year')
delta = pd.DataFrame(index=land_wide.index)
for col in COMMON_COLS:
    if 2025 in land_wide[col] and 2021 in land_wide[col]:
        delta[f'Δ{LABELS[col]}'] = (land_wide[col][2025] - land_wide[col][2021]).astype(float)

pm_wide = air_land.pivot(index='city', columns='year', values='PM2.5')
delta['ΔPM2.5'] = pm_wide[2025] - pm_wide[2021] if 2025 in pm_wide and 2021 in pm_wide else np.nan
if delta['ΔPM2.5'].isna().all():
    print("!! 空气数据里没有 2025 年的 PM2.5")
delta = delta.dropna(subset=['ΔPM2.5'])
print(f"[Δ表] {delta.shape[0]} 个城市有完整两年数据")
delta.to_csv(os.path.join(OUT_DIR, 'table3_delta_land.csv'), encoding='utf-8-sig')


# ==================================================================
# 五、全部图表（集中绘制）
# ==================================================================
print("\n=== 绘图 ===")

# ---- 图1：PM2.5 热力图（两套口径）----
for pm_col, pm_label in [('pm_daily', '每日均值口径'), ('pm_intl', '国际算法年均口径')]:
    pivot = df.pivot_table(index='city', columns='year', values=pm_col)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    plt.figure(figsize=(11, 7))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=.5,
                cbar_kws={'label': f'PM2.5 ({pm_label}) μg/m³'})
    plt.title(f'江苏13市 PM2.5 热力图（{pm_label}）', fontsize=14)
    plt.xlabel('年份'); plt.ylabel('城市（按多年均值升序）')
    save_close(f'fig1_heatmap_{pm_col}.png')

# ---- 图2：PM2.5 多年均值对比 ----
mean_df = df.groupby('city')[['pm_daily', 'pm_intl']].mean().sort_values('pm_daily')
plt.figure(figsize=(10, 6))
x = np.arange(len(mean_df)); w = 0.4
plt.bar(x - w/2, mean_df['pm_daily'], w, label='每日均值口径', color='#e07a5f')
plt.bar(x + w/2, mean_df['pm_intl'],  w, label='国际算法年均口径', color='#3d5a80')
plt.xticks(x, mean_df.index, rotation=45)
plt.ylabel('PM2.5 多年均值 (μg/m³)')
plt.title('江苏13市 PM2.5 多年均值对比（2014-2024）', fontsize=14)
plt.legend()
save_close('fig2_mean_comparison.png')

# ---- 图3：经济-生态四象限 ----
city_avg = df.groupby('city')[['gdp_percap', 'green', 'pm_daily']].mean().reset_index()
plt.figure(figsize=(9, 7))
sc = plt.scatter(city_avg['gdp_percap'], city_avg['green'],
                 s=city_avg['pm_daily'] * 15, c=city_avg['pm_daily'],
                 cmap='RdYlGn_r', alpha=0.8, edgecolors='k')
for _, row in city_avg.iterrows():
    plt.annotate(row['city'], (row['gdp_percap'], row['green']),
                 textcoords='offset points', xytext=(0, 6), fontsize=9)
plt.colorbar(sc, label='PM2.5 多年均值 (μg/m³)')
plt.xlabel('人均GDP（元）'); plt.ylabel('建成区绿化覆盖率（%）')
plt.title('江苏13市 经济-生态-空气质量 四象限图（2014-2024均值）', fontsize=13)
plt.axvline(city_avg['gdp_percap'].median(), ls='--', color='gray', alpha=.6)
plt.axhline(city_avg['green'].median(), ls='--', color='gray', alpha=.6)
save_close('fig3_four_quadrant.png')

# ---- 图4：K-Means 聚类（仅第一层使用）----
cluster_feats = df.groupby('city')[['sec_ratio', 'ln_gdp', 'green', 'park']].mean()
X_scaled = StandardScaler().fit_transform(cluster_feats)
cluster_feats['cluster'] = KMeans(
    n_clusters=N_CLUSTERS, random_state=CLUSTER_SEED, n_init=10
).fit_predict(X_scaled)

pm_by_city = df.groupby('city')[['pm_daily', 'pm_intl']].mean()
cluster_df = cluster_feats.join(pm_by_city)
cluster_df['gdp_percap'] = np.exp(cluster_df['ln_gdp'])
print("\n各类型特征:"); print(cluster_df.groupby('cluster').mean().round(2))
cluster_df.to_csv(os.path.join(OUT_DIR, 'table2_cluster.csv'), encoding='utf-8-sig')

plot_df = cluster_df.reset_index()
plt.figure(figsize=(9, 7))
palette = ['#d64550', '#448aff', '#26a69a'][:N_CLUSTERS]
for cid in sorted(cluster_df['cluster'].unique()):
    sub = plot_df[plot_df['cluster'] == cid]
    plt.scatter(sub['gdp_percap'], sub['green'], s=140, c=palette[cid],
                label=f'类型{cid}', edgecolors='k', alpha=.8)
    for _, row in sub.iterrows():
        plt.annotate(row['city'], (row['gdp_percap'], row['green']),
                     textcoords='offset points', xytext=(0, 6), fontsize=9)
plt.xlabel('人均GDP（元）'); plt.ylabel('建成区绿化覆盖率（%）')
plt.title(f'江苏13市 K-Means 绿色转型类型划分（K={N_CLUSTERS}）', fontsize=13)
plt.legend()
save_close('fig4_cluster.png')

# ---- 图5：用地结构两年对比 ----
avg_2021 = land[land['year'] == 2021].set_index('City')[COMMON_COLS].mean()
avg_2025 = land[land['year'] == 2025].set_index('City')[COMMON_COLS].mean()
bar_df = pd.DataFrame({'2021': avg_2021.values, '2025': avg_2025.values},
                      index=[LABELS[c] for c in COMMON_COLS])
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for i, yr in enumerate(['2021', '2025']):
    ax[i].bar(bar_df.index, bar_df[yr],
              color=['#8d99ae', '#e07a5f', '#3d5a80', '#81b29a', '#f2cc8f'])
    ax[i].set_title(f'{yr} 江苏13市平均用地结构 (%)')
    ax[i].tick_params(axis='x', labelrotation=30)
    ax[i].set_ylim(0, 100)
    for xv, v in enumerate(bar_df[yr]):
        ax[i].text(xv, v + 1, f'{v:.1f}', ha='center', fontsize=9)
save_close('fig5_land_2021_vs_2025.png')

# ---- 图6：各城市用地结构变化 ----
delta_plot = delta[[c for c in delta.columns if c != 'ΔPM2.5']]
delta_plot = delta_plot.loc[delta_plot.sum(axis=1).sort_values().index]
plt.figure(figsize=(10, 6))
delta_plot.plot(kind='barh', figsize=(10, 6), ax=plt.gca(),
                color=['#e07a5f', '#3d5a80', '#81b29a', '#f2cc8f', '#8d99ae'])
plt.xlabel('用地占比变化 (百分点, 2025-2021)'); plt.ylabel('城市')
plt.axvline(0, color='gray', ls='--', linewidth=.8)
plt.title('江苏13市 土地利用结构变化 2021->2025')
save_close('fig6_delta_land_by_city.png')

# ---- 图7：Δ建设用地 vs ΔPM2.5 ----
if 'Δ建设用地' in delta.columns:
    plt.figure(figsize=(8, 6))
    xv = delta['Δ建设用地']; yv = delta['ΔPM2.5']
    plt.scatter(xv, yv, s=80, c='#3d5a80', edgecolors='k', alpha=.8)
    if len(delta) > 2:
        z = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 50)
        plt.plot(xs, np.polyval(z, xs), '--', color='#e07a5f')
    for name, row in delta.iterrows():
        plt.annotate(name, (row['Δ建设用地'], row['ΔPM2.5']),
                     textcoords='offset points', xytext=(0, 6), fontsize=9)
    plt.xlabel('Δ建设用地占比 (百分点, 2025-2021)')
    plt.ylabel('ΔPM2.5 (μg/m$^3$, 2025-2021)')
    plt.axhline(0, color='gray', ls='--', linewidth=.8)
    plt.axvline(0, color='gray', ls='--', linewidth=.8)
    plt.title('建设用地扩张 vs PM2.5变化')
    save_close('fig7_delta_urban_pm.png')


# ==================================================================
# 六、回归模型（集中拟合 + 存表）
# ==================================================================
print("\n=== 回归 ===")

# ---- 6.1 面板 OLS：PM2.5 ~ 二产/lnGDP/绿化/公园 ----
X = sm.add_constant(df[['sec_ratio', 'ln_gdp', 'green', 'park']])
reg_results = {}
for pm_col, pm_label in [('pm_daily', '每日均值口径'), ('pm_intl', '国际算法年均口径')]:
    m = sm.OLS(df[pm_col], X).fit()
    reg_results[pm_label] = m
    print(f"\n----- PM2.5（{pm_label}）-----")
    print(m.summary2().tables[1])

summary_tables = []
for label, m in reg_results.items():
    tab = m.summary2().tables[1][['Coef.', 'Std.Err.', 't', 'P>|t|']].copy()
    tab.columns = [f'{label}_{c}' for c in tab.columns]
    summary_tables.append(tab)
pd.concat(summary_tables, axis=1).to_csv(
    os.path.join(OUT_DIR, 'table1_regression.csv'), encoding='utf-8-sig')

# ---- 6.2 一阶差分：ΔPM2.5 ~ Δ用地结构 ----
delta_clean = delta.dropna()
X_cols = [c for c in delta_clean.columns if c != 'ΔPM2.5']
m_diff = sm.OLS(delta_clean['ΔPM2.5'],
                sm.add_constant(delta_clean[X_cols])).fit()
print(f"\n----- ΔPM2.5 ~ Δ用地结构（n={len(delta_clean)}，小样本注意解读）-----")
print(m_diff.summary2().tables[1].round(3))
m_diff.summary2().tables[1].to_csv(
    os.path.join(OUT_DIR, 'table4_delta_regression.csv'), encoding='utf-8-sig')


# ==================================================================
# 七、结论汇总（一个文件）
# ==================================================================
print("\n=== 结论 ===")
lines = []
for label, m in reg_results.items():
    lines.append(
        f"[{label}] 绿化覆盖率系数 = {m.params['green']:.3f} (p={m.pvalues['green']:.3f})，"
        f"二产占比系数 = {m.params['sec_ratio']:.3f} (p={m.pvalues['sec_ratio']:.3f})"
    )

with open(os.path.join(OUT_DIR, 'conclusion.txt'), 'w', encoding='utf-8') as f:
    f.write("江苏13市 产业-生态-土地利用 对 PM2.5 影响 分析结论\n" + "=" * 50 + "\n\n")
    f.write("【面板 OLS（2014-2024）】\n" + "\n".join(lines))
    f.write("\n\n【K-Means 类型划分】\n")
    f.write(cluster_df.groupby('cluster').mean().round(2).to_string())
    f.write("\n\n【Δ表（2025-2021）】\n")
    f.write(delta_clean.round(2).to_string())
    f.write("\n\n【一阶差分回归：ΔPM2.5 ~ Δ用地结构】\n")
    f.write(m_diff.summary2().as_text())

print("\n".join(lines))
print("\n完成！所有产出在", os.path.abspath(OUT_DIR))