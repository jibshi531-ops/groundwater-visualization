"""
甘肃省地下水-降水-高程多源数据可视化平台
页面风格参考：左侧上传 + 中部地图 + 右侧趋势图/散点图/饼图 + 底部融合统计表

运行方式：
1. 保存为 app.py
2. 安装依赖：
   pip install streamlit geopandas rasterio folium streamlit-folium plotly pandas numpy matplotlib pillow shapely pyproj fiona affine
3. 运行：
   streamlit run app.py

数据要求：
1. 甘肃省边界 SHP：压缩成 zip 上传，zip 内必须包含 .shp/.shx/.dbf/.prj
2. 10 年降水 TIF：建议命名为 pre_2015.tif、pre_2016.tif ...
3. 10 年地下水变化 TIF：建议命名为 gw_2015.tif、gw_2016.tif ...
4. DEM 高程 TIF：一个 tif 文件
"""

import os
import re
import io
import base64
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from affine import Affine

import folium
from streamlit_folium import st_folium

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image


# =========================================================
# 1. 页面基础设置
# =========================================================
st.set_page_config(
    page_title="甘肃省地下水-降水-高程多源数据可视化平台",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. 页面 CSS 美化
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #f2f6fb 100%);
        border-right: 1px solid #e5e7eb;
    }

    h1 {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    .small-subtitle {
        color: #475569;
        font-size: 0.92rem;
        margin-top: -0.6rem;
        margin-bottom: 1rem;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        margin-bottom: 14px;
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .success-box {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 0.5rem 0.7rem;
        border-radius: 10px;
        font-size: 0.86rem;
        margin-bottom: 0.5rem;
    }

    .warning-box {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        font-size: 0.88rem;
        margin-bottom: 0.75rem;
    }

    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 0.3rem;
    }

    .stButton>button {
        border-radius: 12px;
        height: 2.7rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ef4444, #f97316);
        color: white;
        border: none;
    }

    .stDownloadButton>button {
        border-radius: 10px;
        font-weight: 700;
        background: #22c55e;
        color: white;
        border: none;
    }

    .footer-note {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.5rem;
    }

    iframe {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. 工具函数
# =========================================================
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_uploaded_file(uploaded_file, out_dir):
    safe_mkdir(out_dir)
    out_path = os.path.join(out_dir, uploaded_file.name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def unzip_shp(zip_file_path, out_dir):
    safe_mkdir(out_dir)
    with zipfile.ZipFile(zip_file_path, "r") as z:
        z.extractall(out_dir)

    shp_files = list(Path(out_dir).rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("zip 中没有找到 .shp 文件。请确认压缩包内包含 .shp/.shx/.dbf/.prj。")
    return str(shp_files[0])


def extract_year(filename):
    m = re.search(r"(19\d{2}|20\d{2})", Path(filename).stem)
    return int(m.group(1)) if m else None


def save_tif_files(uploaded_files, out_dir):
    tif_dict = {}
    if not uploaded_files:
        return tif_dict

    for i, uf in enumerate(uploaded_files, start=1):
        path = save_uploaded_file(uf, out_dir)
        year = extract_year(uf.name)
        if year is None:
            year = i
        tif_dict[year] = path

    return dict(sorted(tif_dict.items(), key=lambda x: x[0]))


def read_boundary_from_zip(uploaded_zip, work_dir):
    zip_path = save_uploaded_file(uploaded_zip, os.path.join(work_dir, "boundary_zip"))
    shp_path = unzip_shp(zip_path, os.path.join(work_dir, "boundary_unzip"))

    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError("边界 SHP 为空。")

    if gdf.crs is None:
        raise ValueError("边界 SHP 没有坐标系。请先在 ArcGIS/QGIS 中定义投影。")

    # 合并所有面，避免多图斑重复边界
    gdf = gdf.dissolve().reset_index(drop=True)

    return gdf


def crop_raster_by_boundary(raster_path, boundary_gdf):
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"{Path(raster_path).name} 没有坐标系，请先定义投影。")

        boundary_in_raster_crs = boundary_gdf.to_crs(src.crs)
        geoms = [geom for geom in boundary_in_raster_crs.geometry if geom is not None]

        out_image, out_transform = mask(
            src,
            geoms,
            crop=True,
            filled=False,
            all_touched=True
        )

        arr = out_image[0].astype("float64")
        arr = np.ma.filled(arr, np.nan)

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        arr[np.isinf(arr)] = np.nan

        profile = src.profile.copy()
        profile.update(
            {
                "height": arr.shape[0],
                "width": arr.shape[1],
                "transform": out_transform
            }
        )

        return arr, out_transform, src.crs, profile


def raster_stats(arr):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {
            "均值": np.nan,
            "最小值": np.nan,
            "最大值": np.nan,
            "标准差": np.nan,
            "有效像元数": 0
        }

    return {
        "均值": float(np.nanmean(valid)),
        "最小值": float(np.nanmin(valid)),
        "最大值": float(np.nanmax(valid)),
        "标准差": float(np.nanstd(valid)),
        "有效像元数": int(valid.size)
    }


def raster_to_wgs84(arr, transform, crs, max_size=900):
    height, width = arr.shape
    left, bottom, right, top = array_bounds(height, width, transform)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs,
        "EPSG:4326",
        width,
        height,
        left,
        bottom,
        right,
        top
    )

    # 控制网页展示尺寸，避免 TIF 太大导致卡顿
    max_dim = max(dst_width, dst_height)
    if max_dim > max_size:
        scale = max_dim / max_size
        new_width = max(1, int(dst_width / scale))
        new_height = max(1, int(dst_height / scale))
        dst_transform = dst_transform * Affine.scale(dst_width / new_width, dst_height / new_height)
        dst_width, dst_height = new_width, new_height

    dst = np.full((dst_height, dst_width), np.nan, dtype="float32")

    reproject(
        source=arr.astype("float32"),
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear
    )

    south, west, north, east = array_bounds(dst_height, dst_width, dst_transform)
    return dst, [[south, west], [north, east]]


def array_to_png_data_uri(arr, cmap_name="Blues", opacity=0.78):
    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    else:
        vmin = np.nanpercentile(valid, 2)
        vmax = np.nanpercentile(valid, 98)

        if np.isclose(vmin, vmax):
            vmin = np.nanmin(valid)
            vmax = np.nanmax(valid)

        if np.isclose(vmin, vmax):
            vmax = vmin + 1

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
        rgba_float = cmap(norm(arr))
        rgba_float[..., 3] = np.where(np.isfinite(arr), opacity, 0)
        rgba = (rgba_float * 255).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    encoded = base64.b64encode(bio.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def make_base_map(boundary_gdf):
    boundary_wgs = boundary_gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = boundary_wgs.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    m = folium.Map(
        location=center,
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True
    )

    folium.GeoJson(
        boundary_wgs,
        name="甘肃省边界",
        style_function=lambda x: {
            "color": "#111827",
            "weight": 2,
            "fillOpacity": 0
        }
    ).add_to(m)

    m.fit_bounds([[miny, minx], [maxy, maxx]])
    return m


def add_raster_layer(m, raster_path, boundary_gdf, layer_name, cmap_name, opacity):
    arr, transform, crs, profile = crop_raster_by_boundary(raster_path, boundary_gdf)
    arr_wgs, bounds = raster_to_wgs84(arr, transform, crs)
    png_uri = array_to_png_data_uri(arr_wgs, cmap_name=cmap_name, opacity=opacity)

    folium.raster_layers.ImageOverlay(
        image=png_uri,
        bounds=bounds,
        name=layer_name,
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=2
    ).add_to(m)

    return arr


def build_yearly_table(pre_dict, gw_dict, dem_path, boundary_gdf):
    common_years = sorted(set(pre_dict.keys()) & set(gw_dict.keys()))

    dem_arr, _, _, _ = crop_raster_by_boundary(dem_path, boundary_gdf)
    dem_stats = raster_stats(dem_arr)

    records = []
    for year in common_years:
        pre_arr, _, _, _ = crop_raster_by_boundary(pre_dict[year], boundary_gdf)
        gw_arr, _, _, _ = crop_raster_by_boundary(gw_dict[year], boundary_gdf)

        pre_stats = raster_stats(pre_arr)
        gw_stats = raster_stats(gw_arr)

        records.append(
            {
                "年份": year,
                "降水量(mm)": pre_stats["均值"],
                "地下水变化量(m)": gw_stats["均值"],
                "平均高程(m)": dem_stats["均值"],
                "高程最小值(m)": dem_stats["最小值"],
                "高程最大值(m)": dem_stats["最大值"],
                "有效像元数": min(pre_stats["有效像元数"], gw_stats["有效像元数"])
            }
        )

    df = pd.DataFrame(records)
    return df


def groundwater_pie_df(arr, threshold=0.1):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return pd.DataFrame({"类型": [], "像元数": []})

    up = np.sum(valid > threshold)
    stable = np.sum((valid >= -threshold) & (valid <= threshold))
    down = np.sum(valid < -threshold)

    return pd.DataFrame(
        {
            "类型": [
                f"上升区（>{threshold}）",
                f"稳定区（-{threshold}~{threshold}）",
                f"下降区（<-{threshold}）"
            ],
            "像元数": [up, stable, down]
        }
    )


def elevation_pie_df(arr):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return pd.DataFrame({"类型": [], "像元数": []})

    q1, q2, q3 = np.nanpercentile(valid, [25, 50, 75])

    zones = [
        np.sum(valid <= q1),
        np.sum((valid > q1) & (valid <= q2)),
        np.sum((valid > q2) & (valid <= q3)),
        np.sum(valid > q3)
    ]

    names = [
        f"低高程区 ≤{q1:.0f}m",
        f"中低高程区 {q1:.0f}-{q2:.0f}m",
        f"中高高程区 {q2:.0f}-{q3:.0f}m",
        f"高高程区 >{q3:.0f}m"
    ]

    return pd.DataFrame({"类型": names, "像元数": zones})


def demo_dataframe():
    years = list(range(2015, 2025))
    rainfall = [428.6, 475.2, 513.7, 562.1, 498.3, 610.5, 535.6, 505.2, 462.4, 438.9]
    groundwater = [-0.35, -0.18, 0.05, 0.21, -0.02, 0.32, 0.08, 0.12, -0.10, 0.00]
    elevation = [1876.2] * 10

    return pd.DataFrame(
        {
            "年份": years,
            "降水量(mm)": rainfall,
            "地下水变化量(m)": groundwater,
            "平均高程(m)": elevation,
            "高程最小值(m)": [650.0] * 10,
            "高程最大值(m)": [4800.0] * 10,
            "有效像元数": [10000] * 10
        }
    )


def demo_map():
    # 粗略甘肃范围示意，不代表真实边界
    m = folium.Map(location=[38.5, 101.5], zoom_start=6, tiles="CartoDB positron", control_scale=True)

    polygon = [
        [40.0, 93.5],
        [39.7, 96.0],
        [40.5, 98.2],
        [39.1, 100.0],
        [39.6, 102.2],
        [38.5, 104.5],
        [36.5, 106.0],
        [35.0, 105.0],
        [34.6, 102.8],
        [36.0, 100.0],
        [37.0, 97.0],
        [38.2, 94.5]
    ]

    folium.Polygon(
        locations=polygon,
        color="#111827",
        weight=2,
        fill=True,
        fill_color="#60a5fa",
        fill_opacity=0.45,
        tooltip="甘肃省示意范围"
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def card_start(title):
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 4. 左侧栏：上传和控制
# =========================================================
with st.sidebar:
    st.markdown("## 数据上传")

    boundary_zip = st.file_uploader(
        "1. 上传甘肃省边界 SHP 压缩包（zip）",
        type=["zip"],
        help="必须包含 .shp/.shx/.dbf/.prj。"
    )

    pre_files = st.file_uploader(
        "2. 上传 10 年降水 TIF 文件（2015-2024）",
        type=["tif", "tiff"],
        accept_multiple_files=True
    )

    gw_files = st.file_uploader(
        "3. 上传 10 年地下水变化 TIF 文件（2015-2024）",
        type=["tif", "tiff"],
        accept_multiple_files=True
    )

    dem_file = st.file_uploader(
        "4. 上传甘肃省 DEM 高程 TIF 文件",
        type=["tif", "tiff"]
    )

    st.markdown("## 显示设置")

    demo_mode = st.toggle(
        "演示模式",
        value=not all([boundary_zip, pre_files, gw_files, dem_file]),
        help="没有上传数据时，可以先用演示模式查看页面效果。上传真实数据后建议关闭。"
    )

    selected_year_sidebar = st.selectbox(
        "选择年份",
        list(range(2015, 2025)),
        index=5
    )

    layer_choice = st.radio(
        "底图选择",
        ["降水图", "地下水变化图", "高程图", "三图叠加"],
        index=0
    )

    opacity = st.slider("透明度", 0.1, 1.0, 0.75, 0.05)

    stable_threshold = st.number_input(
        "地下水稳定阈值",
        min_value=0.0,
        value=0.10,
        step=0.05,
        help="例如 ±0.10m 内可视为基本稳定。"
    )

    refresh = st.button("刷新地图", use_container_width=True)


# =========================================================
# 5. 标题区
# =========================================================
st.markdown("### 甘肃省地下水-降水-高程多源数据可视化平台")
st.markdown(
    '<div class="small-subtitle">上传甘肃省边界 SHP、10 年降水 TIF、10 年地下水变化 TIF 和 DEM 高程 TIF，进行多源数据可视化分析</div>',
    unsafe_allow_html=True
)


# =========================================================
# 6. 主体逻辑
# =========================================================
has_real_data = all([boundary_zip, pre_files, gw_files, dem_file]) and not demo_mode

try:
    if has_real_data:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = safe_mkdir(os.path.join(tmpdir, "gansu_app"))

            boundary_gdf = read_boundary_from_zip(boundary_zip, work_dir)
            pre_dict = save_tif_files(pre_files, os.path.join(work_dir, "pre"))
            gw_dict = save_tif_files(gw_files, os.path.join(work_dir, "gw"))
            dem_path = save_uploaded_file(dem_file, os.path.join(work_dir, "dem"))

            common_years = sorted(set(pre_dict.keys()) & set(gw_dict.keys()))
            if len(common_years) == 0:
                st.error("降水 TIF 和地下水变化 TIF 没有匹配年份，请检查文件名是否包含相同年份。")
                st.stop()

            selected_year = selected_year_sidebar if selected_year_sidebar in common_years else common_years[-1]

            # 当前年份栅格
            current_pre_arr = None
            current_gw_arr = None
            current_dem_arr = None

            # 地图
            fmap = make_base_map(boundary_gdf)

            if layer_choice in ["降水图", "三图叠加"]:
                current_pre_arr = add_raster_layer(
                    fmap, pre_dict[selected_year], boundary_gdf,
                    f"{selected_year} 年降水量", "Blues", opacity
                )

            if layer_choice in ["地下水变化图", "三图叠加"]:
                current_gw_arr = add_raster_layer(
                    fmap, gw_dict[selected_year], boundary_gdf,
                    f"{selected_year} 年地下水变化", "RdYlBu", opacity
                )

            if layer_choice in ["高程图", "三图叠加"]:
                current_dem_arr = add_raster_layer(
                    fmap, dem_path, boundary_gdf,
                    "DEM 高程", "terrain", opacity
                )

            folium.LayerControl(collapsed=False).add_to(fmap)

            # 若部分数组没有通过当前图层加载，则单独计算
            if current_pre_arr is None:
                current_pre_arr, _, _, _ = crop_raster_by_boundary(pre_dict[selected_year], boundary_gdf)
            if current_gw_arr is None:
                current_gw_arr, _, _, _ = crop_raster_by_boundary(gw_dict[selected_year], boundary_gdf)
            if current_dem_arr is None:
                current_dem_arr, _, _, _ = crop_raster_by_boundary(dem_path, boundary_gdf)

            yearly_df = build_yearly_table(pre_dict, gw_dict, dem_path, boundary_gdf)

            pre_stats = raster_stats(current_pre_arr)
            gw_stats = raster_stats(current_gw_arr)
            dem_stats = raster_stats(current_dem_arr)

            source_note = "当前使用真实上传数据。"

            # 页面布局
            top1, top2, top3, top4 = st.columns(4)
            with top1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">匹配年份</div><div class="metric-value">{len(common_years)} 年</div></div>', unsafe_allow_html=True)
            with top2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">当前年份</div><div class="metric-value">{selected_year}</div></div>', unsafe_allow_html=True)
            with top3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">降水均值</div><div class="metric-value">{pre_stats["均值"]:.2f}</div></div>', unsafe_allow_html=True)
            with top4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">地下水变化均值</div><div class="metric-value">{gw_stats["均值"]:.3f}</div></div>', unsafe_allow_html=True)

            row1_col1, row1_col2 = st.columns([1.06, 1.72])

            with row1_col1:
                card_start(f"甘肃省空间数据展示（{selected_year}年）")
                st_folium(fmap, width=None, height=430)
                st.markdown(f'<div class="footer-note">{source_note}</div>', unsafe_allow_html=True)
                card_end()

            with row1_col2:
                card_start("10 年变化趋势")
                chart_a, chart_b = st.columns(2)

                with chart_a:
                    fig_rain = px.line(
                        yearly_df,
                        x="年份",
                        y="降水量(mm)",
                        markers=True,
                        title="年降水量变化趋势"
                    )
                    fig_rain.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_rain, use_container_width=True)

                with chart_b:
                    fig_gw = px.line(
                        yearly_df,
                        x="年份",
                        y="地下水变化量(m)",
                        markers=True,
                        title="年地下水变化趋势"
                    )
                    fig_gw.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_gw.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_gw, use_container_width=True)

                card_end()

            row2_col1, row2_col2 = st.columns([1, 1])

            with row2_col1:
                card_start(f"降水量 vs 地下水变化（{yearly_df['年份'].min()}-{yearly_df['年份'].max()}）")
                if len(yearly_df) >= 2:
                    corr = yearly_df[["降水量(mm)", "地下水变化量(m)"]].corr().iloc[0, 1]
                else:
                    corr = np.nan

                fig_scatter = px.scatter(
                    yearly_df,
                    x="降水量(mm)",
                    y="地下水变化量(m)",
                    text="年份",
                    trendline="ols" if len(yearly_df) >= 3 else None
                )
                fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
                fig_scatter.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption(f"相关系数：{corr:.3f}" if np.isfinite(corr) else "相关系数：样本不足")
                card_end()

            with row2_col2:
                card_start("地下水变化分区占比")
                pie_df = groundwater_pie_df(current_gw_arr, stable_threshold)
                fig_pie = px.pie(
                    pie_df,
                    names="类型",
                    values="像元数",
                    hole=0.45
                )
                fig_pie.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
                card_end()

            row3_col1, row3_col2 = st.columns([1.38, 1])

            with row3_col1:
                card_start("多源数据融合统计表（区域平均值）")
                show_df = yearly_df.copy()
                numeric_cols = show_df.select_dtypes(include=["float", "float64"]).columns
                show_df[numeric_cols] = show_df[numeric_cols].round(3)
                st.dataframe(show_df, use_container_width=True, height=310)

                csv = show_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "导出 CSV",
                    data=csv,
                    file_name="甘肃省_降水_地下水_高程_融合统计表.csv",
                    mime="text/csv"
                )
                card_end()

            with row3_col2:
                card_start("高程分区占比")
                elev_df = elevation_pie_df(current_dem_arr)
                fig_elev = px.pie(
                    elev_df,
                    names="类型",
                    values="像元数"
                )
                fig_elev.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_elev, use_container_width=True)
                card_end()

    else:
        # =====================================================
        # 演示模式：不用真实数据也能看到页面效果
        # =====================================================
        selected_year = selected_year_sidebar
        yearly_df = demo_dataframe()

        st.markdown(
            '<div class="warning-box">当前为演示模式：页面使用模拟数据展示效果。上传真实 SHP/TIF 后，关闭“演示模式”即可进行真实计算。</div>',
            unsafe_allow_html=True
        )

        top1, top2, top3, top4 = st.columns(4)
        with top1:
            st.markdown('<div class="metric-card"><div class="metric-label">匹配年份</div><div class="metric-value">10 年</div></div>', unsafe_allow_html=True)
        with top2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">当前年份</div><div class="metric-value">{selected_year}</div></div>', unsafe_allow_html=True)
        with top3:
            value = yearly_df.loc[yearly_df["年份"] == selected_year, "降水量(mm)"].iloc[0]
            st.markdown(f'<div class="metric-card"><div class="metric-label">降水均值</div><div class="metric-value">{value:.1f} mm</div></div>', unsafe_allow_html=True)
        with top4:
            value = yearly_df.loc[yearly_df["年份"] == selected_year, "地下水变化量(m)"].iloc[0]
            st.markdown(f'<div class="metric-card"><div class="metric-label">地下水变化均值</div><div class="metric-value">{value:.2f} m</div></div>', unsafe_allow_html=True)

        row1_col1, row1_col2 = st.columns([1.06, 1.72])

        with row1_col1:
            card_start(f"甘肃省空间数据展示（{selected_year}年）")
            st_folium(demo_map(), width=None, height=430)
            st.markdown('<div class="footer-note">演示地图仅为页面效果示意，真实边界以上传 SHP 为准。</div>', unsafe_allow_html=True)
            card_end()

        with row1_col2:
            card_start("10 年变化趋势")
            chart_a, chart_b = st.columns(2)

            with chart_a:
                fig_rain = px.line(
                    yearly_df,
                    x="年份",
                    y="降水量(mm)",
                    markers=True,
                    title="年降水量变化趋势"
                )
                fig_rain.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_rain, use_container_width=True)

            with chart_b:
                fig_gw = px.line(
                    yearly_df,
                    x="年份",
                    y="地下水变化量(m)",
                    markers=True,
                    title="年地下水变化趋势"
                )
                fig_gw.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_gw.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_gw, use_container_width=True)

            card_end()

        row2_col1, row2_col2 = st.columns([1, 1])

        with row2_col1:
            card_start("降水量 vs 地下水变化（2015-2024）")
            corr = yearly_df[["降水量(mm)", "地下水变化量(m)"]].corr().iloc[0, 1]
            fig_scatter = px.scatter(
                yearly_df,
                x="降水量(mm)",
                y="地下水变化量(m)",
                text="年份",
                trendline="ols"
            )
            fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
            fig_scatter.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"相关系数：{corr:.3f}")
            card_end()

        with row2_col2:
            card_start("地下水变化分区占比")
            pie_df = pd.DataFrame(
                {
                    "类型": ["上升区（>0.1m）", "稳定区（-0.1m~0.1m）", "下降区（<-0.1m）"],
                    "像元数": [2830, 4670, 2500]
                }
            )
            fig_pie = px.pie(
                pie_df,
                names="类型",
                values="像元数",
                hole=0.45
            )
            fig_pie.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            card_end()

        row3_col1, row3_col2 = st.columns([1.38, 1])

        with row3_col1:
            card_start("多源数据融合统计表（区域平均值）")
            show_df = yearly_df.copy()
            show_df[["降水量(mm)", "地下水变化量(m)", "平均高程(m)"]] = show_df[
                ["降水量(mm)", "地下水变化量(m)", "平均高程(m)"]
            ].round(3)
            st.dataframe(show_df, use_container_width=True, height=310)

            csv = show_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "导出 CSV",
                data=csv,
                file_name="演示_甘肃省_降水_地下水_高程_融合统计表.csv",
                mime="text/csv"
            )
            card_end()

        with row3_col2:
            card_start("高程分区占比")
            elev_df = pd.DataFrame(
                {
                    "类型": [">3000m", "2000-3000m", "1000-2000m", "<1000m"],
                    "像元数": [2750, 3510, 2480, 1260]
                }
            )
            fig_elev = px.pie(
                elev_df,
                names="类型",
                values="像元数"
            )
            fig_elev.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_elev, use_container_width=True)
            card_end()

except Exception as e:
    st.error("程序运行出错，请检查 SHP/TIF 坐标系、文件格式、文件名年份是否正确。")
    st.exception(e)


# =========================================================
# 7. 底部说明
# =========================================================
with st.expander("方法说明"):
    st.markdown(
        """
        本平台以甘肃省边界为统一空间范围，将降水 TIF、地下水变化 TIF 和 DEM 高程 TIF 进行裁剪、统计和可视化。

        **基本处理流程：**

        1. 上传甘肃省边界 SHP 压缩包；
        2. 根据边界裁剪降水、地下水变化和 DEM 栅格；
        3. 通过文件名中的年份匹配 10 年降水和地下水变化数据；
        4. 计算每一年区域平均降水量、地下水变化量和高程统计值；
        5. 通过地图、趋势图、散点图、饼图和统计表展示多源数据关系。

        **地下水变化解释建议：**

        - 正值：地下水位上升；
        - 负值：地下水位下降；
        - 接近 0：地下水基本稳定。

        如果你的地下水数据是“埋深变化量”，则解释方向可能相反，需要在论文或报告中说明。
        """
    )
