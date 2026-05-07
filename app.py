# -*- coding: utf-8 -*-
"""
Design and Development of a Groundwater Observation Visualization Platform
Based on Multi-source Data Fusion

Version:
GitHub auto data loading + optional manual upload + demo mode

Recommended GitHub repository structure:

groundwater-visualization/
├─ app.py
├─ requirements.txt
├─ packages.txt
├─ .streamlit/
│  └─ config.toml
├─ boundary/
│  └─ gansu_boundary.zip
├─ precip/
│  ├─ pre_2012_sum_clip.tif
│  ├─ pre_2013_sum_clip.tif
│  └─ ...
├─ groundwater/
│  ├─ GWs_2012_mean_gansu.tif
│  ├─ GWs_2013_mean_gansu.tif
│  └─ ...
└─ dem/
   └─ dem_gansu.tif

Run:
streamlit run app.py
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

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image


# =========================================================
# 1. Page settings
# =========================================================
st.set_page_config(
    page_title="Groundwater Observation Visualization Platform Based on Multi-source Data Fusion",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CSS style
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

    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 800 !important;
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
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        font-size: 0.88rem;
        margin-bottom: 0.75rem;
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

    iframe {
        border-radius: 12px !important;
    }

    .stDownloadButton>button {
        border-radius: 10px;
        font-weight: 700;
        background: #22c55e;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. Data directory settings
# =========================================================
BASE_DIR = Path(__file__).parent

AUTO_BOUNDARY_DIR = BASE_DIR / "boundary"
AUTO_PRECIP_DIR = BASE_DIR / "precip"
AUTO_GW_DIR = BASE_DIR / "groundwater"
AUTO_DEM_DIR = BASE_DIR / "dem"


# =========================================================
# 4. Utility functions
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
        raise FileNotFoundError(
            "No .shp file was found in the ZIP file. Please make sure the ZIP contains .shp, .shx, .dbf and .prj files."
        )
    return str(shp_files[0])


def extract_year(filename):
    m = re.search(r"(19\d{2}|20\d{2})", Path(filename).stem)
    return int(m.group(1)) if m else None


def make_tif_dict_from_paths(paths):
    tif_dict = {}
    for i, p in enumerate(sorted(paths), start=1):
        year = extract_year(Path(p).name)
        if year is None:
            year = i
        tif_dict[year] = str(p)
    return dict(sorted(tif_dict.items(), key=lambda x: x[0]))


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


def find_auto_data():
    boundary_zip_list = sorted(list(AUTO_BOUNDARY_DIR.glob("*.zip")))
    pre_list = sorted(list(AUTO_PRECIP_DIR.glob("*.tif")) + list(AUTO_PRECIP_DIR.glob("*.tiff")))
    gw_list = sorted(list(AUTO_GW_DIR.glob("*.tif")) + list(AUTO_GW_DIR.glob("*.tiff")))
    dem_list = sorted(list(AUTO_DEM_DIR.glob("*.tif")) + list(AUTO_DEM_DIR.glob("*.tiff")))

    return {
        "boundary_zip": str(boundary_zip_list[0]) if boundary_zip_list else None,
        "pre_dict": make_tif_dict_from_paths(pre_list),
        "gw_dict": make_tif_dict_from_paths(gw_list),
        "dem_path": str(dem_list[0]) if dem_list else None,
        "counts": {
            "boundary_zip": len(boundary_zip_list),
            "precip_tif": len(pre_list),
            "groundwater_tif": len(gw_list),
            "dem_tif": len(dem_list)
        }
    }


def read_boundary_from_zip_path(zip_path, work_dir):
    shp_path = unzip_shp(zip_path, os.path.join(work_dir, "boundary_unzip"))
    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError("The boundary shapefile is empty.")

    if gdf.crs is None:
        raise ValueError("The boundary shapefile has no coordinate reference system.")

    return gdf.dissolve().reset_index(drop=True)


def read_boundary_from_uploaded_zip(uploaded_zip, work_dir):
    zip_path = save_uploaded_file(uploaded_zip, os.path.join(work_dir, "boundary_zip"))
    return read_boundary_from_zip_path(zip_path, work_dir)


def crop_raster_by_boundary(raster_path, boundary_gdf):
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"{Path(raster_path).name} has no coordinate reference system.")

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
            "Mean": np.nan,
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Std": np.nan,
            "Valid Pixels": 0
        }

    return {
        "Mean": float(np.nanmean(valid)),
        "Minimum": float(np.nanmin(valid)),
        "Maximum": float(np.nanmax(valid)),
        "Std": float(np.nanstd(valid)),
        "Valid Pixels": int(valid.size)
    }


def raster_to_wgs84(arr, transform, crs, max_size=300):
    height, width = arr.shape
    west, south, east, north = array_bounds(height, width, transform)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs,
        "EPSG:4326",
        width,
        height,
        west,
        south,
        east,
        north
    )

    max_dim = max(dst_width, dst_height)
    if max_dim > max_size:
        scale = max_dim / max_size
        new_width = max(1, int(dst_width / scale))
        new_height = max(1, int(dst_height / scale))

        dst_transform = dst_transform * Affine.scale(
            dst_width / new_width,
            dst_height / new_height
        )

        dst_width, dst_height = new_width, new_height

    nodata_value = -9999.0

    src_arr = arr.astype("float32")
    src_arr = np.where(np.isfinite(src_arr), src_arr, nodata_value)

    dst = np.full((dst_height, dst_width), nodata_value, dtype="float32")

    reproject(
        source=src_arr,
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=nodata_value,
        dst_nodata=nodata_value,
        resampling=Resampling.bilinear
    )

    dst = np.where(dst == nodata_value, np.nan, dst)

    west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)
    bounds = [[south, west], [north, east]]

    return dst, bounds


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

    # No online basemap is used, so there will be no English map labels.
    m = folium.Map(
        location=center,
        zoom_start=6,
        tiles=None,
        control_scale=True,
        zoom_control=True,
        attribution_control=False
    )

    # White background without covering raster layers.
    m.get_root().html.add_child(
        folium.Element(
            """
            <style>
            .leaflet-container {
                background: #ffffff !important;
            }
            </style>
            """
        )
    )

    folium.GeoJson(
        boundary_wgs,
        name="Gansu Boundary",
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


@st.cache_data(show_spinner=False)
def build_yearly_table_cached(pre_items, gw_items, dem_path, boundary_zip_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        boundary_gdf = read_boundary_from_zip_path(boundary_zip_path, tmpdir)
        pre_dict = dict(pre_items)
        gw_dict = dict(gw_items)
        return build_yearly_table(pre_dict, gw_dict, dem_path, boundary_gdf)


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
                "Year": year,
                "Precipitation (mm)": pre_stats["Mean"],
                "Groundwater Value": gw_stats["Mean"],
                "Mean Elevation (m)": dem_stats["Mean"],
                "Minimum Elevation (m)": dem_stats["Minimum"],
                "Maximum Elevation (m)": dem_stats["Maximum"],
                "Valid Pixels": min(pre_stats["Valid Pixels"], gw_stats["Valid Pixels"])
            }
        )

    return pd.DataFrame(records)


def groundwater_pie_df(arr, threshold=0.1):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return pd.DataFrame({"Class": [], "Pixels": []})

    high = np.sum(valid > threshold)
    stable = np.sum((valid >= -threshold) & (valid <= threshold))
    low = np.sum(valid < -threshold)

    return pd.DataFrame(
        {
            "Class": [
                f"High Zone (> {threshold})",
                f"Stable Zone (-{threshold} to {threshold})",
                f"Low Zone (< -{threshold})"
            ],
            "Pixels": [high, stable, low]
        }
    )


def elevation_pie_df(arr):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return pd.DataFrame({"Class": [], "Pixels": []})

    q1, q2, q3 = np.nanpercentile(valid, [25, 50, 75])

    zones = [
        np.sum(valid <= q1),
        np.sum((valid > q1) & (valid <= q2)),
        np.sum((valid > q2) & (valid <= q3)),
        np.sum(valid > q3)
    ]

    names = [
        f"Low Elevation ≤ {q1:.0f} m",
        f"Lower-middle Elevation {q1:.0f}-{q2:.0f} m",
        f"Upper-middle Elevation {q2:.0f}-{q3:.0f} m",
        f"High Elevation > {q3:.0f} m"
    ]

    return pd.DataFrame({"Class": names, "Pixels": zones})


def demo_dataframe():
    years = list(range(2012, 2022))
    rainfall = [513.7, 562.1, 498.3, 610.5, 535.6, 505.2, 462.4, 438.9, 520.1, 488.2]
    groundwater = [0.05, 0.21, -0.02, 0.32, 0.08, 0.12, -0.10, 0.00, 0.16, -0.04]
    elevation = [1876.2] * 10

    return pd.DataFrame(
        {
            "Year": years,
            "Precipitation (mm)": rainfall,
            "Groundwater Value": groundwater,
            "Mean Elevation (m)": elevation,
            "Minimum Elevation (m)": [650.0] * 10,
            "Maximum Elevation (m)": [4800.0] * 10,
            "Valid Pixels": [10000] * 10
        }
    )


def demo_map():
    m = folium.Map(
        location=[38.5, 101.5],
        zoom_start=6,
        tiles=None,
        control_scale=True,
        zoom_control=True,
        attribution_control=False
    )

    m.get_root().html.add_child(
        folium.Element(
            """
            <style>
            .leaflet-container {
                background: #ffffff !important;
            }
            </style>
            """
        )
    )

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
        tooltip="Gansu Demo Area"
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def card_start(title):
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 5. Sidebar controls
# =========================================================
auto_info = find_auto_data()
auto_data_ready = (
    auto_info["boundary_zip"] is not None
    and len(auto_info["pre_dict"]) > 0
    and len(auto_info["gw_dict"]) > 0
    and auto_info["dem_path"] is not None
)

with st.sidebar:
    st.markdown("## Data Source")

    if auto_data_ready:
        default_mode = "GitHub Auto Loading"
    else:
        default_mode = "Demo Mode"

    data_mode = st.radio(
        "Select Data Mode",
        ["GitHub Auto Loading", "Manual Upload", "Demo Mode"],
        index=["GitHub Auto Loading", "Manual Upload", "Demo Mode"].index(default_mode)
    )

    if auto_data_ready:
        st.markdown(
            f"""
            <div class="success-box">
            GitHub data detected:<br>
            Boundary ZIP: {auto_info['counts']['boundary_zip']}<br>
            Precipitation TIF: {auto_info['counts']['precip_tif']}<br>
            Groundwater TIF: {auto_info['counts']['groundwater_tif']}<br>
            DEM TIF: {auto_info['counts']['dem_tif']}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="warning-box">
            Complete GitHub data was not detected. Please check the boundary, precip, groundwater and dem folders.
            </div>
            """,
            unsafe_allow_html=True
        )

    boundary_zip_upload = None
    pre_files_upload = None
    gw_files_upload = None
    dem_file_upload = None

    if data_mode == "Manual Upload":
        st.markdown("## Manual Upload")

        boundary_zip_upload = st.file_uploader(
            "1. Upload Boundary SHP ZIP",
            type=["zip"]
        )

        pre_files_upload = st.file_uploader(
            "2. Upload Precipitation TIF Files",
            type=["tif", "tiff"],
            accept_multiple_files=True
        )

        gw_files_upload = st.file_uploader(
            "3. Upload Groundwater TIF Files",
            type=["tif", "tiff"],
            accept_multiple_files=True
        )

        dem_file_upload = st.file_uploader(
            "4. Upload DEM TIF File",
            type=["tif", "tiff"]
        )

    st.markdown("## Display Settings")

    available_years = sorted(set(auto_info["pre_dict"].keys()) & set(auto_info["gw_dict"].keys()))
    if not available_years:
        available_years = list(range(2012, 2022))

    selected_year_sidebar = st.selectbox(
        "Select Year",
        available_years,
        index=len(available_years) - 1
    )

    layer_choice = st.radio(
        "Layer Selection",
        ["Precipitation", "Groundwater", "Elevation", "Overlay All"],
        index=0
    )

    opacity = st.slider("Layer Opacity", 0.1, 1.0, 0.75, 0.05)

    stable_threshold = st.number_input(
        "Groundwater Stability Threshold",
        min_value=0.0,
        value=0.10,
        step=0.05,
        help="Values within ±0.10 can be regarded as relatively stable."
    )


# =========================================================
# 6. Title area
# =========================================================
st.markdown("### Groundwater Observation Visualization Platform Based on Multi-source Data Fusion")
st.markdown(
    '<div class="small-subtitle">A multi-source visualization platform for groundwater, precipitation and elevation data in Gansu Province</div>',
    unsafe_allow_html=True
)


# =========================================================
# 7. Dashboard renderer
# =========================================================
def render_dashboard(boundary_gdf, pre_dict, gw_dict, dem_path, selected_year, source_note, use_cache_table=False, boundary_zip_path=None):
    common_years = sorted(set(pre_dict.keys()) & set(gw_dict.keys()))
    if len(common_years) == 0:
        st.error("No matching years were found between precipitation TIF files and groundwater TIF files.")
        st.stop()

    if selected_year not in common_years:
        selected_year = common_years[-1]

    fmap = make_base_map(boundary_gdf)

    current_pre_arr = None
    current_gw_arr = None
    current_dem_arr = None

    if layer_choice in ["Precipitation", "Overlay All"]:
        current_pre_arr = add_raster_layer(
            fmap, pre_dict[selected_year], boundary_gdf,
            f"{selected_year} Precipitation", "Blues", opacity
        )

    if layer_choice in ["Groundwater", "Overlay All"]:
        current_gw_arr = add_raster_layer(
            fmap, gw_dict[selected_year], boundary_gdf,
            f"{selected_year} Groundwater", "RdYlBu", opacity
        )

    if layer_choice in ["Elevation", "Overlay All"]:
        current_dem_arr = add_raster_layer(
            fmap, dem_path, boundary_gdf,
            "DEM Elevation", "terrain", opacity
        )

    folium.LayerControl(collapsed=False).add_to(fmap)

    if current_pre_arr is None:
        current_pre_arr, _, _, _ = crop_raster_by_boundary(pre_dict[selected_year], boundary_gdf)
    if current_gw_arr is None:
        current_gw_arr, _, _, _ = crop_raster_by_boundary(gw_dict[selected_year], boundary_gdf)
    if current_dem_arr is None:
        current_dem_arr, _, _, _ = crop_raster_by_boundary(dem_path, boundary_gdf)

    if use_cache_table and boundary_zip_path is not None:
        yearly_df = build_yearly_table_cached(
            tuple(pre_dict.items()),
            tuple(gw_dict.items()),
            dem_path,
            boundary_zip_path
        )
    else:
        yearly_df = build_yearly_table(pre_dict, gw_dict, dem_path, boundary_gdf)

    pre_stats = raster_stats(current_pre_arr)
    gw_stats = raster_stats(current_gw_arr)

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Matched Years</div><div class="metric-value">{len(common_years)}</div></div>',
            unsafe_allow_html=True
        )
    with top2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Current Year</div><div class="metric-value">{selected_year}</div></div>',
            unsafe_allow_html=True
        )
    with top3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Mean Precipitation</div><div class="metric-value">{pre_stats["Mean"]:.2f}</div></div>',
            unsafe_allow_html=True
        )
    with top4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Mean Groundwater Value</div><div class="metric-value">{gw_stats["Mean"]:.3f}</div></div>',
            unsafe_allow_html=True
        )

    row1_col1, row1_col2 = st.columns([1.06, 1.72])

    with row1_col1:
        card_start(f"Spatial Data Display in Gansu Province ({selected_year})")
        st_folium(fmap, width=None, height=430)
        st.markdown(f'<div class="footer-note">{source_note}</div>', unsafe_allow_html=True)
        card_end()

    with row1_col2:
        card_start("Multi-year Change Trend")
        chart_a, chart_b = st.columns(2)

        with chart_a:
            fig_rain = px.line(
                yearly_df,
                x="Year",
                y="Precipitation (mm)",
                markers=True,
                title="Annual Precipitation Trend"
            )
            fig_rain.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_rain, use_container_width=True)

        with chart_b:
            fig_gw = px.line(
                yearly_df,
                x="Year",
                y="Groundwater Value",
                markers=True,
                title="Annual Groundwater Trend"
            )
            fig_gw.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_gw.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gw, use_container_width=True)

        card_end()

    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        card_start(f"Precipitation vs Groundwater ({yearly_df['Year'].min()}-{yearly_df['Year'].max()})")
        if len(yearly_df) >= 2:
            corr = yearly_df[["Precipitation (mm)", "Groundwater Value"]].corr().iloc[0, 1]
        else:
            corr = np.nan

        fig_scatter = px.scatter(
            yearly_df,
            x="Precipitation (mm)",
            y="Groundwater Value",
            text="Year",
            trendline="ols" if len(yearly_df) >= 3 else None
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
        fig_scatter.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(f"Correlation Coefficient: {corr:.3f}" if np.isfinite(corr) else "Correlation Coefficient: insufficient samples")
        card_end()

    with row2_col2:
        card_start("Groundwater Classification Proportion")
        pie_df = groundwater_pie_df(current_gw_arr, stable_threshold)
        fig_pie = px.pie(
            pie_df,
            names="Class",
            values="Pixels",
            hole=0.45
        )
        fig_pie.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        card_end()

    row3_col1, row3_col2 = st.columns([1.38, 1])

    with row3_col1:
        card_start("Multi-source Data Fusion Table")
        show_df = yearly_df.copy()
        numeric_cols = show_df.select_dtypes(include=["float", "float64"]).columns
        show_df[numeric_cols] = show_df[numeric_cols].round(3)
        st.dataframe(show_df, use_container_width=True, height=310)

        csv = show_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "Export CSV",
            data=csv,
            file_name="gansu_multisource_fusion_statistics.csv",
            mime="text/csv"
        )
        card_end()

    with row3_col2:
        card_start("Elevation Zone Proportion")
        elev_df = elevation_pie_df(current_dem_arr)
        fig_elev = px.pie(
            elev_df,
            names="Class",
            values="Pixels"
        )
        fig_elev.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_elev, use_container_width=True)
        card_end()


# =========================================================
# 8. Main program
# =========================================================
try:
    if data_mode == "GitHub Auto Loading":
        if not auto_data_ready:
            st.error("GitHub data is incomplete. Please check the boundary, precip, groundwater and dem folders.")
            st.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            boundary_gdf = read_boundary_from_zip_path(auto_info["boundary_zip"], tmpdir)
            render_dashboard(
                boundary_gdf=boundary_gdf,
                pre_dict=auto_info["pre_dict"],
                gw_dict=auto_info["gw_dict"],
                dem_path=auto_info["dem_path"],
                selected_year=selected_year_sidebar,
                source_note="Built-in data from the GitHub repository is currently used.",
                use_cache_table=True,
                boundary_zip_path=auto_info["boundary_zip"]
            )

    elif data_mode == "Manual Upload":
        if not all([boundary_zip_upload, pre_files_upload, gw_files_upload, dem_file_upload]):
            st.info("Please upload the boundary ZIP, precipitation TIF files, groundwater TIF files and DEM TIF file from the sidebar.")
            st.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = safe_mkdir(os.path.join(tmpdir, "manual_upload"))
            boundary_gdf = read_boundary_from_uploaded_zip(boundary_zip_upload, work_dir)
            pre_dict = save_tif_files(pre_files_upload, os.path.join(work_dir, "pre"))
            gw_dict = save_tif_files(gw_files_upload, os.path.join(work_dir, "gw"))
            dem_path = save_uploaded_file(dem_file_upload, os.path.join(work_dir, "dem"))

            render_dashboard(
                boundary_gdf=boundary_gdf,
                pre_dict=pre_dict,
                gw_dict=gw_dict,
                dem_path=dem_path,
                selected_year=selected_year_sidebar,
                source_note="Manual uploaded data is currently used."
            )

    else:
        st.markdown(
            '<div class="warning-box">Demo Mode: simulated data is used for interface demonstration.</div>',
            unsafe_allow_html=True
        )

        yearly_df = demo_dataframe()
        selected_year = selected_year_sidebar if selected_year_sidebar in yearly_df["Year"].tolist() else yearly_df["Year"].iloc[-1]

        top1, top2, top3, top4 = st.columns(4)
        with top1:
            st.markdown('<div class="metric-card"><div class="metric-label">Matched Years</div><div class="metric-value">10</div></div>', unsafe_allow_html=True)
        with top2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Current Year</div><div class="metric-value">{selected_year}</div></div>', unsafe_allow_html=True)
        with top3:
            value = yearly_df.loc[yearly_df["Year"] == selected_year, "Precipitation (mm)"].iloc[0]
            st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Precipitation</div><div class="metric-value">{value:.1f} mm</div></div>', unsafe_allow_html=True)
        with top4:
            value = yearly_df.loc[yearly_df["Year"] == selected_year, "Groundwater Value"].iloc[0]
            st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Groundwater Value</div><div class="metric-value">{value:.2f}</div></div>', unsafe_allow_html=True)

        row1_col1, row1_col2 = st.columns([1.06, 1.72])

        with row1_col1:
            card_start(f"Spatial Data Display in Gansu Province ({selected_year})")
            st_folium(demo_map(), width=None, height=430)
            card_end()

        with row1_col2:
            card_start("Multi-year Change Trend")
            chart_a, chart_b = st.columns(2)

            with chart_a:
                fig_rain = px.line(yearly_df, x="Year", y="Precipitation (mm)", markers=True, title="Annual Precipitation Trend")
                fig_rain.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_rain, use_container_width=True)

            with chart_b:
                fig_gw = px.line(yearly_df, x="Year", y="Groundwater Value", markers=True, title="Annual Groundwater Trend")
                fig_gw.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_gw.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_gw, use_container_width=True)

            card_end()

        row2_col1, row2_col2 = st.columns([1, 1])

        with row2_col1:
            card_start("Precipitation vs Groundwater")
            corr = yearly_df[["Precipitation (mm)", "Groundwater Value"]].corr().iloc[0, 1]
            fig_scatter = px.scatter(yearly_df, x="Precipitation (mm)", y="Groundwater Value", text="Year", trendline="ols")
            fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
            fig_scatter.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"Correlation Coefficient: {corr:.3f}")
            card_end()

        with row2_col2:
            card_start("Groundwater Classification Proportion")
            pie_df = pd.DataFrame({"Class": ["High Zone", "Stable Zone", "Low Zone"], "Pixels": [2830, 4670, 2500]})
            fig_pie = px.pie(pie_df, names="Class", values="Pixels", hole=0.45)
            fig_pie.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            card_end()

        row3_col1, row3_col2 = st.columns([1.38, 1])

        with row3_col1:
            card_start("Multi-source Data Fusion Table")
            st.dataframe(yearly_df.round(3), use_container_width=True, height=310)
            card_end()

        with row3_col2:
            card_start("Elevation Zone Proportion")
            elev_df = pd.DataFrame({"Class": [">3000 m", "2000-3000 m", "1000-2000 m", "<1000 m"], "Pixels": [2750, 3510, 2480, 1260]})
            fig_elev = px.pie(elev_df, names="Class", values="Pixels")
            fig_elev.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_elev, use_container_width=True)
            card_end()

except Exception as e:
    st.error("Program error. Please check the coordinate system, file format and year information in the filenames.")
    st.exception(e)


# =========================================================
# 9. Method description
# =========================================================
with st.expander("Method Description"):
    st.markdown(
        """
        This platform supports three data modes:

        1. **GitHub Auto Loading**: automatically reads boundary ZIP, precipitation TIF files, groundwater TIF files and DEM TIF files from the repository;
        2. **Manual Upload**: users can upload spatial data from the sidebar;
        3. **Demo Mode**: simulated data is used when real data is not available.

        Recommended GitHub data structure:

        ```text
        boundary/
        └─ gansu_boundary.zip

        precip/
        ├─ pre_2012.tif
        ├─ pre_2013.tif
        └─ ...

        groundwater/
        ├─ GWs_2012.tif
        ├─ GWs_2013.tif
        └─ ...

        dem/
        └─ dem_gansu.tif
        ```

        Filenames should contain year information, such as `2012` or `2013`, so that the system can automatically match precipitation and groundwater TIF files by year.
        """
    )
