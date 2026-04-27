# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# 1. 页面设置
# =========================
st.set_page_config(
    page_title="地下水-降雨-高程可视化平台",
    layout="wide"
)

st.title("地下水—降雨—高程多源数据可视化平台")

st.write(
    "本平台用于上传地下水观测数据、降雨数据和高程数据，"
    "实现空间分布、时间变化和关联分析可视化。"
)

st.success("如需数据处理、GIS制图或平台定制，请加 QQ：3916748449")

# =========================
# 2. 上传数据
# =========================
st.sidebar.header("一、上传数据")

uploaded_file = st.sidebar.file_uploader(
    "上传 CSV / Excel 数据表",
    type=["csv", "xlsx", "xls"]
)

st.sidebar.markdown("""
### 数据表字段要求

必须包含以下字段：

- `well_id`：井号
- `lon`：经度
- `lat`：纬度
- `date`：日期
- `water_level`：地下水水位
- `depth`：地下水埋深
- `rainfall`：降雨量
- `elevation`：高程
""")

# =========================
# 3. 读取数据
# =========================
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(file)

    df["date"] = pd.to_datetime(df["date"])
    return df

# =========================
# 4. 未上传时显示模板
# =========================
if uploaded_file is None:
    st.info("请在左侧上传 CSV 或 Excel 数据表。")

    demo = pd.DataFrame({
        "well_id": ["GW001", "GW001", "GW002", "GW002", "GW003", "GW003"],
        "lon": [102.712, 102.712, 102.735, 102.735, 102.760, 102.760],
        "lat": [25.048, 25.048, 25.060, 25.060, 25.070, 25.070],
        "date": ["2024-01", "2024-02", "2024-01", "2024-02", "2024-01", "2024-02"],
        "water_level": [12.35, 12.10, 10.82, 10.60, 9.85, 9.60],
        "depth": [5.60, 5.85, 6.20, 6.42, 7.10, 7.35],
        "rainfall": [32.5, 18.6, 32.5, 18.6, 32.5, 18.6],
        "elevation": [1890, 1890, 1925, 1925, 1980, 1980]
    })

    st.subheader("示例数据格式")
    st.dataframe(demo, use_container_width=True)

    st.download_button(
        label="下载示例 CSV 模板",
        data=demo.to_csv(index=False, encoding="utf-8-sig"),
        file_name="groundwater_rainfall_elevation_template.csv",
        mime="text/csv"
    )

    st.stop()

df = load_data(uploaded_file)

# =========================
# 5. 字段检查
# =========================
required_cols = [
    "well_id",
    "lon",
    "lat",
    "date",
    "water_level",
    "depth",
    "rainfall",
    "elevation"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error("你的数据缺少以下字段：")
    st.write(missing_cols)
    st.stop()

# =========================
# 6. 侧边栏筛选
# =========================
st.sidebar.header("二、筛选条件")

well_list = sorted(df["well_id"].dropna().unique())

selected_well = st.sidebar.selectbox(
    "选择观测井",
    ["全部"] + well_list
)

date_min = df["date"].min()
date_max = df["date"].max()

selected_date = st.sidebar.date_input(
    "选择时间范围",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

filtered = df.copy()

if selected_well != "全部":
    filtered = filtered[filtered["well_id"] == selected_well]

if isinstance(selected_date, tuple) and len(selected_date) == 2:
    start_date = pd.to_datetime(selected_date[0])
    end_date = pd.to_datetime(selected_date[1])

    filtered = filtered[
        (filtered["date"] >= start_date) &
        (filtered["date"] <= end_date)
    ]

# =========================
# 7. 基础统计信息
# =========================
st.subheader("一、基础统计信息")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("记录数", len(filtered))
col2.metric("观测井数量", filtered["well_id"].nunique())

if len(filtered) > 0:
    col3.metric("平均地下水水位/m", round(filtered["water_level"].mean(), 2))
    col4.metric("平均降雨量/mm", round(filtered["rainfall"].mean(), 2))
    col5.metric("平均高程/m", round(filtered["elevation"].mean(), 2))
else:
    col3.metric("平均地下水水位/m", "无数据")
    col4.metric("平均降雨量/mm", "无数据")
    col5.metric("平均高程/m", "无数据")

# =========================
# 8. 地图空间分布
# =========================
st.subheader("二、地下水观测井空间分布图")

if len(filtered) > 0:
    well_avg = filtered.groupby("well_id", as_index=False).agg({
        "lon": "first",
        "lat": "first",
        "water_level": "mean",
        "depth": "mean",
        "rainfall": "mean",
        "elevation": "first"
    })

    fig_map = px.scatter_mapbox(
        well_avg,
        lat="lat",
        lon="lon",
        size="water_level",
        color="elevation",
        hover_name="well_id",
        hover_data={
            "water_level": ":.2f",
            "depth": ":.2f",
            "rainfall": ":.1f",
            "elevation": True
        },
        zoom=8,
        height=550,
        title="地下水井点空间分布图"
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("当前筛选条件下没有地图数据。")

# =========================
# 9. 数据表
# =========================
st.subheader("三、多源融合数据表")
st.dataframe(filtered, use_container_width=True)

# =========================
# 10. 时间变化分析
# =========================
st.subheader("四、时间变化分析")

if len(filtered) > 0:
    monthly = filtered.groupby("date", as_index=False).agg({
        "water_level": "mean",
        "depth": "mean",
        "rainfall": "mean",
        "elevation": "mean"
    })

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = px.line(
            monthly,
            x="date",
            y="water_level",
            markers=True,
            title="地下水平均水位时间变化",
            labels={
                "date": "时间",
                "water_level": "地下水水位/m"
            }
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            monthly,
            x="date",
            y="rainfall",
            title="降雨量时间变化",
            labels={
                "date": "时间",
                "rainfall": "降雨量/mm"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig3 = px.line(
            monthly,
            x="date",
            y="depth",
            markers=True,
            title="地下水埋深时间变化",
            labels={
                "date": "时间",
                "depth": "地下水埋深/m"
            }
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.bar(
            monthly,
            x="date",
            y="elevation",
            title="平均高程统计",
            labels={
                "date": "时间",
                "elevation": "高程/m"
            }
        )
        st.plotly_chart(fig4, use_container_width=True)

else:
    st.warning("当前筛选条件下没有时间序列数据。")

# =========================
# 11. 关联分析
# =========================
st.subheader("五、地下水—降雨—高程关联分析")

if len(filtered) > 0:
    col_e, col_f = st.columns(2)

    with col_e:
        fig5 = px.scatter(
            filtered,
            x="rainfall",
            y="water_level",
            size="depth",
            color="elevation",
            hover_data=["well_id", "date"],
            title="降雨量与地下水水位关系",
            labels={
                "rainfall": "降雨量/mm",
                "water_level": "地下水水位/m",
                "depth": "地下水埋深/m",
                "elevation": "高程/m"
            }
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        fig6 = px.scatter(
            filtered,
            x="elevation",
            y="depth",
            color="water_level",
            size="rainfall",
            hover_data=["well_id", "date"],
            title="高程与地下水埋深关系",
            labels={
                "elevation": "高程/m",
                "depth": "地下水埋深/m",
                "water_level": "地下水水位/m",
                "rainfall": "降雨量/mm"
            }
        )
        st.plotly_chart(fig6, use_container_width=True)

    col_g, col_h = st.columns(2)

    with col_g:
        fig7 = px.scatter(
            filtered,
            x="rainfall",
            y="depth",
            color="elevation",
            hover_data=["well_id", "date"],
            title="降雨量与地下水埋深关系",
            labels={
                "rainfall": "降雨量/mm",
                "depth": "地下水埋深/m",
                "elevation": "高程/m"
            }
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_h:
        well_level = filtered.groupby("well_id", as_index=False).agg({
            "water_level": "mean",
            "rainfall": "mean",
            "elevation": "first",
            "depth": "mean"
        })

        fig8 = px.bar(
            well_level,
            x="well_id",
            y="water_level",
            color="elevation",
            title="不同观测井平均地下水水位",
            labels={
                "well_id": "观测井",
                "water_level": "平均地下水水位/m",
                "elevation": "高程/m"
            }
        )
        st.plotly_chart(fig8, use_container_width=True)

# =========================
# 12. 相关系数
# =========================
st.subheader("六、简单相关系数")

if len(filtered) >= 2:
    corr1 = filtered["rainfall"].corr(filtered["water_level"])
    corr2 = filtered["rainfall"].corr(filtered["depth"])
    corr3 = filtered["elevation"].corr(filtered["depth"])
    corr4 = filtered["elevation"].corr(filtered["water_level"])

    st.write(f"降雨量与地下水水位相关系数：**{corr1:.3f}**")
    st.write(f"降雨量与地下水埋深相关系数：**{corr2:.3f}**")
    st.write(f"高程与地下水埋深相关系数：**{corr3:.3f}**")
    st.write(f"高程与地下水水位相关系数：**{corr4:.3f}**")

    st.info("相关系数接近 1 表示正相关，接近 -1 表示负相关，接近 0 表示线性关系不明显。")
else:
    st.warning("数据量不足，无法计算相关系数。")

# =========================
# 13. 下载筛选结果
# =========================
st.subheader("七、导出筛选结果")

st.download_button(
    label="下载筛选后的 CSV 数据",
    data=filtered.to_csv(index=False, encoding="utf-8-sig"),
    file_name="filtered_groundwater_rainfall_elevation.csv",
    mime="text/csv"
)

# =========================
# 14. 说明
# =========================
st.subheader("八、平台说明")

st.write(
    "本平台以地下水观测井为基本空间单元，将地下水水位、埋深、降雨量和高程数据进行融合，"
    "通过地图、数据表、折线图、柱状图和散点图展示地下水与环境因子之间的关系。"
)
