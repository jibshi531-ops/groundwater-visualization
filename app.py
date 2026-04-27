# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="地下水多源数据可视化平台",
    layout="wide"
)

st.title("地下水—气象—地形多源数据可视化平台")
st.write("上传地下水、降水、气温、高程等融合数据后，自动生成地图、图表和关联分析结果。")

st.sidebar.header("一、上传数据")

uploaded_file = st.sidebar.file_uploader(
    "上传 CSV / Excel 数据表",
    type=["csv", "xlsx", "xls"]
)

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(file)
    df["date"] = pd.to_datetime(df["date"])
    return df

if uploaded_file is None:
    st.info("请在左侧上传 CSV 或 Excel 数据表。")

    demo = pd.DataFrame({
        "well_id": ["GW001", "GW001", "GW002", "GW002"],
        "lon": [102.712, 102.712, 102.735, 102.735],
        "lat": [25.048, 25.048, 25.060, 25.060],
        "date": ["2024-01", "2024-02", "2024-01", "2024-02"],
        "water_level": [12.35, 12.10, 10.82, 10.60],
        "depth": [5.60, 5.85, 6.20, 6.42],
        "rainfall": [32.5, 18.6, 32.5, 18.6],
        "temperature": [15.2, 16.1, 15.2, 16.1],
        "elevation": [1890, 1890, 1925, 1925],
        "slope": [3.2, 3.2, 6.7, 6.7],
        "terrain_type": ["平原", "平原", "丘陵", "丘陵"]
    })

    st.subheader("示例数据格式")
    st.dataframe(demo, use_container_width=True)

    st.download_button(
        label="下载示例 CSV 模板",
        data=demo.to_csv(index=False, encoding="utf-8-sig"),
        file_name="groundwater_template.csv",
        mime="text/csv"
    )

    st.stop()

df = load_data(uploaded_file)

required_cols = [
    "well_id", "lon", "lat", "date", "water_level", "depth",
    "rainfall", "temperature", "elevation", "slope", "terrain_type"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error("数据缺少以下字段，请检查表头：")
    st.write(missing_cols)
    st.stop()

st.sidebar.header("二、筛选条件")

selected_well = st.sidebar.selectbox(
    "选择观测井",
    ["全部"] + sorted(df["well_id"].dropna().unique())
)

selected_terrain = st.sidebar.selectbox(
    "选择地形类型",
    ["全部"] + sorted(df["terrain_type"].dropna().unique())
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

if selected_terrain != "全部":
    filtered = filtered[filtered["terrain_type"] == selected_terrain]

if isinstance(selected_date, tuple) and len(selected_date) == 2:
    start_date = pd.to_datetime(selected_date[0])
    end_date = pd.to_datetime(selected_date[1])
    filtered = filtered[
        (filtered["date"] >= start_date) &
        (filtered["date"] <= end_date)
    ]

st.subheader("一、基础统计信息")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("记录数", len(filtered))
col2.metric("观测井数量", filtered["well_id"].nunique())

if len(filtered) > 0:
    col3.metric("平均水位/m", round(filtered["water_level"].mean(), 2))
    col4.metric("平均降水/mm", round(filtered["rainfall"].mean(), 2))
    col5.metric("平均气温/℃", round(filtered["temperature"].mean(), 2))

st.subheader("二、地下水观测井空间分布图")

if len(filtered) > 0:
    well_avg = filtered.groupby("well_id", as_index=False).agg({
        "lon": "first",
        "lat": "first",
        "water_level": "mean",
        "depth": "mean",
        "rainfall": "mean",
        "temperature": "mean",
        "elevation": "first",
        "slope": "first",
        "terrain_type": "first"
    })

    fig_map = px.scatter_mapbox(
        well_avg,
        lat="lat",
        lon="lon",
        color="terrain_type",
        size="water_level",
        hover_name="well_id",
        hover_data={
            "water_level": ":.2f",
            "depth": ":.2f",
            "rainfall": ":.1f",
            "temperature": ":.1f",
            "elevation": True,
            "slope": True
        },
        zoom=8,
        height=520,
        title="地下水井点空间分布"
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_map, use_container_width=True)

st.subheader("三、多源融合数据表")
st.dataframe(filtered, use_container_width=True)

st.subheader("四、时间变化分析")

if len(filtered) > 0:
    monthly = filtered.groupby("date", as_index=False).agg({
        "water_level": "mean",
        "depth": "mean",
        "rainfall": "mean",
        "temperature": "mean"
    })

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = px.line(
            monthly,
            x="date",
            y="water_level",
            markers=True,
            title="地下水平均水位时间变化",
            labels={"date": "时间", "water_level": "地下水水位/m"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            monthly,
            x="date",
            y="rainfall",
            title="月降水量变化",
            labels={"date": "时间", "rainfall": "降水量/mm"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig3 = px.line(
            monthly,
            x="date",
            y="temperature",
            markers=True,
            title="平均气温时间变化",
            labels={"date": "时间", "temperature": "气温/℃"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.line(
            monthly,
            x="date",
            y="depth",
            markers=True,
            title="地下水埋深时间变化",
            labels={"date": "时间", "depth": "地下水埋深/m"}
        )
        st.plotly_chart(fig4, use_container_width=True)

st.subheader("五、地下水与环境因子关联分析")

if len(filtered) > 0:
    col_e, col_f = st.columns(2)

    with col_e:
        fig5 = px.scatter(
            filtered,
            x="rainfall",
            y="water_level",
            color="terrain_type",
            size="depth",
            hover_data=["well_id", "date"],
            title="降水量与地下水水位关系",
            labels={
                "rainfall": "降水量/mm",
                "water_level": "地下水水位/m",
                "terrain_type": "地形类型",
                "depth": "地下水埋深/m"
            }
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        fig6 = px.scatter(
            filtered,
            x="elevation",
            y="depth",
            color="terrain_type",
            hover_data=["well_id", "date"],
            title="高程与地下水埋深关系",
            labels={
                "elevation": "高程/m",
                "depth": "地下水埋深/m",
                "terrain_type": "地形类型"
            }
        )
        st.plotly_chart(fig6, use_container_width=True)

    col_g, col_h = st.columns(2)

    with col_g:
        fig7 = px.scatter(
            filtered,
            x="temperature",
            y="water_level",
            color="terrain_type",
            hover_data=["well_id", "date"],
            title="气温与地下水水位关系",
            labels={
                "temperature": "气温/℃",
                "water_level": "地下水水位/m",
                "terrain_type": "地形类型"
            }
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_h:
        terrain_count = filtered.drop_duplicates("well_id").groupby(
            "terrain_type",
            as_index=False
        ).agg({"well_id": "count"})

        terrain_count.rename(columns={"well_id": "井点数量"}, inplace=True)

        fig8 = px.pie(
            terrain_count,
            names="terrain_type",
            values="井点数量",
            title="不同地形类型观测井占比"
        )
        st.plotly_chart(fig8, use_container_width=True)

st.subheader("六、简单相关系数")

if len(filtered) >= 2:
    st.write(f"降水量与地下水水位相关系数：**{filtered['rainfall'].corr(filtered['water_level']):.3f}**")
    st.write(f"气温与地下水水位相关系数：**{filtered['temperature'].corr(filtered['water_level']):.3f}**")
    st.write(f"高程与地下水埋深相关系数：**{filtered['elevation'].corr(filtered['depth']):.3f}**")

st.subheader("七、导出筛选结果")

st.download_button(
    label="下载筛选后的 CSV 数据",
    data=filtered.to_csv(index=False, encoding="utf-8-sig"),
    file_name="filtered_groundwater_data.csv",
    mime="text/csv"
)

st.subheader("八、分析结论说明")
st.write("本平台支持上传地下水、降水、气温、高程和地形等多源融合数据，并通过地图、数据表、折线图、柱状图、散点图和饼图进行综合可视化展示。")