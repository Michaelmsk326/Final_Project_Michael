import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

st.set_page_config(page_title="Chicago Housing Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("merged_output/final_merged_dataset.csv")
    return df

df = load_data()

st.title("Chicago Housing Dashboard")
st.markdown("""
This dashboard allows you to explore Chicago housing data by ZIP code, 
including housing prices, school quality metrics, and crime statistics.
""")

st.sidebar.header("Filter Options")

selected_zip = st.sidebar.selectbox("Choose ZIP Code", sorted(df["ZipCode"].unique()))

price_range = st.sidebar.slider(
    "Housing Price Range ($)",
    float(df["average_housing_cost_2023"].min()),
    float(df["average_housing_cost_2023"].max()),
    (float(df["average_housing_cost_2023"].min()), float(df["average_housing_cost_2023"].max()))
)

school_rating_options = st.sidebar.multiselect(
    "School Rating", 
    options=[1, 2, 3],
    default=[1, 2, 3],
    help="1 = Highest Performance, 3 = Needs Improvement"
)

crime_threshold = st.sidebar.slider(
    "Maximum Crime Rate",
    0,
    int(df["Total_Crimes"].max()),
    int(df["Total_Crimes"].max())
)

income_range = st.sidebar.slider(
    "Low Income Percentage",
    0.0,
    1.0,
    (0.0, 1.0),
    0.05,
    format="%d%%"
)

filtered_df = df[df["ZipCode"] == selected_zip]

st.header("Chicago Housing Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Housing Price", f"${df['average_housing_cost_2023'].mean():,.2f}")
    
with col2:
    avg_rating = df['Overall_Rating'].mean()
    st.metric("Average School Rating", f"{avg_rating:.2f}")
    
with col3:
    st.metric("Average Crime Rate", f"{df['Total_Crimes'].mean():,.0f}")

st.subheader("Chicago ZIP Code Overview")

if 'School_Latitude' in df.columns and 'School_Longitude' in df.columns:
    fig = px.scatter_mapbox(
        df,
        lat="School_Latitude",
        lon="School_Longitude",
        color="average_housing_cost_2023",
        size="SchoolCount" if "SchoolCount" in df.columns else None,
        color_continuous_scale="Viridis",
        zoom=9,
        mapbox_style="carto-positron",
        hover_name="ZipCode",
        title="Chicago Housing Overview"
    )
    st.plotly_chart(fig, use_container_width=True)

tabs = st.tabs(["Housing Analysis", "School Quality", "Crime Statistics", "Correlations", "Prediction Models"])

with tabs[0]:
    st.header("Housing Price Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="average_housing_cost_2023", nbins=20,
                          title="Housing Cost Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'School_Latitude' in df.columns and 'School_Longitude' in df.columns:
            fig = px.scatter_mapbox(
                df,
                lat="School_Latitude",
                lon="School_Longitude",
                color="average_housing_cost_2023",
                color_continuous_scale="Viridis",
                zoom=10,
                mapbox_style="carto-positron",
                title="Housing Costs Across Chicago"
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[1]:  
    st.header("School Quality Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df['Overall_Rating'].value_counts().reset_index(),
            x="index",
            y="Overall_Rating",
            title="Distribution of School Ratings",
            labels={"index": "Rating", "Overall_Rating": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'School_Latitude' in df.columns and 'School_Longitude' in df.columns:
            fig = px.scatter_mapbox(
                df,
                lat="School_Latitude",
                lon="School_Longitude",
                color="Overall_Rating",
                color_continuous_scale="RdYlGn_r",
                zoom=10,
                mapbox_style="carto-positron",
                title="School Ratings Across Chicago"
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Crime Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Total_Crimes", nbins=20,
                          title="Crime Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'School_Latitude' in df.columns and 'School_Longitude' in df.columns:
            fig = px.scatter_mapbox(
                df,
                lat="School_Latitude",
                lon="School_Longitude",
                color="Total_Crimes",
                color_continuous_scale="Reds",
                zoom=10,
                mapbox_style="carto-positron",
                title="Crime Rates Across Chicago"
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.header("Data Correlations")
    
    correlation = df[['average_housing_cost_2023', 'Total_Crimes']].corr().iloc[0, 1]
    st.write(f"Correlation between housing costs and crime rates: {correlation:.2f}")
    
    fig = px.scatter(
        df,
        x="Total_Crimes",
        y="average_housing_cost_2023",
        hover_name="ZipCode",
        color="Overall_Rating" if "Overall_Rating" in df.columns else None,
        title="Relationship Between Crime Rates and Housing Costs"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(
        df['Total_Crimes'],
        df['average_housing_cost_2023'],
        c=df['Overall_Rating'],
        cmap='viridis',
        alpha=0.7,
        s=80
    )
    plt.colorbar(label='School Rating')
    plt.title('Housing Cost vs. Crime Rate (colored by School Rating)')
    plt.xlabel('Crime Rate')
    plt.ylabel('Average Housing Cost ($)')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(
        df['Low_Income_Percentage'] * 100,
        df['average_housing_cost_2023'],
        c=df['Overall_Rating'],
        cmap='viridis',
        alpha=0.7,
        s=80
    )
    plt.colorbar(label='School Rating')
    plt.title('Housing Cost vs. Low Income Percentage (colored by School Rating)')
    plt.xlabel('Low Income Percentage (%)')
    plt.ylabel('Average Housing Cost ($)')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

with tabs[4]:
    st.header("Housing Price Prediction Models")
    
    st.subheader("Feature Importance for Housing Price Prediction")
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    st.pyplot(fig)
    
    st.subheader("Model Performance")
    metrics_col1, metrics_col2 = st.columns(2)
    
    st.subheader("Actual vs Predicted Housing Costs")
    fig = plt.figure(figsize=(10, 6))
    plt.xlabel('Actual Housing Cost')
    plt.ylabel('Predicted Housing Cost')
    plt.title('Actual vs Predicted Housing Costs')
    st.pyplot(fig)
    
    st.header("School Rating Prediction Models")
    
    st.subheader("Feature Importance for School Rating Prediction")
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    st.pyplot(fig)

st.sidebar.markdown("---")
comparison_zips = st.sidebar.multiselect(
    "Compare with other ZIP codes",
    options=[zip_code for zip_code in sorted(df["ZipCode"].unique()) if zip_code != selected_zip],
    max_selections=5
)

if comparison_zips:
    st.header("ZIP Code Comparisons")
    comparison_data = df[df["ZipCode"].isin([selected_zip] + comparison_zips)]
    
    if 'average_housing_cost_2023' in comparison_data.columns:
        fig1 = px.bar(
            comparison_data,
            x="ZipCode",
            y="average_housing_cost_2023",
            title="Housing Price Comparison",
            labels={"average_housing_cost_2023": "Average Housing Price ($)", "ZipCode": "ZIP Code"}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    if 'Total_Crimes' in comparison_data.columns:
        fig2 = px.bar(
            comparison_data,
            x="ZipCode",
            y="Total_Crimes",
            title="Crime Comparison",
            labels={"Total_Crimes": "Total Crimes", "ZipCode": "ZIP Code"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    if 'Overall_Rating' in comparison_data.columns:
        fig3 = px.bar(
            comparison_data,
            x="ZipCode",
            y="Overall_Rating",
            title="School Rating Comparison (Lower is Better)",
            labels={"Overall_Rating": "School Rating", "ZipCode": "ZIP Code"}
        )
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("**Data Sources**: Chicago Housing Data, School Quality Metrics, and Crime Statistics")
st.markdown("**Dashboard created by**: Your Name")


