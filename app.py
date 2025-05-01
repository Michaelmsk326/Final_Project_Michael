import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

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

filtered_df = df[df["ZipCode"] == selected_zip]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Housing Price Information")
    if not filtered_df.empty and 'average_housing_cost_2023' in filtered_df.columns:
        housing_price = filtered_df['average_housing_cost_2023'].values[0]
        st.metric("Average Housing Price (2023)", f"${housing_price:,.2f}")
    else:
        st.write("No housing data available for this ZIP code.")

    st.subheader("School Quality")
    if not filtered_df.empty and 'Overall_Rating' in filtered_df.columns:
        school_rating = filtered_df['Overall_Rating'].values[0]
        if pd.notna(school_rating):
            st.metric("School Rating", f"{school_rating}")
            rating_desc = {
                1: "Level 1 - Highest Performance",
                2: "Level 2 - Good Standing",
                3: "Level 3 - Needs Improvement"
            }
            st.write(f"Rating Description: {rating_desc.get(school_rating, 'Not Rated')}")
        else:
            st.write("No school rating available for this ZIP code.")
    else:
        st.write("No school data available for this ZIP code.")

with col2:
    st.subheader("Crime Statistics")
    if not filtered_df.empty and 'Total_Crimes' in filtered_df.columns:
        crime_count = filtered_df['Total_Crimes'].values[0]
        st.metric("Total Crimes", f"{crime_count:,.0f}")
    else:
        st.write("No crime data available for this ZIP code.")

    if not filtered_df.empty and 'Low_Income_Percentage' in filtered_df.columns:
        low_income_pct = filtered_df['Low_Income_Percentage'].values[0]
        if pd.notna(low_income_pct):
            st.metric("Low Income Student Percentage", f"{low_income_pct:.1%}")
    elif 'Student_Count_Low_Income' in filtered_df.columns and 'Student_Count_Total' in filtered_df.columns:
        low_income = filtered_df['Student_Count_Low_Income'].values[0]
        total = filtered_df['Student_Count_Total'].values[0]
        if total > 0:
            low_income_pct = low_income / total
            st.metric("Low Income Student Percentage", f"{low_income_pct:.1%}")

st.subheader("ZIP Code Comparisons")
comparison_zips = st.multiselect(
    "Compare with other ZIP codes",
    options=[zip_code for zip_code in sorted(df["ZipCode"].unique()) if zip_code != selected_zip],
    max_selections=5
)

if comparison_zips:
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

st.subheader("Geographic Distribution")

if 'School_Latitude' in df.columns and 'School_Longitude' in df.columns:
    st.write("Explore the geographic distribution of housing costs, school ratings, and crime across Chicago.")
    
    map_df = df.dropna(subset=['School_Latitude', 'School_Longitude'])
    
    map_option = st.selectbox(
        "Select data to visualize on map:",
        ["Housing Costs", "School Ratings", "Crime Rates"]
    )
    
    if map_option == "Housing Costs" and 'average_housing_cost_2023' in df.columns:
        fig = px.scatter_mapbox(
            map_df,
            lat="School_Latitude",
            lon="School_Longitude",
            color="average_housing_cost_2023",
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=10,
            mapbox_style="carto-positron",
            hover_name="ZipCode",
            hover_data={"School_Latitude": False, "School_Longitude": False},
            title="Housing Costs Across Chicago"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif map_option == "School Ratings" and 'Overall_Rating' in df.columns:
        fig = px.scatter_mapbox(
            map_df,
            lat="School_Latitude",
            lon="School_Longitude",
            color="Overall_Rating",
            color_continuous_scale="RdYlGn_r",
            size_max=15,
            zoom=10,
            mapbox_style="carto-positron",
            hover_name="ZipCode",
            hover_data={"School_Latitude": False, "School_Longitude": False},
            title="School Ratings Across Chicago"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif map_option == "Crime Rates" and 'Total_Crimes' in df.columns:
        fig = px.scatter_mapbox(
            map_df,
            lat="School_Latitude",
            lon="School_Longitude",
            color="Total_Crimes",
            color_continuous_scale="Reds",
            size_max=15,
            zoom=10,
            mapbox_style="carto-positron",
            hover_name="ZipCode",
            hover_data={"School_Latitude": False, "School_Longitude": False},
            title="Crime Rates Across Chicago"
        )
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Data Insights")

if 'average_housing_cost_2023' in df.columns and 'Total_Crimes' in df.columns:
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

st.markdown("---")
st.markdown("**Data Sources**: Chicago Housing Data, School Quality Metrics, and Crime Statistics")
st.markdown("**Dashboard created by**: Your Name")


