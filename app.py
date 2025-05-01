import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from PIL import Image

# Page configuration
st.set_page_config(page_title="Chicago Housing Dashboard", layout="wide")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("merged_output/final_merged_dataset.csv")
    return df

df = load_data()

# Title and introduction
st.title("Chicago Housing Dashboard")
st.markdown("""
This dashboard allows you to explore Chicago housing data by ZIP code, 
including housing prices, school quality metrics, and crime statistics.
""")

# Sidebar for ZIP code selection
st.sidebar.header("Filter Options")
selected_zip = st.sidebar.selectbox("Choose ZIP Code", sorted(df["ZipCode"].unique()))

# Filter data based on selection
filtered_df = df[df["ZipCode"] == selected_zip]

# Main dashboard content
col1, col2 = st.columns(2)

with col1:
    # Housing Price Section
    st.subheader("Housing Price Information")
    if not filtered_df.empty and 'average_housing_cost_2023' in filtered_df.columns:
        housing_price = filtered_df['average_housing_cost_2023'].values[0]
        st.metric("Average Housing Price (2023)", f"${housing_price:,.2f}")
    else:
        st.write("No housing data available for this ZIP code.")

    # School Quality Section
    st.subheader("School Quality")
    if not filtered_df.empty and 'Overall_Rating' in filtered_df.columns:
        school_rating = filtered_df['Overall_Rating'].values[0]
        if pd.notna(school_rating):
            st.metric("School Rating", f"{school_rating}")
            
            # Convert rating to descriptive text
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
    # Crime Statistics Section
    st.subheader("Crime Statistics")
    if not filtered_df.empty and 'Total_Crimes' in filtered_df.columns:
        crime_count = filtered_df['Total_Crimes'].values[0]
        st.metric("Total Crimes", f"{crime_count:,.0f}")
    else:
        st.write("No crime data available for this ZIP code.")
    
    # Low Income Percentage (if available)
    if not filtered_df.empty and 'Low_Income_Percentage' in filtered_df.columns:
        low_income_pct = filtered_df['Low_Income_Percentage'].values[0]
        if pd.notna(low_income_pct):
            st.metric("Low Income Student Percentage", f"{low_income_pct:.1%}")

# Display PNG files section
st.subheader("Visualizations")
png_directory = "png_files"
if os.path.exists(png_directory):
    png_files = [f for f in os.listdir(png_directory) if f.endswith('.png')]
    if png_files:
        for png_file in png_files:
            image_path = os.path.join(png_directory, png_file)
            try:
                image = Image.open(image_path)
                st.image(image, caption=png_file.replace('.png', '').replace('_', ' ').title())
            except Exception as e:
                st.error(f"Error loading image {png_file}: {e}")
    else:
        st.write("No PNG files found in the directory.")
else:
    st.write("PNG files directory not found.")
    
# Visualization section
st.subheader("ZIP Code Comparisons")

# Allow comparing with other ZIP codes
comparison_zips = st.multiselect(
    "Compare with other ZIP codes",
    options=[zip_code for zip_code in sorted(df["ZipCode"].unique()) if zip_code != selected_zip],
    max_selections=5
)

if comparison_zips:
    comparison_data = df[df["ZipCode"].isin([selected_zip] + comparison_zips)]
    
    # Housing price comparison
    if 'average_housing_cost_2023' in comparison_data.columns:
        fig1 = px.bar(
            comparison_data, 
            x="ZipCode", 
            y="average_housing_cost_2023",
            title="Housing Price Comparison",
            labels={"average_housing_cost_2023": "Average Housing Price ($)", "ZipCode": "ZIP Code"}
        )
        st.plotly_chart(fig1)
    
    # Crime comparison
    if 'Total_Crimes' in comparison_data.columns:
        fig2 = px.bar(
            comparison_data, 
            x="ZipCode", 
            y="Total_Crimes",
            title="Crime Comparison",
            labels={"Total_Crimes": "Total Crimes", "ZipCode": "ZIP Code"}
        )
        st.plotly_chart(fig2)
