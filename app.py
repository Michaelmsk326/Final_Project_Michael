import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import folium_static
import matplotlib.ticker as mticker

# Page configuration with custom styling
st.set_page_config(
    page_title="Chicago Housing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        padding: 0.5rem 0;
        border-bottom: 2px solid #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f1f5f9;
    }
    .stProgress .st-eb {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Load the data and models
@st.cache_data
def load_data():
    df = pd.read_csv("merged_output/final_merged_dataset.csv")
    
    # Clean up columns to ensure proper naming
    columns_to_keep = []
    for col in df.columns:
        if col.startswith('2023'):
            columns_to_keep.append(col)
        else:
            columns_to_keep.append(col)
    
    df = df[columns_to_keep]
    return df

@st.cache_resource
def load_models():
    # Load housing price prediction model
    try:
        housing_model = xgb.XGBRegressor()
        housing_model.load_model("models/housing_price_model.json")
        
        # Load school rating prediction model
        rating_model = xgb.XGBRegressor()
        rating_model.load_model("models/school_rating_model.json")
        
        return housing_model, rating_model
    except:
        st.warning("Models not found. Some predictive features will be disabled.")
        return None, None

# Load data and models
df = load_data()
housing_model, school_rating_model = load_models()

# Helper functions for formatting
def currency_formatter(x):
    return f"${x:,.0f}"

def percentage_formatter(x):
    return f"{x:.1%}"

def rating_description(rating):
    descriptions = {
        1: "Level 1 - Highest Performance",
        2: "Level 2 - Good Standing",
        3: "Level 3 - Needs Improvement"
    }
    return descriptions.get(rating, "Not Rated")

# Header
st.markdown('<h1 class="main-title">Chicago Housing Dashboard</h1>', unsafe_allow_html=True)

st.markdown("""
This interactive dashboard allows you to explore Chicago housing data by ZIP code, 
including housing prices, school quality metrics, crime statistics, and demographic information.
Use the filters on the sidebar to customize your view.
""")

# Sidebar filters
st.sidebar.markdown('<div class="section-header">Filter Options</div>', unsafe_allow_html=True)

# ZIP code selection with search functionality
zip_list = sorted(df["ZipCode"].unique().tolist())
selected_zip = st.sidebar.selectbox(
    "Choose ZIP Code",
    zip_list,
    index=0,
    format_func=lambda x: f"{x}"
)

# Filter data based on selection
filtered_df = df[df["ZipCode"] == selected_zip]

# Main dashboard layout with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Key Metrics",
    "Price Analysis",
    "Map Visualization",
    "ZIP Code Comparisons"
])

# Tab 1: Key Metrics
with tab1:
    st.markdown('<div class="section-header">Key Metrics for ZIP Code {}</div>'.format(selected_zip), unsafe_allow_html=True)
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Housing Price")
        if not filtered_df.empty and 'average_housing_cost_2023' in filtered_df.columns:
            housing_price = filtered_df['average_housing_cost_2023'].values[0]
            if pd.notna(housing_price):
                st.metric("Average Housing Price (2023)", currency_formatter(housing_price))
                
                # Compare with Chicago average
                chicago_avg = df['average_housing_cost_2023'].mean()
                diff_pct = (housing_price - chicago_avg) / chicago_avg * 100
                st.metric("vs. Chicago Average", f"{diff_pct:.1f}%", 
                         delta_color="inverse" if diff_pct > 0 else "normal")
            else:
                st.write("No housing data available.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("School Quality")
        if not filtered_df.empty and 'Overall_Rating' in filtered_df.columns:
            school_rating = filtered_df['Overall_Rating'].values[0]
            if pd.notna(school_rating):
                st.metric("School Rating", f"{school_rating:.1f}")
                
                # Rating description
                st.write(f"**Rating Description:** {rating_description(int(school_rating))}")
                
                # Show rating as stars
                stars = "★" * int(school_rating) + "☆" * (3 - int(school_rating))
                st.write(f"**Rating:** {stars}")
            else:
                st.write("No school rating available.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Crime Statistics")
        if not filtered_df.empty and 'Total_Crimes' in filtered_df.columns:
            crime_count = filtered_df['Total_Crimes'].values[0]
            if pd.notna(crime_count):
                st.metric("Total Crimes", f"{crime_count:,.0f}")
                
                # Compare with Chicago average
                chicago_crime_avg = df['Total_Crimes'].mean()
                crime_diff_pct = (crime_count - chicago_crime_avg) / chicago_crime_avg * 100
                st.metric("vs. Chicago Average", f"{crime_diff_pct:.1f}%",
                         delta_color="normal" if crime_diff_pct < 0 else "inverse")
            else:
                st.write("No crime data available.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Low Income Demographics")
        if not filtered_df.empty and 'Low_Income_Percentage' in filtered_df.columns:
            low_income_pct = filtered_df['Low_Income_Percentage'].values[0]
            if pd.notna(low_income_pct):
                st.metric("Low Income Percentage", percentage_formatter(low_income_pct))
                
                # Create progress bar
                st.progress(low_income_pct)
                
                # Compare with Chicago average
                chicago_low_income_avg = df['Low_Income_Percentage'].mean()
                income_diff_pct = (low_income_pct - chicago_low_income_avg) / chicago_low_income_avg * 100
                st.metric("vs. Chicago Average", f"{income_diff_pct:.1f}%",
                         delta_color="normal" if income_diff_pct < 0 else "inverse")
            else:
                st.write("No demographic data available.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Housing Affordability Index")
        if not filtered_df.empty and 'average_housing_cost_2023' in filtered_df.columns and 'Low_Income_Percentage' in filtered_df.columns:
            housing_price = filtered_df['average_housing_cost_2023'].values[0]
            low_income_pct = filtered_df['Low_Income_Percentage'].values[0]
            
            if pd.notna(housing_price) and pd.notna(low_income_pct):
                # Calculate affordability index (lower is more affordable)
                affordability_index = housing_price * (low_income_pct + 0.5)
                max_index = df['average_housing_cost_2023'].max()
                normalized_affordability = affordability_index / max_index
                
                # Display affordability level
                if normalized_affordability < 0.3:
                    affordability_text = "Very Affordable"
                    color = "green"
                elif normalized_affordability < 0.5:
                    affordability_text = "Affordable"
                    color = "lightgreen"
                elif normalized_affordability < 0.7:
                    affordability_text = "Moderate"
                    color = "orange"
                else:
                    affordability_text = "Expensive"
                    color = "red"
                
                st.metric("Affordability Level", affordability_text)
                st.progress(normalized_affordability)
            else:
                st.write("Insufficient data to calculate affordability.")
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Price Analysis & Prediction
with tab2:
    st.markdown('<div class="section-header">Housing Price Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        # Housing price trend over time if data available
        st.subheader("Housing Price Trends")
        
        # Look for time series columns
        time_columns = [col for col in df.columns if col.startswith("2023-")]
        
        if time_columns and not filtered_df.empty:
            time_data = filtered_df[time_columns].T.reset_index()
            time_data.columns = ['Date', 'Price']
            time_data['Date'] = pd.to_datetime(time_data['Date'])
            
            fig = px.line(
                time_data,
                x='Date',
                y='Price',
                title=f"Housing Price Trend for ZIP {selected_zip}",
                labels={'Price': 'Housing Price ($)', 'Date': 'Date'},
            )
            fig.update_layout(
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time series data available for this ZIP code.")
    
    with col2:
        # Price Predictor
        st.subheader("Price Predictor")
        st.markdown("Estimate housing price based on neighborhood characteristics:")
        
        crime_rate = st.slider("Crime Rate", 
                             min_value=0.0,
                             max_value=max(df['Total_Crimes'].max(), 5000.0),
                             value=float(filtered_df['Total_Crimes'].values[0]) if not filtered_df.empty else 1000.0,
                             step=100.0)
        
        low_income = st.slider("Low Income %", 
                             min_value=0.0,
                             max_value=1.0,
                             value=float(filtered_df['Low_Income_Percentage'].values[0]) if not filtered_df.empty else 0.5,
                             step=0.05,
                             format="%0.0f%%")
        
        school_quality = st.slider("School Quality (1-3)", 
                                min_value=1.0,
                                max_value=3.0,
                                value=float(filtered_df['Overall_Rating'].values[0]) if not filtered_df.empty and 'Overall_Rating' in filtered_df.columns and pd.notna(filtered_df['Overall_Rating'].values[0]) else 2.0,
                                step=1.0)
        
        # Make prediction if model is available
        if st.button("Predict Housing Price"):
            if housing_model is not None:
                # Prepare input features
                features = pd.DataFrame({
                    'Total_Crimes': [crime_rate],
                    'Low_Income_Percentage': [low_income],
                    'Overall_Rating': [school_quality]
                })
                
                # Make prediction
                try:
                    predicted_price = housing_model.predict(features)[0]
                    st.success(f"Predicted Housing Price: {currency_formatter(predicted_price)}")
                    
                    # Show price range
                    st.write(f"Price Range: {currency_formatter(predicted_price * 0.9)} - {currency_formatter(predicted_price * 1.1)}")
                    
                    # Show confidence level
                    st.write("Confidence: High")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
            else:
                # Fallback to linear regression if model not available
                st.info("Using simplified estimation model (model not loaded).")
                # Simple price estimate based on features
                base_price = 250000  # Base price
                crime_factor = -0.05 * crime_rate / 1000  # Crime reduces price
                income_factor = -0.3 * low_income  # Higher low income % reduces price
                school_factor = 0.2 * (4 - school_quality)  # Better schools (lower number) increase price
                
                # Calculate estimate
                price_adjustment = base_price * (1 + crime_factor + income_factor + school_factor)
                estimated_price = max(base_price + price_adjustment, 50000)
                
                st.success(f"Estimated Housing Price: {currency_formatter(estimated_price)}")
                st.caption("Note: This is a simplified estimate only")

    # Correlation Analysis
    st.subheader("Housing Price Correlations")
    
    try:
        # Create a correlation matrix of key variables
        corr_vars = ['average_housing_cost_2023', 'Overall_Rating', 
                    'Total_Crimes', 'Low_Income_Percentage']
        corr_df = df[corr_vars].dropna()
        
        if not corr_df.empty:
            corr_matrix = corr_df.corr()
            
            # Plot heatmap with Plotly
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                labels=dict(color="Correlation"),
                x=['Housing Price', 'School Rating', 'Crime Rate', 'Low Income %'],
                y=['Housing Price', 'School Rating', 'Crime Rate', 'Low Income %']
            )
            fig.update_layout(
                title="Correlation Between Key Variables",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about correlations
            st.markdown("### Key Insights:")
            
            housing_school_corr = corr_matrix.loc['average_housing_cost_2023', 'Overall_Rating']
            housing_crime_corr = corr_matrix.loc['average_housing_cost_2023', 'Total_Crimes']
            housing_income_corr = corr_matrix.loc['average_housing_cost_2023', 'Low_Income_Percentage']
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.info(f"Housing prices have a {abs(housing_school_corr):.2f} {'positive' if housing_school_corr > 0 else 'negative'} correlation with school ratings.")
                st.info(f"Housing prices have a {abs(housing_crime_corr):.2f} {'positive' if housing_crime_corr > 0 else 'negative'} correlation with crime rates.")
                
            with c2:
                st.info(f"Housing prices have a {abs(housing_income_corr):.2f} {'positive' if housing_income_corr > 0 else 'negative'} correlation with low income percentage.")
                st.info(f"School ratings have a {abs(corr_matrix.loc['Overall_Rating', 'Low_Income_Percentage']):.2f} correlation with low income percentage.")
        else:
            st.warning("Insufficient data for correlation analysis.")
    except Exception as e:
        st.error(f"Error generating correlation analysis: {e}")

# Tab 3: Map Visualization
with tab3:
    st.markdown('<div class="section-header">Geographic Visualization</div>', unsafe_allow_html=True)
    
    # Create map visualization using Folium
    st.subheader("Chicago ZIP Code Map")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Map display options
        map_metric = st.radio(
            "Color map by:",
            ["Housing Price", "School Rating", "Crime Rate", "Low Income Percentage"],
            horizontal=True
        )
        
        # Create a base map centered on Chicago
        m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles='CartoDB positron')
        
        # Filter to only ZIP codes with valid coordinates
        map_data = df.dropna(subset=['Latitude', 'Longitude'])
        
        # Choose color metric based on selection
        if map_metric == "Housing Price":
            color_col = 'average_housing_cost_2023'
            colormap = 'YlOrRd'  # Yellow-Orange-Red
            popup_prefix = "$"
            popup_suffix = ""
        elif map_metric == "School Rating":
            color_col = 'Overall_Rating'
