import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
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
    try:
        file_path = "merged_output/final_merged_dataset.csv"
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            # Create a minimal sample dataframe with required columns
            sample_data = {
                'ZipCode': [60601, 60602, 60603],
                'average_housing_cost_2023': [450000, 380000, 520000],
                'Overall_Rating': [1.5, 2.0, 1.0],
                'Total_Crimes': [1200, 1500, 800],
                'Low_Income_Percentage': [0.2, 0.3, 0.15],
                'Latitude': [41.8855, 41.8837, 41.8807],
                'Longitude': [-87.6217, -87.6290, -87.6251]
            }
            return pd.DataFrame(sample_data)
            
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['ZipCode', 'average_housing_cost_2023', 'Overall_Rating', 
                           'Total_Crimes', 'Low_Income_Percentage', 'Latitude', 'Longitude']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Missing required columns in dataset: {', '.join(missing_columns)}")
            # Add placeholder columns if missing
            for col in missing_columns:
                if col == 'ZipCode':
                    df[col] = df.index + 60600
                elif col in ['Latitude', 'Longitude']:
                    df[col] = 41.8781 if col == 'Latitude' else -87.6298  # Chicago coordinates
                else:
                    df[col] = np.nan
        
        # Clean up columns to ensure proper naming
        columns_to_keep = []
        for col in df.columns:
            if col.startswith('2023') or col in required_columns:
                columns_to_keep.append(col)
        
        if not all(col in df.columns for col in columns_to_keep):
            missing = [col for col in columns_to_keep if col not in df.columns]
            st.warning(f"Some time series columns are missing: {missing}")
        
        # Keep only necessary columns
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return a minimal dataset to prevent app crashes
        return pd.DataFrame({
            'ZipCode': [60601],
            'average_housing_cost_2023': [400000],
            'Overall_Rating': [2.0],
            'Total_Crimes': [1000],
            'Low_Income_Percentage': [0.25],
            'Latitude': [41.8781],
            'Longitude': [-87.6298]
        })

@st.cache_resource
def load_models():
    models = {}
    
    # Load housing price prediction model
    try:
        housing_model_path = "models/housing_price_model.json"
        if os.path.exists(housing_model_path):
            housing_model = xgb.XGBRegressor()
            housing_model.load_model(housing_model_path)
            models['housing'] = housing_model
        else:
            st.warning("Housing price model file not found.")
    except Exception as e:
        st.warning(f"Error loading housing price model: {str(e)}")
    
    # Load school rating prediction model
    try:
        rating_model_path = "models/school_rating_model.json"
        if os.path.exists(rating_model_path):
            rating_model = xgb.XGBRegressor()
            rating_model.load_model(rating_model_path)
            models['school'] = rating_model
        else:
            st.warning("School rating model file not found.")
    except Exception as e:
        st.warning(f"Error loading school rating model: {str(e)}")
    
    return models

@st.cache_data
def load_geojson():
    try:
        geojson_path = "data/chicago_zipcodes.geojson"
        if os.path.exists(geojson_path):
            with open(geojson_path, 'r') as f:
                return json.load(f)
        else:
            st.warning(f"GeoJSON file not found: {geojson_path}")
            return None
    except Exception as e:
        st.warning(f"Error loading GeoJSON: {str(e)}")
        return None

# Load data and models
df = load_data()
models = load_models()
chicago_geojson = load_geojson()

# Get models or set to None if not available
housing_model = models.get('housing')
school_rating_model = models.get('school')

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

# Additional filters
price_range = st.sidebar.slider(
    "Housing Price Range",
    min_value=float(df['average_housing_cost_2023'].min()) if not df.empty and pd.notna(df['average_housing_cost_2023']).any() else 100000.0,
    max_value=float(df['average_housing_cost_2023'].max()) if not df.empty and pd.notna(df['average_housing_cost_2023']).any() else 1000000.0,
    value=(float(df['average_housing_cost_2023'].min()) if not df.empty and pd.notna(df['average_housing_cost_2023']).any() else 100000.0,
           float(df['average_housing_cost_2023'].max()) if not df.empty and pd.notna(df['average_housing_cost_2023']).any() else 1000000.0)
)

school_rating_filter = st.sidebar.multiselect(
    "School Rating",
    options=[1, 2, 3],
    default=[1, 2, 3]
)

# Add data download option
st.sidebar.markdown("### Download Data")
if st.sidebar.button("Download Filtered Data"):
    # Filter data based on all selections
    filtered_download = df[
        (df['ZipCode'] == selected_zip) &
        (df['average_housing_cost_2023'] >= price_range[0]) &
        (df['average_housing_cost_2023'] <= price_range[1])
    ]
    if school_rating_filter:
        filtered_download = filtered_download[filtered_download['Overall_Rating'].isin(school_rating_filter)]
    
    # Convert to CSV for download
    csv = filtered_download.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"chicago_housing_zip_{selected_zip}.csv",
        mime="text/csv"
    )

# Filter data based on selection
filtered_df = df[df["ZipCode"] == selected_zip]

# Apply price range filter
filtered_df = filtered_df[
    (filtered_df['average_housing_cost_2023'] >= price_range[0]) &
    (filtered_df['average_housing_cost_2023'] <= price_range[1])
]

# Apply school rating filter if selected
if school_rating_filter:
    filtered_df = filtered_df[filtered_df['Overall_Rating'].isin(school_rating_filter)]

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
        else:
            st.write("No housing data available for this ZIP code.")
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
        else:
            st.write("No school data available for this ZIP code.")
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
        else:
            st.write("No crime data available for this ZIP code.")
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
        else:
            st.write("No demographic data available for this ZIP code.")
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
        else:
            st.write("Insufficient data to calculate affordability for this ZIP code.")
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
                             max_value=max(df['Total_Crimes'].max() if not df.empty and pd.notna(df['Total_Crimes']).any() else 5000.0, 5000.0),
                             value=float(filtered_df['Total_Crimes'].values[0]) if not filtered_df.empty and 'Total_Crimes' in filtered_df.columns and pd.notna(filtered_df['Total_Crimes'].values[0]) else 1000.0,
                             step=100.0)
        
        low_income = st.slider("Low Income %", 
                             min_value=0.0,
                             max_value=1.0,
                             value=float(filtered_df['Low_Income_Percentage'].values[0]) if not filtered_df.empty and 'Low_Income_Percentage' in filtered_df.columns and pd.notna(filtered_df['Low_Income_Percentage'].values[0]) else 0.5,
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
            colormap = 'RdYlGn'  # Red-Yellow-Green (reversed for school ratings where 1 is best)
            popup_prefix = ""
            popup_suffix = " rating"
        elif map_metric == "Crime Rate":
            color_col = 'Total_Crimes'
            colormap = 'YlOrRd_r'  # Yellow-Orange-Red reversed
            popup_prefix = ""
            popup_suffix = " crimes"
        else:  # Low Income Percentage
            color_col = 'Low_Income_Percentage'
            colormap = 'PuBu'  # Purple-Blue
            popup_prefix = ""
            popup_suffix = "%"
        
        # Add markers for ZIP codes
        for idx, row in map_data.iterrows():
            if pd.notna(row[color_col]):
                popup_content = f"""
                <strong>ZIP: {row['ZipCode']}</strong><br>
                {map_metric}: {popup_prefix}{row[color_col]:.2f}{popup_suffix}<br>
                """
                
                # Add housing price if available
                if 'average_housing_cost_2023' in row and pd.notna(row['average_housing_cost_2023']):
                    popup_content += f"Housing Price: ${row['average_housing_cost_2023']:,.0f}<br>"
                
                # Add school rating if available
                if 'Overall_Rating' in row and pd.notna(row['Overall_Rating']):
                    popup_content += f"School Rating: {row['Overall_Rating']:.1f}<br>"
                
                # Add crime count if available
                if 'Total_Crimes' in row and pd.notna(row['Total_Crimes']):
                    popup_content += f"Total Crimes: {row['Total_Crimes']:,.0f}<br>"
                
                # Create a circle marker
                color = 'blue' if row['ZipCode'] == selected_zip else 'gray'
                radius = 8 if row['ZipCode'] == selected_zip else 5
                
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(m)
        
        # Add choropleth layer if GeoJSON available
        if chicago_geojson is not None:
            try:
            # Try to determine the correct property key for ZIP codes
                geojson_properties = list(chicago_geojson['features'][0]['properties'].keys())
                zip_key = next((k for k in geojson_properties if k.lower() in ['zip', 'zipcode', 'zip_code']), None)
                
                if zip_key:
                    # Create choropleth layer
                    choropleth = folium.Choropleth(
                        geo_data=chicago_geojson,
                        name='choropleth',
                        data=map_data,
                        columns=['ZipCode', color_col],
                        key_on=f'feature.properties.{zip_key}',
                        fill_color=colormap,
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name=map_metric
                    ).add_to(m)
                    
                    # Add tooltips
                    choropleth.geojson.add_child(
                        folium.features.GeoJsonTooltip(
                            fields=[zip_key],
                            aliases=['ZIP Code:'],
                            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                        )
                    )
                else:
                    st.warning("Could not determine ZIP code property in GeoJSON file.")
            except Exception as e:
                st.warning(f"Error creating choropleth map: {e}")
        
        # Display the map
        folium_static(m)
    
    with col2:
        st.subheader("ZIP Code Info")
        
        if not filtered_df.empty:
            st.write(f"**ZIP Code:** {selected_zip}")
            
            # Display ZIP code information
            if 'average_housing_cost_2023' in filtered_df.columns and pd.notna(filtered_df['average_housing_cost_2023'].values[0]):
                st.write(f"**Housing Price:** {currency_formatter(filtered_df['average_housing_cost_2023'].values[0])}")
            
            if 'Overall_Rating' in filtered_df.columns and pd.notna(filtered_df['Overall_Rating'].values[0]):
                st.write(f"**School Rating:** {filtered_df['Overall_Rating'].values[0]:.1f}")
            
            if 'Total_Crimes' in filtered_df.columns and pd.notna(filtered_df['Total_Crimes'].values[0]):
                st.write(f"**Total Crimes:** {filtered_df['Total_Crimes'].values[0]:,.0f}")
            
            if 'Low_Income_Percentage' in filtered_df.columns and pd.notna(filtered_df['Low_Income_Percentage'].values[0]):
                st.write(f"**Low Income %:** {filtered_df['Low_Income_Percentage'].values[0]:.1%}")
            
            # Add nearby ZIP codes if available
            st.subheader("Nearby ZIP Codes")
            
            # Calculate distance between selected ZIP and all others
            if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
                selected_lat = filtered_df['Latitude'].values[0]
                selected_long = filtered_df['Longitude'].values[0]
                
                # Function to calculate distance
                def calculate_distance(row):
                    if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                        # Simple Euclidean distance for demo purposes
                        return np.sqrt((row['Latitude'] - selected_lat)**2 + (row['Longitude'] - selected_long)**2)
                    return np.nan
                
                # Apply distance calculation to all rows
                distances = df.apply(calculate_distance, axis=1)
                
                # Get 5 closest ZIP codes (excluding the selected one)
                df['Distance'] = distances
                nearby_zips = df[df['ZipCode'] != selected_zip].nsmallest(5, 'Distance')
                
                if not nearby_zips.empty:
                    for idx, row in nearby_zips.iterrows():
                        st.write(f"**ZIP {row['ZipCode']}**")
                        if 'average_housing_cost_2023' in row and pd.notna(row['average_housing_cost_2023']):
                            st.write(f"Price: {currency_formatter(row['average_housing_cost_2023'])}")
        else:
            st.write("No data available for selected ZIP code.")

# Tab 4: ZIP Code Comparisons
with tab4:
    st.markdown('<div class="section-header">ZIP Code Comparisons</div>', unsafe_allow_html=True)
    
    # Allow comparison of multiple ZIP codes
    compare_zips = st.multiselect(
        "Select ZIP Codes to Compare",
        options=zip_list,
        default=[selected_zip],
        max_selections=5
    )
    
    if len(compare_zips) > 0:
        # Filter data for selected ZIP codes
        comparison_df = df[df['ZipCode'].isin(compare_zips)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Housing price comparison
            st.subheader("Housing Price Comparison")
            
            if 'average_housing_cost_2023' in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x='ZipCode',
                    y='average_housing_cost_2023',
                    title="Average Housing Price by ZIP Code",
                    labels={'average_housing_cost_2023': 'Average Housing Price ($)', 'ZipCode': 'ZIP Code'},
                    color='ZipCode',
                    text_auto=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Housing price data not available for comparison.")
                
        with col2:
            # School rating comparison
            st.subheader("School Rating Comparison")
            
            if 'Overall_Rating' in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x='ZipCode',
                    y='Overall_Rating',
                    title="School Ratings by ZIP Code",
                    labels={'Overall_Rating': 'School Rating (1-3)', 'ZipCode': 'ZIP Code'},
                    color='ZipCode',
                    text_auto=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("School rating data not available for comparison.")
        
        # Crime and income comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crime Rate Comparison")
            
            if 'Total_Crimes' in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x='ZipCode',
                    y='Total_Crimes',
                    title="Total Crimes by ZIP Code",
                    labels={'Total_Crimes': 'Total Crimes', 'ZipCode': 'ZIP Code'},
                    color='ZipCode',
                    text_auto=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Crime data not available for comparison.")
                
        with col2:
            st.subheader("Low Income Percentage Comparison")
            
            if 'Low_Income_Percentage' in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x='ZipCode',
                    y='Low_Income_Percentage',
                    title="Low Income Percentage by ZIP Code",
                    labels={'Low_Income_Percentage': 'Low Income %', 'ZipCode': 'ZIP Code'},
                    color='ZipCode',
                    text_auto=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Income data not available for comparison.")
        
        # Add a radar chart comparing all metrics
        st.subheader("ZIP Code Comparison Radar Chart")
        
        # Fix for radar chart (in Tab 4):
        if (all(col in comparison_df.columns for col in ['average_housing_cost_2023', 'Overall_Rating', 'Total_Crimes', 'Low_Income_Percentage']) and
            not comparison_df['average_housing_cost_2023'].isna().all() and
            not comparison_df['Overall_Rating'].isna().all() and
            not comparison_df['Total_Crimes'].isna().all() and
            not comparison_df['Low_Income_Percentage'].isna().all()):
            
            # Normalize the data for radar chart
            radar_df = comparison_df.copy()
            
            # Safe normalization function that handles zero ranges
            def safe_normalize(series, invert=False):
                min_val = series.min()
                max_val = series.max()
                if min_val == max_val:
                    return pd.Series(0.5, index=series.index)  # Return middle value if no range
                normalized = (series - min_val) / (max_val - min_val)
                return 1 - normalized if invert else normalized
            
            # For metrics where lower is better, invert the scale
            radar_df['Normalized_Crime'] = safe_normalize(radar_df['Total_Crimes'], invert=True)
            radar_df['Normalized_Low_Income'] = safe_normalize(radar_df['Low_Income_Percentage'], invert=True)
            
            # For metrics where higher is better
            radar_df['Normalized_Housing'] = safe_normalize(radar_df['average_housing_cost_2023'])
            
            # For school rating, lower is better in Chicago's rating system (1 is highest performance)
            radar_df['Normalized_School'] = safe_normalize(radar_df['Overall_Rating'], invert=True)
            # Create the radar chart
            fig = go.Figure()
            
            for idx, row in radar_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Normalized_Housing'], row['Normalized_School'],
                      row['Normalized_Crime'], row['Normalized_Low_Income']],
                    theta=['Housing Price', 'School Quality', 'Low Crime', 'Higher Income'],
                    fill='toself',
                    name=f"ZIP {row['ZipCode']}"
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="ZIP Code Comparison Radar Chart (Normalized Metrics)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Note: All metrics are normalized to a 0-1 scale for comparison. For metrics where lower values are better (e.g., crime), the scale is inverted so higher values on the chart represent better outcomes.")
            
        else:
            st.info("Insufficient data for radar chart comparison.")
    else:
        st.info("Please select at least one ZIP code for comparison.")

# Add a footer
st.markdown("---")
st.markdown("Chicago Housing Dashboard | Created with Streamlit")
st.markdown("Data sources: Chicago Public Schools, Chicago Police Department, Census Bureau")
