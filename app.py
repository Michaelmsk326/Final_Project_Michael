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

st.set_page_config(
    page_title="Chicago Housing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #F5F5F5;
}

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

@st.cache_data
def load_data():
    try:
        file_path = "merged_output/final_merged_dataset.csv"
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
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
        required_columns = ['ZipCode', 'average_housing_cost_2023', 'Overall_Rating',
                           'Total_Crimes', 'Low_Income_Percentage', 'Latitude', 'Longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Missing required columns in dataset: {', '.join(missing_columns)}")
            for col in missing_columns:
                if col == 'ZipCode':
                    df[col] = df.index + 60600
                elif col in ['Latitude', 'Longitude']:
                    df[col] = 41.8781 if col == 'Latitude' else -87.6298
                else:
                    df[col] = np.nan
        
        columns_to_keep = []
        for col in df.columns:
            if col.startswith('2023') or col in required_columns:
                columns_to_keep.append(col)
        
        if not all(col in df.columns for col in columns_to_keep):
            missing = [col for col in columns_to_keep if col not in df.columns]
            st.warning(f"Some time series columns are missing: {missing}")
        
        df = df[[col for col in columns_to_keep if col in df.columns]]
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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

df = load_data()
models = load_models()
chicago_geojson = load_geojson()

housing_model = models.get('housing')
school_rating_model = models.get('school')

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

st.markdown('<h1 class="main-title">Chicago Housing Dashboard</h1>', unsafe_allow_html=True)

st.markdown("""
This interactive dashboard allows you to explore Chicago housing data by ZIP code, 
including housing prices, school quality metrics, crime statistics, and demographic information.
Use the filters on the sidebar to customize your view.
""")

st.sidebar.markdown('<div class="section-header">Filter Options</div>', unsafe_allow_html=True)

zip_list = sorted(df["ZipCode"].unique().tolist())
selected_zip = st.sidebar.selectbox(
    "Choose ZIP Code",
    zip_list,
    index=0,
    format_func=lambda x: f"{x}"
)

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

st.sidebar.markdown("### Download Data")
if st.sidebar.button("Download Filtered Data"):
    filtered_download = df[
        (df['ZipCode'] == selected_zip) &
        (df['average_housing_cost_2023'] >= price_range[0]) &
        (df['average_housing_cost_2023'] <= price_range[1])
    ]
    
    if school_rating_filter:
        filtered_download = filtered_download[filtered_download['Overall_Rating'].isin(school_rating_filter)]
    
    csv = filtered_download.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"chicago_housing_zip_{selected_zip}.csv",
        mime="text/csv"
    )

filtered_df = df[df["ZipCode"] == selected_zip]

filtered_df = filtered_df[
    (filtered_df['average_housing_cost_2023'] >= price_range[0]) &
    (filtered_df['average_housing_cost_2023'] <= price_range[1])
]

if school_rating_filter:
    filtered_df = filtered_df[filtered_df['Overall_Rating'].isin(school_rating_filter)]

tab1, tab2, tab3, tab4 = st.tabs([
    "Key Metrics",
    "Price Analysis",
    "Map Visualization",
    "ZIP Code Comparisons"
])

with tab1:
    st.markdown('<div class="section-header">Key Metrics for ZIP Code {}</div>'.format(selected_zip), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Housing Price")
        
        if not filtered_df.empty and 'average_housing_cost_2023' in filtered_df.columns:
            housing_price = filtered_df['average_housing_cost_2023'].values[0]
            if pd.notna(housing_price):
                st.metric("Average Housing Price (2023)", currency_formatter(housing_price))
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
                st.write(f"**Rating Description:** {rating_description(int(school_rating))}")
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
                chicago_crime_avg = df['Total_Crimes'].mean()
                crime_diff_pct = (crime_count - chicago_crime_avg) / chicago_crime_avg * 100
                st.metric("vs. Chicago Average", f"{crime_diff_pct:.1f}%",
                         delta_color="normal" if crime_diff_pct < 0 else "inverse")
            else:
                st.write("No crime data available.")
        else:
            st.write("No crime data available for this ZIP code.")
            
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header">Map Visualization</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        map_metric = st.radio(
            "Select Metric for Map",
            ["Housing Price", "School Rating", "Crime Rate", "Low Income"]
        )
        
        if map_metric == "Housing Price":
            color_col = "average_housing_cost_2023"
            popup_prefix = "$"
            popup_suffix = ""
            colormap = "Viridis"
            legend_title = "Housing Price ($)"
        elif map_metric == "School Rating":
            color_col = "Overall_Rating"
            popup_prefix = ""
            popup_suffix = " (lower is better)"
            colormap = "RdYlGn_r"
            legend_title = "School Rating"
        elif map_metric == "Crime Rate":
            color_col = "Total_Crimes"
            popup_prefix = ""
            popup_suffix = " crimes"
            colormap = "Reds"
            legend_title = "Crime Count"
        else:
            color_col = "Low_Income_Percentage"
            popup_prefix = ""
            popup_suffix = "%"
            colormap = "Blues"
            legend_title = "Low Income %"
    
    with col2:
        if chicago_geojson:
            m = folium.Map(
                location=[41.8781, -87.6298],
                zoom_start=10,
                tiles="CartoDB positron"
            )
            
            folium.GeoJson(
                chicago_geojson,
                name="Chicago ZIP Codes",
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.1,
                }
            ).add_to(m)
            
            # Merge geojson with data
            for feature in chicago_geojson['features']:
                zip_code = feature['properties']['zip']
                matched_rows = df[df['ZipCode'] == int(zip_code)]
                
                if not matched_rows.empty and color_col in matched_rows.columns:
                    row = matched_rows.iloc[0]
                    
                    # Fix for ValueError: Unknown format code 'f' for object of type 'str'
                    if pd.notna(row[color_col]):
                        if isinstance(row[color_col], (int, float)):
                            popup_content = f"""
                            <strong>ZIP: {row['ZipCode']}</strong><br>
                            {map_metric}: {popup_prefix}{row[color_col]:.2f}{popup_suffix}<br>
                            """
                        else:
                            popup_content = f"""
                            <strong>ZIP: {row['ZipCode']}</strong><br>
                            {map_metric}: {popup_prefix}{row[color_col]}{popup_suffix}<br>
                            """
                        
                        folium.Marker(
                            location=[row['Latitude'], row['Longitude']],
                            popup=folium.Popup(popup_content, max_width=300),
                            icon=folium.Icon(color='blue', icon='info-sign')
                        ).add_to(m)
            
            folium_static(m, width=800, height=500)
        else:
            st.error("GeoJSON data not available. Cannot display map.")

