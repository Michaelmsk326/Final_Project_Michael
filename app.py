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

with tabs[0]:  # Housing Analysis Tab
    st.header("Housing Price Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        # Housing cost distribution
        fig = px.histogram(df, x="average_housing_cost_2023", nbins=20,
                          title="Housing Cost Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Housing cost map
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
        # School ratings map
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

with tabs[2]:  # Crime Statistics Tab
  
    
with tabs[3]:  # Correlations Tab
   
    
with tabs[4]:  # Prediction Models Tab
   

with tabs[4]:  # Prediction Models Tab
    
    st.header("Housing Price Prediction Models")
    
    st.subheader("Feature Importance for Housing Price Prediction")
    

    fig, ax = plt.figure(figsize=(12, 8))
    xgb.plot_importance(model_housing, max_num_features=10, ax=ax)
    st.pyplot(fig)
    
  
    st.subheader("Model Performance")
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("Mean Absolute Error", f"${mae:.2f}")
    
    with metrics_col2:
        st.metric("Root Mean Squared Error", f"${rmse:.2f}")
    

    st.subheader("Actual vs Predicted Housing Costs")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Housing Cost')
    plt.ylabel('Predicted Housing Cost')
    plt.title('Actual vs Predicted Housing Costs')
    st.pyplot(fig)
    

    st.header("School Rating Prediction Models")


fig = plt.figure(figsize=(12, 8))
plt.scatter(
    map_df['Total_Crimes'],
    map_df['average_housing_cost_2023'],
    c=map_df['Overall_Rating'],
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
    map_df['Low_Income_Percentage'] * 100,
    map_df['average_housing_cost_2023'],
    c=map_df['Overall_Rating'],
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

def create_chicago_map(data, feature, title, cmap, format_func=None):
    fig, ax = plt.subplots(figsize=(15, 13))
    
    ax.set_title(title, fontsize=16)
    
    scatter = ax.scatter(
        data['School_Longitude'],
        data['School_Latitude'],
        c=data[feature],
        cmap=cmap,
        alpha=0.8,
        s=100,
        edgecolors='black',
        linewidths=0.5
    )
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    if format_func:
        cbar = plt.colorbar(scatter, cax=cax, format=format_func)
    else:
        cbar = plt.colorbar(scatter, cax=cax)
        
    cbar.set_label(feature, fontsize=14)
    
  
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron)
    
    return fig

