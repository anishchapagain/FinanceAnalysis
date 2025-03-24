import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

def analyze_dataframe(df):
    """
    Main function to analyze a pandas DataFrame with various data types.
    Dispatches to specialized functions based on data type and analysis needs.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame to analyze
    """
    st.subheader("**Analysis Dashboard**")
    
    # Display basic dataframe overview
    # display_dataframe_overview(df)
    
    # Identify column types
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = identify_datetime_columns(df)
    
    # Remove identified datetime columns from categorical list
    categorical_columns = [col for col in categorical_columns if col not in datetime_columns]
    

    analysis_type = st.selectbox(
        "**Select Analysis Method**",
        ("Exploratory Data Analysis", "Univariate Analysis", "Bivariate Analysis", 
                "Correlation Analysis", "Time Series Analysis"),
    )
    
    if analysis_type == "Exploratory Data Analysis":
        perform_eda(df, numeric_columns, categorical_columns, datetime_columns)
    
    elif analysis_type == "Univariate Analysis":
        column_type = st.selectbox(
            "*Select Column Type:*",
            ("Numeric", "Categorical", "Datetime"),
        )
        
        if column_type == "Numeric" and numeric_columns:
            analyze_numeric_column(df, numeric_columns)
        elif column_type == "Categorical" and categorical_columns:
            analyze_categorical_column(df, categorical_columns)
        elif column_type == "Datetime" and datetime_columns:
            analyze_datetime_column(df, datetime_columns)
        else:
            st.warning(f"No {column_type.lower()} columns found in the dataframe.")
    
    elif analysis_type == "Bivariate Analysis":
        perform_bivariate_analysis(df, numeric_columns, categorical_columns, datetime_columns)
    
    elif analysis_type == "Correlation Analysis":
        perform_correlation_analysis(df, numeric_columns)
    
    elif analysis_type == "Time Series Analysis":
        perform_time_series_analysis(df, datetime_columns, numeric_columns)


def display_dataframe_overview(df):
    """Display basic information about the dataframe"""
    st.subheader("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isna().sum().sum())
    
    # Show a sample of the data
    with st.expander("Data Preview"):
        st.dataframe(df.head(), use_container_width=True)
        
        # Display column types
        st.write("Column Data Types:")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True)


def identify_datetime_columns(df):
    """Identify columns that contain datetime data"""
    # Find columns with datetime dtype
    datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Check for string columns that may contain dates
    for col in df.select_dtypes(include=['object']).columns:
        if col not in datetime_columns:
            try:
                # Try to convert a sample to datetime
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample and pd.to_datetime(sample, errors='coerce') is not pd.NaT:
                    datetime_columns.append(col)
            except (ValueError, TypeError):
                pass
    
    return datetime_columns


def perform_eda(df, numeric_columns, categorical_columns, datetime_columns):
    """Perform exploratory data analysis on the dataframe"""
    st.subheader("**Exploratory Data Analysis**")
    
    # Missing values analysis
    # st.write("### Missing Values Analysis")
    missing = df.isna().sum().reset_index()
    missing.columns = ['Column', 'Missing Count']
    missing['Missing Percentage'] = (missing['Missing Count'] / len(df) * 100).round(2)
    missing = missing.sort_values('Missing Percentage', ascending=False)
    
    # Plot missing values
    if missing['Missing Count'].sum() > 0:
        fig = px.bar(
            missing[missing['Missing Count'] > 0], 
            x='Column', 
            y='Missing Percentage',
            title="Missing Values by Column (%)",
            color='Missing Percentage',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig)
    else:
        st.write("No missing values found in the dataset.")
    
    # Data distribution overview
    st.divider()
    st.write("## Data Distribution Overview")
    
    # For numeric columns - show distribution
    if numeric_columns:
        # Option to select which numeric columns to view
        st.markdown("#### Select numeric columns to visualize")
        selected_numeric = st.multiselect(
            "Select numeric columns to visualize",
            label_visibility="hidden",
            options=numeric_columns,
            default=numeric_columns[:min(5, len(numeric_columns))]
        )
        
        if selected_numeric:
            # Create histograms for selected columns
            for col in selected_numeric:
                fig = px.histogram(
                    df, 
                    x=col,
                    marginal="box",
                    title=f"Distribution of {col}"
                )
                st.plotly_chart(fig)
            
            # Show numeric summary statistics
            st.divider()
            st.write("**Numeric Summary Statistics (chosen columns)**")
            st.dataframe(df[selected_numeric].describe(), use_container_width=True)
        
        st.divider()
        # Outlier detection for numeric columns
        st.write("### Outlier Detection")
        
        # Option to select which numeric column to view
        outlier_col = st.selectbox(
            "**Select numeric column for outlier detection:**",
            options=numeric_columns
        )
        
        if outlier_col:
            # Calculate IQR
            q1 = df[outlier_col].quantile(0.25)
            q3 = df[outlier_col].quantile(0.75)
            iqr = q3 - q1
            
            # Determine outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
            outlier_count = len(outliers)
            
            st.write(f"**Outlier analysis for {outlier_col}**:")
            st.write(f"**IQR (Interquartile Range):** {iqr:.4f}")
            st.write(f"**Lower bound:** {lower_bound:.4f}")
            st.write(f"**Upper bound:** {upper_bound:.4f}")
            st.write(f"**Number of outliers:** {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
            
            # Box plot to visualize outliers
            fig = px.box(
                df,
                y=outlier_col,
                title=f"**Box Plot with Outliers for** {outlier_col}"
            )
            st.plotly_chart(fig)
            
            # Show outliers if not too many
            if 0 < outlier_count < 100:
                with st.expander("View outlier records"):
                    st.dataframe(outliers, use_container_width=True)

    
    # For categorical columns - show value counts
    if categorical_columns:
        # Option to select which categorical columns to view
        st.divider()
        st.markdown("#### Select Categorical columns to visualize")
        selected_categorical = st.multiselect(
            "Visualize categorical columns",
            label_visibility="hidden",
            options=categorical_columns,
            default=categorical_columns[:min(3, len(categorical_columns))]
        )
        
        if selected_categorical:
            # Create bar charts for selected columns
            for col in selected_categorical:
                # Get value counts
                value_counts = df[col].value_counts().nlargest(15)
                
                fig = px.bar(
                    value_counts,
                    title=f"Top values for {col}",
                    labels={"index": col, "value": "Count"},
                    color=value_counts.values,
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig)
            
            # Show categorical summary statistics
            st.divider()
            st.write("**Categorical Summary Statistics (Common)**")
            cat_stats = pd.DataFrame({
                'Column': selected_categorical,
                'Unique Values': [df[col].nunique() for col in selected_categorical],
                'Most Common': [df[col].value_counts().index[0] if df[col].value_counts().shape[0] > 0 else None for col in selected_categorical],
                'Most Common Count': [df[col].value_counts().iloc[0] if df[col].value_counts().shape[0] > 0 else 0 for col in selected_categorical],
                'Most Common %': [(df[col].value_counts().iloc[0] / len(df) * 100).round(2) if df[col].value_counts().shape[0] > 0 else 0 for col in selected_categorical]
            })
            st.dataframe(cat_stats, use_container_width=True)
    
    # For datetime columns - show timeline
    if datetime_columns:
        # Option to select which datetime column to view
        st.divider()
        st.markdown("#### Select DateTime columns to visualize")
        selected_datetime = st.selectbox(
            "Select DateTime columns to visualize",
            label_visibility="hidden",
            options=datetime_columns
        )
        
        if selected_datetime:
            # Ensure column is datetime
            try:
                if df[selected_datetime].dtype != 'datetime64[ns]':
                    df[selected_datetime] = pd.to_datetime(df[selected_datetime], errors='coerce')
                
                # Create timeline
                timeline_df = df[selected_datetime].value_counts().sort_index().reset_index()
                timeline_df.columns = ['Date', 'Count']
                
                fig = px.line(
                    timeline_df,
                    x='Date',
                    y='Count',
                    title=f"Timeline of {selected_datetime}",
                    markers=True
                )
                st.plotly_chart(fig)
                
                # Show datetime range
                st.write(f"Date range: {df[selected_datetime].min()} to {df[selected_datetime].max()}")
                
                # Show distribution by year, month, day of week
                st.write(f"**Distribution by time components:** {selected_datetime}")
                
                if len(df[selected_datetime].dropna()) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Month distribution
                        month_counts = df[selected_datetime].dt.month.value_counts().sort_index()
                        month_names = {
                            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                        }
                        month_counts.index = month_counts.index.map(month_names)
                        
                        fig = px.bar(
                            month_counts,
                            title="Month Distribution",
                            labels={"index": "Month", "value": "Count"},
                            color=month_counts.values,
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Day of week distribution
                        dow_counts = df[selected_datetime].dt.dayofweek.value_counts().sort_index()
                        dow_names = {
                            0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 
                            4: 'Fri', 5: 'Sat', 6: 'Sun'
                        }
                        dow_counts.index = dow_counts.index.map(dow_names)
                        
                        fig = px.bar(
                            dow_counts,
                            title="Day of Week Distribution",
                            labels={"index": "Day", "value": "Count"},
                            color=dow_counts.values,
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error processing datetime column: {str(e)}")
        

def analyze_numeric_column(df, numeric_columns):
    """Analyze a single numeric column with various visualizations"""
    st.subheader("Numeric Column Analysis")
    
    selected_column = st.selectbox(
        "Select a numeric column:",
        options=numeric_columns
    )
    
    if selected_column:
        # Visualization type
        viz_type = st.selectbox(
            "**Visualization type:**",
            options=["Histogram", "Box Plot", "Violin Plot", "KDE Plot"]
        )
        
        # Filter out NaN values
        valid_data = df[~df[selected_column].isna()]
        
        if viz_type == "Histogram":
            # Create histogram with adjustable bins
            bin_count = st.slider("Number of bins (grouped data interval):", min_value=0, max_value=20, value=5, key="histogram_bins")
            
            fig = px.histogram(
                valid_data, 
                x=selected_column, 
                nbins=bin_count,
                marginal="box",
                title=f"Histogram of {selected_column}"
            )
            st.plotly_chart(fig)
            
        elif viz_type == "Box Plot":
            fig = px.box(
                valid_data, 
                y=selected_column,
                title=f"**Box Plot of** {selected_column}"
            )
            st.plotly_chart(fig)
            
        elif viz_type == "Violin Plot":
            fig = px.violin(
                valid_data,
                y=selected_column,
                box=True,
                points="all",
                title=f"Violin Plot of {selected_column}"
            )
            st.plotly_chart(fig)
            
        elif viz_type == "KDE Plot":
            # Create KDE plot
            try:
                fig = ff.create_distplot(
                    [valid_data[selected_column].dropna()],
                    [selected_column],
                    show_hist=False,
                    curve_type='kde'
                )
                fig.update_layout(title=f"KDE Plot of {selected_column}")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error creating KDE plot: {str(e)}")
                st.info("KDE plots may fail for certain data distributions. Try another visualization.")
        
        # Display statistics
        with st.expander("**Statistical Summary**", expanded=True, icon=":material/monitoring:"):
            stats = df[selected_column].describe()
            st.write(stats)
            
            # Additional stats
            st.write(f"**Skewness:** {df[selected_column].skew():.4f}")
            st.write(f"**Kurtosis:** {df[selected_column].kurtosis():.4f}")
            
            # Check for outliers
            q1 = np.percentile(valid_data[selected_column], 25)
            q3 = np.percentile(valid_data[selected_column], 75)
            iqr = q3 - q1
            outlier_low = q1 - 1.5 * iqr
            outlier_high = q3 + 1.5 * iqr
            outliers = valid_data[(valid_data[selected_column] < outlier_low) | 
                                 (valid_data[selected_column] > outlier_high)]
            
            st.write(f"**Outliers:** {len(outliers)} ({len(outliers)/len(valid_data)*100:.2f}%)")
            
            # Show histogram of z-scores to check normality
            z_scores = (valid_data[selected_column] - valid_data[selected_column].mean()) / valid_data[selected_column].std()
            
            fig = px.histogram(
                z_scores,
                title="Z-Score Distribution (Check for Normality)",
                labels={"value": "Z-Score"},
                marginal="box"
            )
            st.plotly_chart(fig)
            
            # Perform Shapiro-Wilk test for normality
            try:
                from scipy.stats import shapiro
                
                # Use a sample if the dataset is too large
                if len(valid_data) > 5000:
                    sample = valid_data[selected_column].sample(5000)
                else:
                    sample = valid_data[selected_column]
                
                stat, p = shapiro(sample)
                
                st.write(f"Shapiro-Wilk Normality Test:")
                st.write(f"Statistic: {stat:.4f}")
                st.write(f"p-value: {p:.4f}")
                
                if p < 0.05:
                    st.write("The data is not normally distributed (p < 0.05).")
                else:
                    st.write("The data appears to be normally distributed (p >= 0.05).")
            except Exception as e:
                st.write(f"Could not perform normality test: {str(e)}")


def analyze_categorical_column(df, categorical_columns):
    """Analyze a single categorical column with various visualizations"""
    st.subheader("Categorical Column Analysis")
    
    selected_column = st.selectbox(
        "Select a categorical column:",
        options=categorical_columns
    )
    
    if selected_column:
        # Get value counts
        value_counts = df[selected_column].value_counts()
        
        # Limit categories shown if there are too many
        max_categories = st.slider("Max categories to display:", 5, 50, 20)
        if len(value_counts) > max_categories:
            value_counts = value_counts.nlargest(max_categories)
            st.warning(f"Showing only top {max_categories} categories out of {df[selected_column].nunique()}.")
        
        # Visualization type
        viz_type = st.selectbox(
            "**Visualization type:**",
            ("Bar Chart", "Pie Chart", "Tree Map"),
        )
        
        if viz_type == "Bar Chart":
            # Bar chart
            fig = px.bar(
                value_counts,
                title=f"Distribution of {selected_column}",
                labels={"index": selected_column, "value": "Count"},
                color=value_counts.values,
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig)
        
        elif viz_type == "Pie Chart":
            # Pie chart (only if categories are not too many)
            if len(value_counts) <= 20:
                fig = px.pie(
                    names=value_counts.index,
                    values=value_counts.values,
                    title=f"Proportion of {selected_column}"
                )
                st.plotly_chart(fig)
            else:
                st.warning("Too many categories for a pie chart. Consider using the bar chart instead.")
                # Still show pie chart with top 10
                top_10 = value_counts.nlargest(10)
                fig = px.pie(
                    names=top_10.index,
                    values=top_10.values,
                    title=f"Proportion of Top 10 values for {selected_column}"
                )
                st.plotly_chart(fig)
        
        elif viz_type == "Tree Map":
            # Tree map
            tree_data = pd.DataFrame({
                'Category': value_counts.index,
                'Count': value_counts.values
            })
            
            fig = px.treemap(
                tree_data,
                path=['Category'],
                values='Count',
                title=f"Tree Map of {selected_column}"
            )
            st.plotly_chart(fig)
        
        # Show statistics
        with st.expander("**Category Statistics**", expanded=True, icon=":material/analytics:"):
            st.write(f"**Total categories:** {df[selected_column].nunique()}")
            
            if not value_counts.empty:
                st.write(f"Most common: {df[selected_column].mode()[0]} ({value_counts.iloc[0]} occurrences)")
            
            st.write(f"Missing values: {df[selected_column].isna().sum()} ({df[selected_column].isna().sum()/len(df)*100:.2f}%)")
            
            # Show full table
            st.write("**Full category distribution:**")
            st.dataframe(pd.DataFrame({
                'Category': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / value_counts.sum() * 100).round(2)
            }), use_container_width=True)
            
            # Class imbalance analysis
            if value_counts.shape[0] > 1:
                st.write("**Class Imbalance Analysis:**")
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                st.write(f"Imbalance ratio (most frequent / least frequent): {imbalance_ratio:.2f}")
                
                if imbalance_ratio > 10:
                    st.warning("High class imbalance detected (ratio > 10).")
                elif imbalance_ratio > 3:
                    st.info("Moderate class imbalance detected (ratio > 3).")


def analyze_datetime_column(df, datetime_columns):
    """Analyze a single datetime column with various visualizations"""
    st.subheader("Datetime Column Analysis")
    
    selected_column = st.selectbox(
        "Select a datetime column:",
        options=datetime_columns
    )
    
    if selected_column:
        # Ensure column is datetime
        try:
            if df[selected_column].dtype != 'datetime64[ns]':
                df[selected_column] = pd.to_datetime(df[selected_column], errors='coerce')
                st.info(f"Converted {selected_column} to datetime format.")
        except Exception as e:
            st.error(f"Error converting to datetime: {str(e)}")
            return
        
        # Filter valid dates
        valid_dates = df[~df[selected_column].isna()]
        
        # Time unit selection
        time_unit = st.selectbox(
            "Group by:",
            options=["Year", "Month", "Day", "Hour", "Weekday", "Quarter"]
        )
        
        if time_unit == "Year":
            date_counts = valid_dates[selected_column].dt.year.value_counts().sort_index()
        elif time_unit == "Month":
            date_counts = valid_dates[selected_column].dt.to_period('M').value_counts().sort_index()
            date_counts.index = date_counts.index.astype(str)
        elif time_unit == "Day":
            date_counts = valid_dates[selected_column].dt.date.value_counts().sort_index()
            date_counts.index = date_counts.index.astype(str)
        elif time_unit == "Hour":
            date_counts = valid_dates[selected_column].dt.hour.value_counts().sort_index()
        elif time_unit == "Weekday":
            date_counts = valid_dates[selected_column].dt.dayofweek.value_counts().sort_index()
            weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                         3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            date_counts.index = date_counts.index.map(weekday_map)
        else:  # Quarter
            date_counts = valid_dates[selected_column].dt.quarter.value_counts().sort_index()
        
        # Visualization type
        viz_type = st.radio(
            "Visualization type:",
            options=["Bar Chart", "Line Chart", "Heatmap Calendar"]
        )
        
        if viz_type in ["Bar Chart", "Line Chart"]:
            # Plot the distribution
            if viz_type == "Bar Chart":
                fig = px.bar(
                    date_counts,
                    title=f"Distribution of {selected_column} by {time_unit}",
                    labels={"index": time_unit, "value": "Count"},
                    color=date_counts.values,
                    color_continuous_scale="Viridis"
                )
            else:  # Line Chart
                fig = px.line(
                    x=date_counts.index,
                    y=date_counts.values,
                    markers=True,
                    title=f"Trend of {selected_column} by {time_unit}"
                )
            st.plotly_chart(fig)
        
        elif viz_type == "Heatmap Calendar" and time_unit in ["Month", "Day"]:
            # Only show calendar heatmap for Day or Month
            try:
                # Create dataframe for calendar heatmap
                if time_unit == "Day":
                    calendar_df = valid_dates[selected_column].dt.date.value_counts().reset_index()
                    calendar_df.columns = ['Date', 'Count']
                    calendar_df['Year'] = calendar_df['Date'].dt.year
                    calendar_df['Month'] = calendar_df['Date'].dt.month
                    calendar_df['Day'] = calendar_df['Date'].dt.day
                else:  # Month
                    calendar_df = valid_dates[selected_column].dt.to_period('M').value_counts().reset_index()
                    calendar_df.columns = ['Date', 'Count']
                    calendar_df['Date'] = calendar_df['Date'].astype(str)
                    calendar_df['Year'] = calendar_df['Date'].str.split('-').str[0].astype(int)
                    calendar_df['Month'] = calendar_df['Date'].str.split('-').str[1].astype(int)
                
                # Create a pivot table for the heatmap
                if time_unit == "Day":
                    pivot_table = calendar_df.pivot_table(
                        index='Day',
                        columns='Month',
                        values='Count',
                        aggfunc='sum'
                    )
                else:  # Month
                    pivot_table = calendar_df.pivot_table(
                        index='Month',
                        columns='Year',
                        values='Count',
                        aggfunc='sum'
                    )
                
                # Create heatmap
                fig = px.imshow(
                    pivot_table,
                    labels=dict(x="Month" if time_unit == "Day" else "Year", 
                              y="Day" if time_unit == "Day" else "Month",
                              color="Count"),
                    title=f"Calendar Heatmap of {selected_column}",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error creating calendar heatmap: {str(e)}")
                st.info("Showing standard visualization instead:")
                
                fig = px.bar(
                    date_counts,
                    title=f"Distribution of {selected_column} by {time_unit}",
                    labels={"index": time_unit, "value": "Count"}
                )
                st.plotly_chart(fig)
        
        # Show statistics
        with st.expander("Datetime Statistics"):
            st.write(f"Date range: {df[selected_column].min()} to {df[selected_column].max()}")
            if df[selected_column].min() and df[selected_column].max():
                time_span = (df[selected_column].max() - df[selected_column].min()).days
                st.write(f"Time span: {time_span} days")
                
                # Average frequency
                if len(valid_dates) > 1:
                    freq = time_span / (len(valid_dates) - 1)
                    st.write(f"Average interval between records: {freq:.2f} days")
            
            st.write(f"Missing values: {df[selected_column].isna().sum()} ({df[selected_column].isna().sum()/len(df)*100:.2f}%)")
            
            # Time series patterns
            if time_unit in ["Month", "Day", "Hour", "Weekday"]:
                st.write(f"Most frequent {time_unit}: {date_counts.idxmax()} ({date_counts.max()} occurrences)")
                st.write(f"Least frequent {time_unit}: {date_counts.idxmin()} ({date_counts.min()} occurrences)")


def analyze_numeric_vs_numeric_a(df, x_column, y_column):
    """Analyze relationship between two numeric columns"""
    # Scatter plot for numeric vs numeric
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        opacity=0.6,
        title=f"{y_column} vs {x_column}"
    )
    
    # Add trend line option
    add_trendline = st.checkbox("Add trend line")
    if add_trendline:
        fig.update_layout(showlegend=True)
        fig.add_traces(
            px.scatter(
                df, 
                x=x_column, 
                y=y_column, 
                trendline="ols",
                opacity=0
            ).data[1]
        )
    
    st.plotly_chart(fig)
    
    # Show correlation
    corr = df[[x_column, y_column]].corr().iloc[0,1]
    st.write(f"Pearson correlation: {corr:.4f}")
    
    # Add hexbin plot for large datasets
    if len(df) > 1000:
        st.write("Hexbin plot (better for large datasets):")
        fig = px.density_heatmap(
            df,
            x=x_column,
            y=y_column,
            marginal_x="histogram",
            marginal_y="histogram"
        )
        st.plotly_chart(fig)
    
    # Statistical tests
    with st.expander("Statistical Analysis"):
        # Calculate regression
        try:
            from scipy import stats
            
            # Remove NaN values
            valid_data = df[[x_column, y_column]].dropna()
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(valid_data[x_column], valid_data[y_column])
            
            st.write(f"Linear Regression Results:")
            st.write(f"Slope: {slope:.4f}")
            st.write(f"Intercept: {intercept:.4f}")
            st.write(f"R-squared: {r_value**2:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            st.write(f"Standard Error: {std_err:.4f}")
            
            if p_value < 0.05:
                st.write("There is a statistically significant linear relationship (p < 0.05).")
            else:
                st.write("There is no statistically significant linear relationship (p >= 0.05).")
            
            # Equation of the line
            st.write(f"Equation: y = {slope:.4f}x + {intercept:.4f}")
            
            # Spearman correlation (for non-linear relationships)
            spearman_corr, spearman_p = stats.spearmanr(valid_data[x_column], valid_data[y_column])
            
            st.write(f"Spearman Rank Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
            
            if spearman_p < 0.05:
                if abs(spearman_corr) - abs(r_value) > 0.1:
                    st.write("The Spearman correlation is stronger than Pearson, suggesting a non-linear relationship.")
        except Exception as e:
            st.write(f"Error in statistical analysis: {str(e)}")


def analyze_categorical_vs_numeric_a(df, x_column, y_column):
    """Analyze relationship between a categorical and a numeric column"""
    # Offer multiple plot types
    plot_type = st.selectbox(
        "**Select plot type:**",
        ("Box Plot", "Violin Plot", "Bar Chart"),
    )
    
    # Limit categories if too many
    categories = df[x_column].nunique()
    if categories > 20:
        top_n = st.slider("Show top N categories:", 5, 30, 10)
        top_cats = df[x_column].value_counts().nlargest(top_n).index
        filtered_df = df[df[x_column].isin(top_cats)]
        st.warning(f"Showing only top {top_n} categories out of {categories}.")
    else:
        filtered_df = df
    
    if plot_type == "Box Plot":
        fig = px.box(
            filtered_df,
            x=x_column,
            y=y_column,
            title=f"{y_column} by {x_column}"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Violin Plot":
        fig = px.violin(
            filtered_df,
            x=x_column,
            y=y_column,
            box=True,
            title=f"{y_column} by {x_column}"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Bar Chart":
        # Calculate mean, median or sum
        agg_method = st.selectbox(
            "Aggregation method:", 
            options=["Mean", "Median", "Sum", "Count"]
        )
        
        if agg_method == "Mean":
            agg_df = filtered_df.groupby(x_column)[y_column].mean().reset_index()
        elif agg_method == "Median":
            agg_df = filtered_df.groupby(x_column)[y_column].median().reset_index()
        elif agg_method == "Sum":
            agg_df = filtered_df.groupby(x_column)[y_column].sum().reset_index()
        else:  # Count
            agg_df = filtered_df.groupby(x_column)[y_column].count().reset_index()
        
        fig = px.bar(
            agg_df,
            x=x_column,
            y=y_column,
            title=f"{agg_method} of {y_column} by {x_column}",
            color=y_column,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig)
    
    # ANOVA test if applicable
    if agg_method != "Count" and filtered_df[x_column].nunique() > 1:
        with st.expander("Statistical Tests"):
            try:
                from scipy import stats
                
                # One-way ANOVA
                groups = [filtered_df[filtered_df[x_column] == cat][y_column].dropna() 
                         for cat in filtered_df[x_column].unique() if len(filtered_df[filtered_df[x_column] == cat]) > 0]
                
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_val, p_val = stats.f_oneway(*groups)
                    st.write(f"One-way ANOVA p-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.write("There is a statistically significant difference between groups (p < 0.05).")
                    else:
                        st.write("There is no statistically significant difference between groups (p >= 0.05).")
            except ImportError:
                st.write("SciPy not available for statistical tests.")
            except Exception as e:
                st.write(f"Could not perform statistical test: {str(e)}")


def analyze_categorical_vs_categorical_a(df, x_column, y_column):
    """Analyze relationship between two categorical columns"""
    # Contingency table (crosstab)
    crosstab = pd.crosstab(df[x_column], df[y_column])
    
    # Limit categories if too many
    max_cats = 15
    if crosstab.shape[0] > max_cats or crosstab.shape[1] > max_cats:
        st.warning(f"Large contingency table ({crosstab.shape[0]}x{crosstab.shape[1]}). Showing heatmap visualization.")
        
        # Take top categories for each
        if crosstab.shape[0] > max_cats:
            top_rows = df[x_column].value_counts().nlargest(max_cats).index
            crosstab = crosstab.loc[top_rows]
        
        if crosstab.shape[1] > max_cats:
            top_cols = df[y_column].value_counts().nlargest(max_cats).index
            crosstab = crosstab[top_cols]
    
    # Display the crosstab
    with st.expander("Contingency Table"):
        st.write(crosstab)
    
    # Visualization
    viz_type = st.radio(
        "Visualization type:",
        options=["Heatmap", "Stacked Bar", "Mosaic Plot"]
    )
    
    if viz_type == "Heatmap":
        fig = px.imshow(
            crosstab,
            text_auto=True,
            title=f"Heatmap of {x_column} vs {y_column}",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig)
    
    elif viz_type == "Stacked Bar":
        # Normalize option
        normalize = st.checkbox("Show percentages")
        
        if normalize:
            # Normalize by row
            crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            fig = px.bar(
                crosstab_norm,
                title=f"Stacked Bar Chart of {x_column} vs {y_column} (%)",
                labels={"value": "Percentage"}
            )
        else:
            fig = px.bar(
                crosstab,
                title=f"Stacked Bar Chart of {x_column} vs {y_column}",
                labels={"value": "Count"}
            )
        
        st.plotly_chart(fig)
    
    elif viz_type == "Mosaic Plot":
        try:
            # Calculate proportions for mosaic plot
            from statsmodels.graphics.mosaicplot import mosaic
            
            # Create a figure
            plt.figure(figsize=(10, 8))
            mosaic(df, [x_column, y_column])
            plt.title(f"Mosaic Plot of {x_column} vs {y_column}")
            
            # Save and display
            st.pyplot(plt)
        except ImportError:
            st.error("Could not create mosaic plot: statsmodels not available.")
        except Exception as e:
            st.error(f"Error creating mosaic plot: {str(e)}")
            st.write("Defaulting to heatmap:")
            
            fig = px.imshow(
                crosstab,
                text_auto=True,
                title=f"Heatmap of {x_column} vs {y_column}"
            )
            st.plotly_chart(fig)
    
    # Chi-square test
    with st.expander("Statistical Tests"):
        try:
            from scipy.stats import chi2_contingency
            
            # Perform chi-square test
            chi2, p, dof, expected = chi2_contingency(crosstab)
            
            st.write(f"Chi-square statistic: {chi2:.4f}")
            st.write(f"p-value: {p:.4f}")
            st.write(f"Degrees of freedom: {dof}")
            
            if p < 0.05:
                st.write("There is a statistically significant association between variables (p < 0.05).")
            else:
                st.write("There is no statistically significant association between variables (p >= 0.05).")
            
            # Cramér's V for effect size
            n = crosstab.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * min(crosstab.shape[0]-1, crosstab.shape[1]-1)))
            st.write(f"Cramér's V (effect size): {cramers_v:.4f}")
            
            # Interpret Cramér's V
            if cramers_v < 0.1:
                strength = "negligible"
            elif cramers_v < 0.3:
                strength = "weak"
            elif cramers_v < 0.5:
                strength = "moderate"
            else:
                strength = "strong"
            
            st.write(f"Association strength: {strength}")
        except ImportError:
            st.write("SciPy not available for chi-square test.")
        except Exception as e:
            st.write(f"Could not perform chi-square test: {str(e)}")


def analyze_datetime_vs_numeric_a(df, x_column, y_column):
    """Analyze relationship between a datetime and a numeric column"""
    # Ensure column is datetime
    if df[x_column].dtype != 'datetime64[ns]':
        df[x_column] = pd.to_datetime(df[x_column], errors='coerce')
    
    # Time unit selection for grouping
    time_unit = st.selectbox(
        "Group by time unit:",
        options=["Day", "Week", "Month", "Quarter", "Year"]
    )
    
    # Aggregate method
    agg_method = st.selectbox(
        "Aggregation method:", 
        options=["Mean", "Median", "Sum", "Min", "Max"]
    )
    
    # Resample based on selected time unit
    if time_unit == "Day":
        period = "D"
    elif time_unit == "Week":
        period = "W"
    elif time_unit == "Month":
        period = "M"
    elif time_unit == "Quarter":
        period = "Q"
    else:  # Year
        period = "Y"
    
    # Ensure index is datetime for resampling
    temp_df = df[[x_column, y_column]].copy()
    temp_df.set_index(x_column, inplace=True)
    
    # Apply aggregation
    if agg_method == "Mean":
        resampled = temp_df.resample(period).mean()
    elif agg_method == "Median":
        resampled = temp_df.resample(period).median()
    elif agg_method == "Sum":
        resampled = temp_df.resample(period).sum()
    elif agg_method == "Min":
        resampled = temp_df.resample(period).min()
    else:  # Max
        resampled = temp_df.resample(period).max()
    
    # Reset index for plotting
    resampled = resampled.reset_index()
    
    # Plot selection
    plot_type = st.radio(
        "Plot type:",
        options=["Line", "Bar", "Area"]
    )
    
    if plot_type == "Line":
        fig = px.line(
            resampled,
            x=x_column,
            y=y_column,
            markers=True,
            title=f"{agg_method} of {y_column} by {time_unit}"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Bar":
        fig = px.bar(
            resampled,
            x=x_column,
            y=y_column,
            title=f"{agg_method} of {y_column} by {time_unit}"
        )
        st.plotly_chart(fig)
    
    else:  # Area
        fig = px.area(
            resampled,
            x=x_column,
            y=y_column,
            title=f"{agg_method} of {y_column} by {time_unit}"
        )
        st.plotly_chart(fig)
    
    # Show rolling average option
    add_rolling = st.checkbox("Add rolling average")
    if add_rolling:
        window = st.slider("Rolling window size:", 2, 20, 7)
        
        # Calculate rolling average
        resampled[f"{window}-period Rolling Avg"] = resampled[y_column].rolling(window=window).mean()
        
        # Create plot with both series
        fig = px.line(
            resampled,
            x=x_column,
            y=[y_column, f"{window}-period Rolling Avg"],
            title=f"{agg_method} of {y_column} by {time_unit} with Rolling Average",
            markers=True
        )
        st.plotly_chart(fig)
    
    # Show trend analysis
    with st.expander("Trend Analysis"):
        try:
            from scipy import stats
            
            # Calculate trend (linear regression)
            x_numeric = np.arange(len(resampled))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, resampled[y_column])
            
            # Calculate percentage change
            first_value = resampled[y_column].iloc[0]
            last_value = resampled[y_column].iloc[-1]
            pct_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else float('inf')
            
            st.write(f"Slope: {slope:.4f} (per time unit)")
            st.write(f"R-squared: {r_value**2:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            st.write(f"Total change: {last_value - first_value:.4f} ({pct_change:.2f}%)")
            
            if p_value < 0.05:
                st.write("There is a statistically significant trend (p < 0.05).")
            else:
                st.write("There is no statistically significant trend (p >= 0.05).")
        except ImportError:
            st.write("Required libraries not available for trend analysis.")
        except Exception as e:
            st.write(f"Error in trend analysis: {str(e)}")


def perform_bivariate_analysis_a(df, numeric_columns, categorical_columns, datetime_columns):
    """Analyze relationships between two columns of various types"""
    st.subheader("Bivariate Analysis")
    
    # Select the column types for bivariate analysis
    col1, col2 = st.columns(2)
    
    with col1:
        x_type = st.selectbox(
            "Select X-axis column type:",
            options=["Numeric", "Categorical", "Datetime"]
        )
        
        if x_type == "Numeric" and numeric_columns:
            x_column = st.selectbox("Select X-axis column:", numeric_columns, key="x_num")
        elif x_type == "Categorical" and categorical_columns:
            x_column = st.selectbox("Select X-axis column:", categorical_columns, key="x_cat")
        elif x_type == "Datetime" and datetime_columns:
            x_column = st.selectbox("Select X-axis column:", datetime_columns, key="x_date")
        else:
            st.warning(f"No {x_type.lower()} columns available.")
            x_column = None
    
    with col2:
        y_type = st.selectbox(
            "Select Y-axis column type:",
            options=["Numeric", "Categorical", "Datetime"]
        )
        
        if y_type == "Numeric" and numeric_columns:
            y_column = st.selectbox("Select Y-axis column:", numeric_columns, key="y_num")
        elif y_type == "Categorical" and categorical_columns:
            y_column = st.selectbox("Select Y-axis column:", categorical_columns, key="y_cat")
        elif y_type == "Datetime" and datetime_columns:
            y_column = st.selectbox("Select Y-axis column:", datetime_columns, key="y_date")
        else:
            st.warning(f"No {y_type.lower()} columns available.")
            y_column = None
    
    # Check if we can process based on selected column types
    if x_column and y_column:
        # Handle datetime conversions if needed
        if x_type == "Datetime" and df[x_column].dtype != 'datetime64[ns]':
            df[x_column] = pd.to_datetime(df[x_column], errors='coerce')
        
        if y_type == "Datetime" and df[y_column].dtype != 'datetime64[ns]':
            df[y_column] = pd.to_datetime(df[y_column], errors='coerce')
        
        # Visualization based on column type combinations
        if x_type == "Numeric" and y_type == "Numeric":
            analyze_numeric_vs_numeric(df, x_column, y_column)
        
        elif (x_type == "Categorical" and y_type == "Numeric") or (x_type == "Numeric" and y_type == "Categorical"):
            # Ensure categorical is x and numeric is y
            if x_type == "Numeric" and y_type == "Categorical":
                x_column, y_column = y_column, x_column
                x_type, y_type = y_type, x_type
            
            analyze_categorical_vs_numeric(df, x_column, y_column)
        
        elif x_type == "Categorical" and y_type == "Categorical":
            analyze_categorical_vs_categorical(df, x_column, y_column)
        
        elif (x_type == "Datetime" and y_type == "Numeric") or (y_type == "Datetime" and x_type == "Numeric"):
            # Make sure datetime is on x-axis
            if y_type == "Datetime":
                x_column, y_column = y_column, x_column
            
            analyze_datetime_vs_numeric(df, x_column, y_column)
        
        else:
            st.warning("The selected column combination doesn't have a predefined visualization method.")
    else:
        st.warning("Please select valid columns for both axes.")


def perform_time_series_analysis(df, datetime_columns, numeric_columns):
    """Perform time series analysis on datetime and numeric columns"""
    st.subheader("Time Series Analysis")
    
    if not datetime_columns:
        st.warning("Sorry, but data doesnot contain Date and Time related columns for futher Analysis.")
        return
    
    if not numeric_columns:
        st.warning("Sorry, but data doesnot contain any Numeric valued columns for futher Analysis.")
        return
    
    # Select datetime column
    date_col = st.selectbox(
        "Select datetime column:",
        options=datetime_columns
    )
    
    # Ensure column is datetime
    try:
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            st.info(f"Converted {date_col} to datetime format.")
    except Exception as e:
        st.error(f"Error converting to datetime: {str(e)}")
        return
    
    # Select target variable
    target_col = st.selectbox(
        "Select target variable for time series analysis:",
        options=numeric_columns
    )
    
    if not target_col:
        return
    
    # Create time series dataframe
    ts_df = df[[date_col, target_col]].copy()
    ts_df = ts_df.dropna()
    ts_df = ts_df.sort_values(date_col)
    
    # Set datetime as index
    ts_df = ts_df.set_index(date_col)
    
    # Time unit for resampling
    time_unit = st.selectbox(
        "Time unit for analysis:",
        options=["Original", "Day", "Week", "Month", "Quarter", "Year"]
    )
    
    # Add a small sleep time for UI responsiveness
    import time
    time.sleep(0.1)
    
    if time_unit != "Original":
        # Resample based on selected time unit
        if time_unit == "Day":
            period = "D"
        elif time_unit == "Week":
            period = "W"
        elif time_unit == "Month":
            period = "M"
        elif time_unit == "Quarter":
            period = "Q"
        else:  # Year
            period = "Y"
        
        # Apply resampling
        resampled = ts_df.resample(period).mean()
        resampled = resampled.dropna()
    else:
        resampled = ts_df
    
    # Basic time series plot
    fig = px.line(
        resampled.reset_index(),
        x=date_col,
        y=target_col,
        markers=True,
        title=f"Time Series of {target_col}"
    )
    st.plotly_chart(fig)
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select analysis type:",
        ("Decomposition", "Autocorrelation", "Rolling Statistics", "Forecasting"),
    )
    
    time.sleep(0.1)  # Small delay for UI responsiveness
    
    if analysis_type == "Decomposition":
        # Check if we have enough data points
        if len(resampled) < 2 * max(7, 12, 4):  # Minimum 2 periods
            st.warning("Not enough data points for decomposition. For typical time series decomposition it needs at least 2 observations for each period for it to be able to extract a trend")
            return
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Determine period based on time unit
            if time_unit == "Day":
                period = 7  # Weekly seasonality
            elif time_unit == "Week":
                period = 4  # Monthly seasonality
            elif time_unit == "Month":
                period = 12  # Yearly seasonality
            else:
                period = 4  # Quarterly seasonality
            
            # Allow user to specify period
            period = st.slider("Seasonality period:", 2, min(30, len(resampled)//2), period)
            
            # Decompose the time series
            result = seasonal_decompose(resampled, period=period)
            
            # Plot components
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
            result.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            result.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            result.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            result.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Pie chart of component contribution
            st.write("### Component Contribution Analysis")
            
            # Calculate variance of each component
            observed_var = result.observed.var()
            trend_var = result.trend.var()
            seasonal_var = result.seasonal.var()
            resid_var = result.resid.var()
            
            # Calculate percentage contribution
            total_var = trend_var + seasonal_var + resid_var
            trend_pct = (trend_var / total_var * 100).round(2)
            seasonal_pct = (seasonal_var / total_var * 100).round(2)
            resid_pct = (resid_var / total_var * 100).round(2)
            
            # Create pie chart
            component_df = pd.DataFrame({
                'Component': ['Trend', 'Seasonality', 'Residual'],
                'Variance Percentage': [trend_pct, seasonal_pct, resid_pct]
            })
            
            fig = px.pie(
                component_df,
                names='Component',
                values='Variance Percentage',
                title="Variance Contribution of Time Series Components"
            )
            st.plotly_chart(fig)
            
            st.write(f"Trend contribution: {trend_pct:.2f}%")
            st.write(f"Seasonal contribution: {seasonal_pct:.2f}%")
            st.write(f"Residual (unexplained) contribution: {resid_pct:.2f}%")
            
            if trend_pct > 60:
                st.info("The time series is dominated by trend.")
            elif seasonal_pct > 60:
                st.info("The time series is dominated by seasonality.")
            elif resid_pct > 60:
                st.warning("The time series has high unexplained variability.")
            
        except Exception as e:
            st.error(f"Error in decomposition: {str(e)}")
    
    elif analysis_type == "Autocorrelation":
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            # Plot ACF and PACF
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            plot_acf(resampled, ax=ax1)
            plot_pacf(resampled, ax=ax2)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Augmented Dickey-Fuller test for stationarity
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(resampled[target_col].dropna())
            st.write(f"Augmented Dickey-Fuller test statistic: {result[0]:.4f}")
            st.write(f"p-value: {result[1]:.4f}")
            
            if result[1] <= 0.05:
                st.write("The time series is stationary (p <= 0.05).")
            else:
                st.write("The time series is not stationary (p > 0.05).")
                
                # Suggest differencing
                if st.checkbox("View differenced series"):
                    diff = resampled.diff().dropna()
                    
                    fig = px.line(
                        diff.reset_index(),
                        x=date_col,
                        y=target_col,
                        title=f"Differenced Series of {target_col}"
                    )
                    st.plotly_chart(fig)
                    
                    # Test stationarity again
                    diff_result = adfuller(diff[target_col].dropna())
                    st.write(f"Differenced series ADF test statistic: {diff_result[0]:.4f}")
                    st.write(f"p-value: {diff_result[1]:.4f}")
                    
                    if diff_result[1] <= 0.05:
                        st.write("The differenced series is stationary (p <= 0.05).")
                    else:
                        st.write("The differenced series is still not stationary (p >= 0.05).")
        except ImportError:
            st.error("Required libraries not available for autocorrelation analysis.")
        except Exception as e:
            st.error(f"Error in autocorrelation analysis: {str(e)}")
    
    elif analysis_type == "Rolling Statistics":
        window = st.slider("Rolling window size:", 2, min(30, len(resampled)//2), 7)
        
        # Calculate rolling statistics
        rolling_mean = resampled[target_col].rolling(window=window).mean()
        rolling_std = resampled[target_col].rolling(window=window).std()
        
        # Create dataframe for plotting
        rolling_df = pd.DataFrame({
            'Original': resampled[target_col],
            'Rolling Mean': rolling_mean,
            'Rolling STD': rolling_std
        }).reset_index()
        
        # Plot
        fig = px.line(
            rolling_df,
            x=date_col,
            y=['Original', 'Rolling Mean'],
            title=f"Rolling Mean (window={window})"
        )
        st.plotly_chart(fig)
        
        fig = px.line(
            rolling_df,
            x=date_col,
            y='Rolling STD',
            title=f"Rolling Standard Deviation (window={window})"
        )
        st.plotly_chart(fig)
        
        # Add pie chart of data composition
        st.write("### Value Distribution")
        # Create bins for the data
        bins = pd.cut(resampled[target_col], bins=5)
        bin_counts = bins.value_counts().sort_index()
        
        # Create labels for the bins
        bin_labels = [f"{interval.left:.2f} to {interval.right:.2f}" for interval in bin_counts.index]
        
        fig = px.pie(
            values=bin_counts.values,
            names=bin_labels,
            title=f"Distribution of {target_col} values"
        )
        st.plotly_chart(fig)
    
    elif analysis_type == "Forecasting":
        st.write("Simple forecasting models:")
        
        try:
            # Add a small sleep time to ensure UI responsiveness
            time.sleep(0.1)
            
            # Split data into train/test
            train_size = st.slider("Training data percentage:", 50, 90, 80)
            train_size = int(len(resampled) * train_size / 100)
            
            train = resampled.iloc[:train_size]
            test = resampled.iloc[train_size:]
            
            if len(test) > 0:
                # Simple forecasting models
                forecast_type = st.radio(
                    "Forecast method:",
                    options=["Exponential Smoothing", "Moving Average", "Last Value"]
                )
                
                # Add a small sleep time for UI responsiveness
                time.sleep(0.1)
                
                if forecast_type == "Exponential Smoothing":
                    try:
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing
                        
                        # Fit exponential smoothing model
                        seasonal_periods = 12 if time_unit == "Month" else 4
                        
                        # Adjust seasonal_periods if data is insufficient
                        if len(train) < 2 * seasonal_periods:
                            seasonal_periods = max(2, len(train) // 2)
                            st.info(f"Adjusted seasonal periods to {seasonal_periods} due to limited data")
                        
                        model = ExponentialSmoothing(
                            train[target_col],
                            trend='add',
                            seasonal='add',
                            seasonal_periods=seasonal_periods
                        ).fit()
                        
                        # Make forecast
                        forecast = model.forecast(len(test))
                        
                        # Create dataframe for plotting
                        forecast_df = pd.DataFrame({
                            'Actual': resampled[target_col],
                            'Forecast': pd.Series(forecast.values, index=test.index)
                        }).reset_index()
                        
                        # Plot
                        fig = px.line(
                            forecast_df,
                            x=date_col,
                            y=['Actual', 'Forecast'],
                            title="Exponential Smoothing Forecast"
                        )
                        
                        # Add vertical line at train/test split
                        fig.add_vline(x=train.index[-1], line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error in exponential smoothing: {str(e)}")
                        st.info("Falling back to simpler forecasting method.")
                        forecast_type = "Moving Average"
                
                if forecast_type == "Moving Average":
                    # Moving average window
                    ma_window = st.slider("Moving Average Window:", 1, min(20, len(train)), 3)
                    
                    # Calculate moving average
                    ma_forecast = []
                    history = list(train[target_col].values)
                    
                    for _ in range(len(test)):
                        ma = np.mean(history[-ma_window:])
                        ma_forecast.append(ma)
                        history.append(test[target_col].iloc[_])
                    
                    # Create dataframe for plotting
                    forecast_df = pd.DataFrame({
                        'Actual': resampled[target_col],
                        'Forecast': pd.Series(ma_forecast, index=test.index)
                    }).reset_index()
                    
                    # Plot
                    fig = px.line(
                        forecast_df,
                        x=date_col,
                        y=['Actual', 'Forecast'],
                        title=f"Moving Average Forecast (window={ma_window})"
                    )
                    
                    # Add vertical line at train/test split
                    fig.add_vline(x=train.index[-1], line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig)
                
                elif forecast_type == "Last Value":
                    # Use last value as forecast
                    last_value = train[target_col].iloc[-1]
                    lv_forecast = [last_value] * len(test)
                    
                    # Create dataframe for plotting
                    forecast_df = pd.DataFrame({
                        'Actual': resampled[target_col],
                        'Forecast': pd.Series(lv_forecast, index=test.index)
                    }).reset_index()
                    
                    # Plot
                    fig = px.line(
                        forecast_df,
                        x=date_col,
                        y=['Actual', 'Forecast'],
                        title="Last Value Forecast"
                    )
                    
                    # Add vertical line at train/test split
                    fig.add_vline(x=train.index[-1], line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig)
                
                # Calculate error metrics
                try:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    
                    if forecast_type == "Exponential Smoothing":
                        forecast_values = forecast.values
                    elif forecast_type == "Moving Average":
                        forecast_values = ma_forecast
                    else:  # Last Value
                        forecast_values = lv_forecast
                    
                    mae = mean_absolute_error(test[target_col], forecast_values)
                    rmse = np.sqrt(mean_squared_error(test[target_col], forecast_values))
                    
                    # Calculate MAPE carefully to avoid division by zero
                    actual = test[target_col].values
                    mape_values = np.abs((actual - np.array(forecast_values)) / actual) * 100
                    mape_values = mape_values[~np.isinf(mape_values)]  # Remove inf values
                    mape = np.mean(mape_values) if len(mape_values) > 0 else np.nan
                    
                    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    
                    if not np.isnan(mape):
                        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                    else:
                        st.write("Mean Absolute Percentage Error (MAPE): Could not calculate (division by zero)")
                        
                    # Add visualization of error distribution
                    st.write("### Error Distribution")
                    
                    if forecast_type == "Exponential Smoothing":
                        errors = test[target_col].values - forecast.values
                    elif forecast_type == "Moving Average":
                        errors = test[target_col].values - np.array(ma_forecast)
                    else:  # Last Value
                        errors = test[target_col].values - np.array(lv_forecast)
                    
                    error_df = pd.DataFrame({'Error': errors})
                    
                    fig = px.histogram(
                        error_df,
                        x='Error',
                        marginal="box",
                        title="Distribution of Forecast Errors"
                    )
                    st.plotly_chart(fig)
                    
                    # Pie chart of positive vs negative errors
                    positive_errors = (errors > 0).sum()
                    negative_errors = (errors < 0).sum()
                    
                    if positive_errors + negative_errors > 0:
                        fig = px.pie(
                            names=['Positive Errors (Underestimation)', 'Negative Errors (Overestimation)'],
                            values=[positive_errors, negative_errors],
                            title="Direction of Forecast Errors"
                        )
                        st.plotly_chart(fig)
                    
                except ImportError:
                    st.error("Required libraries not available for error metrics.")
                except Exception as e:
                    st.error(f"Error calculating metrics: {str(e)}")
            else:
                st.warning("Not enough data for test set after splitting.")
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")

def perform_correlation_analysis(df, numeric_columns):
    """Analyze correlations between all numeric columns"""
    st.subheader("Correlation Analysis")
    
    if len(numeric_columns) < 2:
        st.warning("For correlation analysis at least two numeric columns is required.")
        return
    
    # Add a small sleep time for UI responsiveness
    import time
    time.sleep(0.2)
    
    # Select correlation method
    corr_method = st.selectbox(
        "**Correlation method:**",
        ("Pearson", "Spearman", "Kendall"),
    )
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_columns].corr(method=corr_method.lower())
    
    # Plot heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title=f"{corr_method} Correlation Matrix"
    )
    st.plotly_chart(fig)
    
    # Option to filter correlations
    filter_correlations = st.checkbox("Filter correlations by strength")
    
    if filter_correlations:
        min_corr = st.slider("Minimum correlation strength:", 0.0, 1.0, 0.5)
        
        # Create filtered matrix for display
        filtered_matrix = corr_matrix.copy()
        filtered_matrix[abs(filtered_matrix) < min_corr] = np.nan
        
        # Only show if we have correlations that meet the threshold
        if not filtered_matrix.isna().all().all():
            fig = px.imshow(
                filtered_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title=f"Filtered {corr_method} Correlation Matrix (|corr| >= {min_corr})"
            )
            st.plotly_chart(fig)
        else:
            st.info(f"No correlations with absolute strength >= {min_corr}")
    
    # Feature selection based on correlation
    st.subheader("Feature Selection")
    
    # Target variable for feature importance
    target = st.selectbox(
        "Select target variable:",
        options=numeric_columns
    )
    
    time.sleep(0.1)  # Small delay for UI responsiveness
    
    if target:
        # Show correlations with target
        correlations = corr_matrix[target].drop(target).sort_values(ascending=False)
        
        # Show bar chart of correlations
        fig = px.bar(
            x=correlations.index,
            y=correlations.values,
            labels={"x": "Features", "y": f"Correlation with {target}"},
            title=f"Feature Correlations with {target}",
            color=correlations.values,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig)
        
        # Add pie chart of absolute correlation contribution
        abs_correlations = correlations.abs()
        total_abs_corr = abs_correlations.sum()
        
        if total_abs_corr > 0:
            # Show top 5 features, group others
            if len(abs_correlations) > 5:
                top_5 = abs_correlations.nlargest(5)
                others = pd.Series({'Others': abs_correlations.iloc[5:].sum()})
                pie_data = pd.concat([top_5, others])
            else:
                pie_data = abs_correlations
                
            fig = px.pie(
                names=pie_data.index,
                values=pie_data.values,
                title=f"Features by Absolute Correlation with {target}"
            )
            st.plotly_chart(fig)
        
        # Identify collinearity
        threshold = st.slider(
            "Collinearity threshold:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.05
        )
        
        # Find highly correlated feature pairs
        corr_matrix_no_target = corr_matrix.drop(target, axis=1).drop(target, axis=0)
        high_corr = (corr_matrix_no_target.abs() > threshold).sum().sum() - len(corr_matrix_no_target.columns)
        
        if high_corr > 0:
            st.write(f"Found {high_corr // 2} pairs of features with correlation > {threshold}")
            
            # Display pairs with high correlation
            high_corr_pairs = []
            for i in range(len(corr_matrix_no_target.columns)):
                for j in range(i+1, len(corr_matrix_no_target.columns)):
                    col1 = corr_matrix_no_target.columns[i]
                    col2 = corr_matrix_no_target.columns[j]
                    corr_val = corr_matrix_no_target.iloc[i, j]
                    if abs(corr_val) > threshold:
                        high_corr_pairs.append({
                            "Feature 1": col1,
                            "Feature 2": col2,
                            "Correlation": corr_val
                        })
            
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("Correlation", ascending=False)
                st.dataframe(high_corr_df, use_container_width=True)
                
                # Visualize top correlated pairs
                st.write("### Top Correlated Feature Pairs")
                num_pairs_to_show = min(3, len(high_corr_pairs))
                for i in range(num_pairs_to_show):
                    pair = high_corr_pairs[i]
                    feat1, feat2 = pair["Feature 1"], pair["Feature 2"]
                    
                    fig = px.scatter(
                        df, 
                        x=feat1, 
                        y=feat2,
                        trendline="ols",
                        title=f"{feat2} vs {feat1} (Correlation: {pair['Correlation']:.4f})"
                    )
                    st.plotly_chart(fig)
        else:
            st.write(f"No feature pairs with correlation > {threshold}")
        
        # Option to show scatter plots
        if st.checkbox("Show scatter plots for top correlated features with target"):
            time.sleep(0.1)  # Small delay for UI responsiveness
            
            top_n = st.slider("Number of top correlated features:", 1, 10, 3)
            
            # Get top correlated features
            top_features = correlations.abs().nlargest(top_n).index
            
            for feature in top_features:
                fig = px.scatter(
                    df,
                    x=feature,
                    y=target,
                    trendline="ols",
                    opacity=0.6,
                    title=f"{target} vs {feature} (Correlation: {correlations[feature]:.4f})"
                )
                st.plotly_chart(fig)
                
                # Calculate and display regression statistics
                try:
                    from scipy import stats
                    
                    # Remove NaN values for regression
                    valid_data = df[[feature, target]].dropna()
                    
                    # Calculate regression stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        valid_data[feature], valid_data[target]
                    )
                    
                    st.info(f"""
                    Regression equation: {target} = {slope:.4f} * {feature} + {intercept:.4f}

                    R-squared: {r_value**2:.4f}

                    P-value: {p_value:.4f}
                    """)
                    
                    if p_value < 0.05:
                        st.info("This relationship is statistically significant (p < 0.05)")
                    else:
                        st.warning("This relationship is not statistically significant (p >= 0.05)")
                        
                except Exception as e:
                    st.error(f"Could not calculate regression statistics: {str(e)}")
        
        # Feature clustering based on correlation
        if len(numeric_columns) > 5 and st.checkbox("Cluster features by correlation"):
            try:
                from scipy.cluster import hierarchy
                from scipy.spatial.distance import squareform
                
                # Convert correlation to distance
                corr_linkage = hierarchy.ward(squareform(1 - abs(corr_matrix)))
                
                # Create dendrogram
                fig, ax = plt.subplots(figsize=(10, 8))
                dendrogram = hierarchy.dendrogram(
                    corr_linkage,
                    labels=corr_matrix.columns,
                    ax=ax,
                    leaf_rotation=90
                )
                plt.title('Hierarchical Clustering of Features by Correlation')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("Features that cluster together are highly correlated. For feature selection, you might choose one representative feature from each cluster.")
                
            except Exception as e:
                st.error(f"Could not perform hierarchical clustering: {str(e)}")


def perform_bivariate_analysis(df, numeric_columns, categorical_columns, datetime_columns):
    """Analyze relationships between two columns of various types"""
    st.subheader("Bivariate Analysis")
    
    # Add a small sleep time for UI responsiveness
    import time
    time.sleep(0.1)
    
    # Select the column types for bivariate analysis
    col1, col2 = st.columns(2)
    
    with col1:
        x_type = st.selectbox(
            "Select X-axis column type:",
            options=["Numeric", "Categorical", "Datetime"]
        )
        
        if x_type == "Numeric" and numeric_columns:
            x_column = st.selectbox("Select X-axis column:", numeric_columns, key="x_num")
        elif x_type == "Categorical" and categorical_columns:
            x_column = st.selectbox("Select X-axis column:", categorical_columns, key="x_cat")
        elif x_type == "Datetime" and datetime_columns:
            x_column = st.selectbox("Select X-axis column:", datetime_columns, key="x_date")
        else:
            st.warning(f"No {x_type.lower()} columns available.")
            x_column = None
    
    with col2:
        y_type = st.selectbox(
            "Select Y-axis column type:",
            options=["Numeric", "Categorical", "Datetime"]
        )
        
        if y_type == "Numeric" and numeric_columns:
            y_column = st.selectbox("Select Y-axis column:", numeric_columns, key="y_num")
        elif y_type == "Categorical" and categorical_columns:
            y_column = st.selectbox("Select Y-axis column:", categorical_columns, key="y_cat")
        elif y_type == "Datetime" and datetime_columns:
            y_column = st.selectbox("Select Y-axis column:", datetime_columns, key="y_date")
        else:
            st.warning(f"No {y_type.lower()} columns available.")
            y_column = None
    
    # Check if we can process based on selected column types
    if x_column and y_column:
        # Add small delay for UI responsiveness
        time.sleep(0.2)
        
        # Handle datetime conversions if needed
        if x_type == "Datetime" and df[x_column].dtype != 'datetime64[ns]':
            df[x_column] = pd.to_datetime(df[x_column], errors='coerce')
        
        if y_type == "Datetime" and df[y_column].dtype != 'datetime64[ns]':
            df[y_column] = pd.to_datetime(df[y_column], errors='coerce')
        
        # Visualization based on column type combinations
        if x_type == "Numeric" and y_type == "Numeric":
            analyze_numeric_vs_numeric(df, x_column, y_column)
        
        elif (x_type == "Categorical" and y_type == "Numeric") or (x_type == "Numeric" and y_type == "Categorical"):
            # Ensure categorical is x and numeric is y
            if x_type == "Numeric" and y_type == "Categorical":
                x_column, y_column = y_column, x_column
                x_type, y_type = y_type, x_type
                st.info(f"Swapped axes for better visualization: {x_column} (categorical) on x-axis, {y_column} (numeric) on y-axis")
            
            analyze_categorical_vs_numeric(df, x_column, y_column)
        
        elif x_type == "Categorical" and y_type == "Categorical":
            analyze_categorical_vs_categorical(df, x_column, y_column)
        
        elif (x_type == "Datetime" and y_type == "Numeric") or (y_type == "Datetime" and x_type == "Numeric"):
            # Make sure datetime is on x-axis
            if y_type == "Datetime":
                x_column, y_column = y_column, x_column
                x_type, y_type = y_type, x_type
                st.info(f"Swapped axes for better visualization: {x_column} (datetime) on x-axis, {y_column} (numeric) on y-axis")
            
            analyze_datetime_vs_numeric(df, x_column, y_column)
        
        elif (x_type == "Datetime" and y_type == "Categorical") or (y_type == "Datetime" and x_type == "Categorical"):
            st.write("### Datetime vs Categorical Analysis")
            
            # Ensure datetime is on x-axis for consistency
            if y_type == "Datetime":
                x_column, y_column = y_column, x_column
                x_type, y_type = y_type, x_type
                st.info(f"Swapped axes for better visualization: {x_column} (datetime) on x-axis, {y_column} (categorical) on y-axis")
            
            # Convert datetime to proper format
            if df[x_column].dtype != 'datetime64[ns]':
                df[x_column] = pd.to_datetime(df[x_column], errors='coerce')
            
            # Group data by category and time component
            time_unit = st.selectbox(
                "Group datetime by:",
                options=["Year", "Month", "Day", "Hour", "Weekday"]
            )
            
            # Extract time component
            if time_unit == "Year":
                time_component = df[x_column].dt.year
            elif time_unit == "Month":
                time_component = df[x_column].dt.month
            elif time_unit == "Day":
                time_component = df[x_column].dt.day
            elif time_unit == "Hour":
                time_component = df[x_column].dt.hour
            else:  # Weekday
                time_component = df[x_column].dt.dayofweek
                
            # Create aggregated dataframe
            valid_data = df[[x_column, y_column]].dropna()
            
            # Group by time component and category
            grouped = valid_data.assign(time_comp=time_component).groupby(['time_comp', y_column]).size().reset_index()
            grouped.columns = ['Time Component', y_column, 'Count']
            
            viz_type = st.radio(
                "Visualization type:",
                options=["Stacked Bar", "Line Chart", "Heatmap"]
            )
            
            if viz_type == "Stacked Bar":
                fig = px.bar(
                    grouped,
                    x='Time Component',
                    y='Count',
                    color=y_column,
                    title=f"Distribution of {y_column} by {time_unit}"
                )
                st.plotly_chart(fig)
                
            elif viz_type == "Line Chart":
                fig = px.line(
                    grouped,
                    x='Time Component',
                    y='Count',
                    color=y_column,
                    markers=True,
                    title=f"Trend of {y_column} by {time_unit}"
                )
                st.plotly_chart(fig)
                
            else:  # Heatmap
                # Create pivot table for heatmap
                pivot = grouped.pivot_table(
                    index=y_column,
                    columns='Time Component',
                    values='Count',
                    aggfunc='sum'
                ).fillna(0)
                
                fig = px.imshow(
                    pivot,
                    labels=dict(x=time_unit, y=y_column, color="Count"),
                    title=f"Heatmap of {y_column} by {time_unit}",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig)
                
            # Chi-square test for independence
            with st.expander("Statistical Tests"):
                try:
                    from scipy.stats import chi2_contingency
                    
                    # Create contingency table
                    cont_table = pd.crosstab(time_component, valid_data[y_column])
                    
                    # Perform chi-square test
                    chi2, p, dof, expected = chi2_contingency(cont_table)
                    
                    st.write(f"Chi-square test for independence:")
                    st.write(f"Chi-square statistic: {chi2:.4f}")
                    st.write(f"p-value: {p:.4f}")
                    
                    if p < 0.05:
                        st.write("There is a statistically significant association between the variables (p < 0.05).")
                    else:
                        st.write("There is no statistically significant association between the variables (p >= 0.05).")
                        
                except Exception as e:
                    st.error(f"Could not perform statistical test: {str(e)}")
                    
            # Add pie chart of category distribution
            st.write(f"### Distribution of {y_column}")
            
            category_counts = valid_data[y_column].value_counts()
            
            # Limit to top 10 categories if too many
            if len(category_counts) > 10:
                top_cats = category_counts.nlargest(10)
                others = pd.Series({'Others': category_counts.iloc[10:].sum()})
                pie_data = pd.concat([top_cats, others])
            else:
                pie_data = category_counts
                
            fig = px.pie(
                names=pie_data.index,
                values=pie_data.values,
                title=f"Distribution of {y_column}"
            )
            st.plotly_chart(fig)
            
        elif x_type == "Datetime" and y_type == "Datetime":
            st.write("### Datetime vs Datetime Analysis")
            
            # Convert both columns to datetime
            if df[x_column].dtype != 'datetime64[ns]':
                df[x_column] = pd.to_datetime(df[x_column], errors='coerce')
            
            if df[y_column].dtype != 'datetime64[ns]':
                df[y_column] = pd.to_datetime(df[y_column], errors='coerce')
            
            # Calculate time difference
            valid_data = df[[x_column, y_column]].dropna()
            valid_data['time_diff'] = (valid_data[y_column] - valid_data[x_column]).dt.total_seconds() / 3600  # diff in hours
            
            # Plot histogram of time differences
            fig = px.histogram(
                valid_data,
                x='time_diff',
                title=f"Distribution of Time Difference ({y_column} - {x_column}) in Hours",
                labels={"time_diff": "Time Difference (hours)"}
            )
            st.plotly_chart(fig)
            
            # Show statistics about the time differences
            st.write("### Time Difference Statistics")
            
            # Convert to more appropriate units (hours, days, or months)
            if valid_data['time_diff'].abs().max() > 24*30:  # More than a month
                valid_data['time_diff_days'] = valid_data['time_diff'] / 24
                unit = "days"
                diff_col = 'time_diff_days'
            elif valid_data['time_diff'].abs().max() > 24:  # More than a day
                valid_data['time_diff_days'] = valid_data['time_diff'] / 24
                unit = "days"
                diff_col = 'time_diff_days'
            else:
                unit = "hours"
                diff_col = 'time_diff'
            
            mean_diff = valid_data[diff_col].mean()
            median_diff = valid_data[diff_col].median()
            min_diff = valid_data[diff_col].min()
            max_diff = valid_data[diff_col].max()
            
            st.write(f"Mean time difference: {mean_diff:.2f} {unit}")
            st.write(f"Median time difference: {median_diff:.2f} {unit}")
            st.write(f"Minimum time difference: {min_diff:.2f} {unit}")
            st.write(f"Maximum time difference: {max_diff:.2f} {unit}")
            
            # Create pie chart for positive/negative differences
            positive_count = (valid_data[diff_col] > 0).sum()
            negative_count = (valid_data[diff_col] < 0).sum()
            zero_count = (valid_data[diff_col] == 0).sum()
            
            if positive_count + negative_count + zero_count > 0:
                pie_data = []
                if positive_count > 0:
                    pie_data.append({"Direction": f"{y_column} after {x_column}", "Count": positive_count})
                if negative_count > 0:
                    pie_data.append({"Direction": f"{y_column} before {x_column}", "Count": negative_count})
                if zero_count > 0:
                    pie_data.append({"Direction": "Same time", "Count": zero_count})
                
                pie_df = pd.DataFrame(pie_data)
                
                fig = px.pie(
                    pie_df,
                    names='Direction',
                    values='Count',
                    title="Directionality of Time Differences"
                )
                st.plotly_chart(fig)
        
        else:
            st.warning("The selected column combination doesn't have a predefined visualization method.")
    else:
        st.warning("Please select valid columns for both axes.")


def analyze_numeric_vs_numeric(df, x_column, y_column):
    """Analyze relationship between two numeric columns"""
    st.write(f"### Numeric Column Analysis: {y_column} vs {x_column}")
    
    # Add a small sleep time for UI responsiveness
    import time
    time.sleep(0.2)
    if y_column == x_column:
        st.warning("Please select valid columns for proper analysis.")
    else:
        # Visualization type selection
        viz_type = st.radio(
            "Visualization type:",
            options=["Scatter Plot", "Hexbin Plot", "Bubble Chart", "Contour Plot", "3D Plot"]
        )
        
        # Filter out NaN values
        valid_data = df[[x_column, y_column]].dropna()
        
        if viz_type == "Scatter Plot":
            # Add option for trend line
            add_trendline = st.checkbox("Add trend line")
            
            # Create scatter plot
            fig = px.scatter(
                valid_data,
                x=x_column,
                y=y_column,
                opacity=0.6,
                title=f"{y_column} vs {x_column}",
                trendline="ols" if add_trendline else None
            )
            st.plotly_chart(fig)
            
            # Show regression statistics if trendline is added
            if add_trendline:
                try:
                    import statsmodels.api as sm
                    
                    # Fit linear regression model
                    X = sm.add_constant(valid_data[x_column])
                    model = sm.OLS(valid_data[y_column], X).fit()
                    
                    # Display regression statistics
                    st.write("### Regression Statistics")
                    st.write(f"Equation: {y_column} = {model.params[1]:.4f} * {x_column} + {model.params[0]:.4f}")
                    st.write(f"R-squared: {model.rsquared:.4f}")
                    st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
                    st.write(f"F-statistic: {model.fvalue:.4f}")
                    st.write(f"p-value: {model.f_pvalue:.4f}")
                    
                    if model.f_pvalue < 0.05:
                        st.write("The relationship is statistically significant (p < 0.05).")
                    else:
                        st.write("The relationship is not statistically significant (p >= 0.05).")
                except Exception as e:
                    st.error(f"Could not calculate regression statistics: {str(e)}")
        
        elif viz_type == "Hexbin Plot":
            # More suitable for large datasets
            nbins = st.slider("Number of hexagons (approx.):", 10, 100, 30)
            
            fig = px.density_heatmap(
                valid_data,
                x=x_column,
                y=y_column,
                nbinsx=nbins,
                nbinsy=nbins,
                marginal_x="histogram",
                marginal_y="histogram",
                title=f"Hexbin Plot of {y_column} vs {x_column}"
            )
            st.plotly_chart(fig)
        
        elif viz_type == "Bubble Chart":
            # Select a third variable for bubble size
            size_options = ["Count"] + [col for col in numeric_columns if col != x_column and col != y_column]
            size_col = st.selectbox("Select variable for bubble size:", size_options)
            
            # Create binned data for bubble chart
            if size_col == "Count":
                # Create bins for x and y values
                bin_count = st.slider("Number of bins per axis:", 5, 30, 10)
                x_bins = pd.cut(valid_data[x_column], bin_count)
                y_bins = pd.cut(valid_data[y_column], bin_count)
                
                # Group by bins and count
                grouped = valid_data.groupby([x_bins, y_bins]).size().reset_index()
                grouped.columns = ['x_bin', 'y_bin', 'count']
                
                # Use midpoints of bins for scatter plot
                grouped['x'] = grouped['x_bin'].apply(lambda x: (x.left + x.right) / 2)
                grouped['y'] = grouped['y_bin'].apply(lambda y: (y.left + y.right) / 2)
                
                # Create bubble chart
                fig = px.scatter(
                    grouped,
                    x='x',
                    y='y',
                    size='count',
                    color='count',
                    labels={
                        'x': x_column,
                        'y': y_column,
                        'count': 'Count',
                        'color': 'Count'
                    },
                    title=f"Bubble Chart of {y_column} vs {x_column} (size = count)"
                )
            else:
                # Use third variable for bubble size
                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    size=size_col,
                    color=size_col,
                    title=f"Bubble Chart of {y_column} vs {x_column} (size = {size_col})"
                )
            
            st.plotly_chart(fig)
        
        elif viz_type == "Contour Plot":
            # Create contour plot
            fig = px.density_contour(
                valid_data,
                x=x_column,
                y=y_column,
                title=f"Contour Plot of {y_column} vs {x_column}"
            )
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            st.plotly_chart(fig)
        
        elif viz_type == "3D Plot":
            # Select a third variable for z-axis
            z_options = [col for col in numeric_columns if col != x_column and col != y_column]
            
            if z_options:
                z_column = st.selectbox("Select Z-axis column:", z_options)
                
                # Filter data for 3D plot
                valid_data_3d = df[[x_column, y_column, z_column]].dropna()
                
                # Create 3D scatter plot
                fig = px.scatter_3d(
                    valid_data_3d,
                    x=x_column,
                    y=y_column,
                    z=z_column,
                    color=z_column,
                    title=f"3D Plot of {x_column} vs {y_column} vs {z_column}"
                )
                st.plotly_chart(fig)
            else:
                st.warning("Need at least one more numeric column for 3D plot.")
        
        # Show correlation statistics
        with st.expander("Correlation Statistics"):
            try:
                # Pearson correlation
                pearson_corr = valid_data[x_column].corr(valid_data[y_column], method='pearson')
                st.write(f"### Pearson correlation coefficient: {pearson_corr:.4f}")
                
                # Spearman rank correlation
                spearman_corr = valid_data[x_column].corr(valid_data[y_column], method='spearman')
                st.write(f"### Spearman rank correlation coefficient: {spearman_corr:.4f}")
                
                # Correlation test
                from scipy import stats
                pearson_r, p_value = stats.pearsonr(valid_data[x_column], valid_data[y_column])
                st.write(f"### Correlation test p-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    st.write("### The correlation is statistically significant (p < 0.05).")
                else:
                    st.write("### The correlation is not statistically significant (p >= 0.05).")
                    
                # R-squared
                st.write(f"### R-squared (coefficient of determination): {pearson_r**2:.4f}")
                
                # Interpretation
                st.write("### Correlation Interpretation")
                if abs(pearson_corr) < 0.3:
                    strength = "weak"
                elif abs(pearson_corr) < 0.7:
                    strength = "moderate"
                else:
                    strength = "strong"
                    
                direction = "positive" if pearson_corr > 0 else "negative"
                
                st.write(f"### This is a {strength} {direction} correlation.")
                
                # Pie chart of explained vs unexplained variance
                explained = pearson_r**2 * 100
                unexplained = 100 - explained
                
                fig = px.pie(
                    names=['Explained Variance', 'Unexplained Variance'],
                    values=[explained, unexplained],
                    title="Variance Explanation"
                )
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error calculating correlation statistics: {str(e)}")


def analyze_categorical_vs_numeric(df, x_column, y_column):
    """Analyze relationship between a categorical and a numeric column"""
    st.write(f"### Categorical vs Numeric Analysis: {y_column} by {x_column}")
    
    # Add a small sleep time for UI responsiveness
    import time
    time.sleep(0.1)
    
    # Offer multiple plot types
    plot_type = st.radio(
        "Select plot type:",
        options=["Box Plot", "Violin Plot", "Bar Chart", "Strip Plot", "Pie Charts"]
    )
    
    # Limit categories if too many
    categories = df[x_column].nunique()
    if categories > 20:
        top_n = st.slider("Show top N categories:", 5, 30, 10)
        top_cats = df[x_column].value_counts().nlargest(top_n).index
        filtered_df = df[df[x_column].isin(top_cats)]
        st.warning(f"Showing only top {top_n} categories out of {categories}.")
    else:
        filtered_df = df
    
    time.sleep(0.1)  # Small delay for UI responsiveness
    
    if plot_type == "Box Plot":
        # Group by option
        show_points = st.checkbox("Show all data points", value=True)
        
        fig = px.box(
            filtered_df,
            x=x_column,
            y=y_column,
            points="all" if show_points else False,
            title=f"{y_column} by {x_column}"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Violin Plot":
        # Option to show box inside violin
        show_box = st.checkbox("Show box plot inside", value=True)
        
        fig = px.violin(
            filtered_df,
            x=x_column,
            y=y_column,
            box=show_box,
            points="all",
            title=f"{y_column} by {x_column}"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Bar Chart":
        # Calculate mean, median or sum
        agg_method = st.selectbox(
            "Aggregation method:", 
            options=["Mean", "Median", "Sum", "Count", "Min", "Max"]
        )
        
        # Create aggregated dataframe
        if agg_method == "Mean":
            agg_df = filtered_df.groupby(x_column)[y_column].mean().reset_index()
        elif agg_method == "Median":
            agg_df = filtered_df.groupby(x_column)[y_column].median().reset_index()
        elif agg_method == "Sum":
            agg_df = filtered_df.groupby(x_column)[y_column].sum().reset_index()
        elif agg_method == "Count":
            agg_df = filtered_df.groupby(x_column)[y_column].count().reset_index()
        elif agg_method == "Min":
            agg_df = filtered_df.groupby(x_column)[y_column].min().reset_index()
        else:  # Max
            agg_df = filtered_df.groupby(x_column)[y_column].max().reset_index()
        
        # Sort by value or name
        sort_by = st.radio("Sort by:", ["Value", "Name"], horizontal=True)
        
        if sort_by == "Value":
            agg_df = agg_df.sort_values(y_column)
        else:
            # Try numeric sort if possible, otherwise lexicographic
            try:
                agg_df[x_column] = pd.to_numeric(agg_df[x_column])
                agg_df = agg_df.sort_values(x_column)
            except:
                agg_df = agg_df.sort_values(x_column)
        
        fig = px.bar(
            agg_df,
            x=x_column,
            y=y_column,
            title=f"{agg_method} of {y_column} by {x_column}",
            color=y_column,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Strip Plot":
        # Jitter option
        jitter = st.slider("Jitter amount:", 0.0, 1.0, 0.5)
        
        fig = px.strip(
            filtered_df,
            x=x_column,
            y=y_column,
            title=f"Strip Plot of {y_column} by {x_column}",
            color=x_column,
            jitter=jitter
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Pie Charts":
        # Create multiple pie charts by category
        max_pies = min(6, filtered_df[x_column].nunique())
        num_pies = st.slider("Number of categories to show:", 1, max_pies, 3)
        
        # Get top categories
        top_categories = filtered_df[x_column].value_counts().nlargest(num_pies).index
        
        # Create binned data for each category
        num_bins = st.slider("Number of bins for numeric values:", 3, 10, 5)
        
        col1, col2 = st.columns(2)
        
        for i, category in enumerate(top_categories):
            # Get data for this category
            cat_data = filtered_df[filtered_df[x_column] == category][y_column]
            
            # Create bins
            bins = pd.cut(cat_data, bins=num_bins)
            bin_counts = bins.value_counts().sort_index()
            
            # Create labels for the bins
            bin_labels = [f"{interval.left:.2f} to {interval.right:.2f}" for interval in bin_counts.index]
            
            # Create pie chart
            fig = px.pie(
                values=bin_counts.values,
                names=bin_labels,
                title=f"Distribution of {y_column} for {x_column}={category}"
            )
            
            # Display in alternating columns
            if i % 2 == 0:
                with col1:
                    st.plotly_chart(fig)
            else:
                with col2:
                    st.plotly_chart(fig)
    
    # ANOVA test if applicable
    with st.expander("Statistical Tests"):
        try:
            from scipy import stats
            
            # One-way ANOVA
            groups = [filtered_df[filtered_df[x_column] == cat][y_column].dropna() 
                     for cat in filtered_df[x_column].unique() if len(filtered_df[filtered_df[x_column] == cat]) > 0]
            
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                try:
                    f_val, p_val = stats.f_oneway(*groups)
                    st.write(f"One-way ANOVA F-statistic: {f_val:.4f}")
                    st.write(f"p-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.write("There is a statistically significant difference between groups (p < 0.05).")
                        
                        # Calculate effect size (eta-squared)
                        # Sum of squares between groups
                        grand_mean = filtered_df[y_column].mean()
                        ss_between = sum(len(g) * ((g.mean() - grand_mean) ** 2) for g in groups)
                        # Total sum of squares
                        ss_total = sum((filtered_df[y_column] - grand_mean) ** 2)
                        # Eta-squared
                        eta_squared = ss_between / ss_total
                        
                        st.write(f"Effect size (Eta-squared): {eta_squared:.4f}")
                        
                        if eta_squared < 0.01:
                            effect = "very small"
                        elif eta_squared < 0.06:
                            effect = "small"
                        elif eta_squared < 0.14:
                            effect = "medium"
                        else:
                            effect = "large"
                            
                        st.write(f"The effect size is {effect}.")
                        
                        # Pie chart of effect size vs unexplained variance
                        fig = px.pie(
                            names=['Explained Variance', 'Unexplained Variance'],
                            values=[eta_squared*100, (1-eta_squared)*100],
                            title="Variance Explanation by Category"
                        )
                        st.plotly_chart(fig)
                        
                    else:
                        st.write("There is no statistically significant difference between groups (p >= 0.05).")
                except Exception as e:
                    st.write(f"Error in ANOVA test: {str(e)}. This can happen if groups have very different variances or sizes.")
            else:
                st.write("Could not perform ANOVA test. Need at least two groups with data.")
        except ImportError:
            st.write("SciPy not available for statistical tests.")
        except Exception as e:
            st.write(f"Could not perform statistical test: {str(e)}")


def analyze_categorical_vs_categorical(df, x_column, y_column):
    """Analyze relationship between two categorical columns"""
    st.write(f"### Categorical Analysis: {y_column} vs {x_column}")
    
    # Add a small sleep time for UI responsiveness
    import time
    time.sleep(0.1)
    
    # Contingency table (crosstab)
    crosstab = pd.crosstab(df[x_column], df[y_column])
    
    # Limit categories if too many
    max_cats = 15
    if crosstab.shape[0] > max_cats or crosstab.shape[1] > max_cats:
        st.warning(f"Large contingency table ({crosstab.shape[0]}x{crosstab.shape[1]}). Showing reduced visualization.")
        
        # Take top categories for each
        if crosstab.shape[0] > max_cats:
            top_rows = df[x_column].value_counts().nlargest(max_cats).index
            crosstab = crosstab.loc[top_rows]
        
        if crosstab.shape[1] > max_cats:
            top_cols = df[y_column].value_counts().nlargest(max_cats).index
            crosstab = crosstab[top_cols]
    
    # Display the crosstab
    with st.expander("Contingency Table"):
        st.write(crosstab)
    
    # Visualization
    viz_type = st.radio(
        "Visualization type:",
        options=["Heatmap", "Stacked Bar", "Grouped Bar", "Mosaic Plot", "Parallel Categories", "Pie Charts"]
    )
    
    time.sleep(0.1)  # Small delay for UI responsiveness
    
    if viz_type == "Heatmap":
        # Normalization option
        norm_option = st.selectbox(
            "Normalization:",
            options=["Raw Counts", "Row Percentages", "Column Percentages", "Overall Percentages"]
        )
        
        # Prepare data based on normalization
        if norm_option == "Raw Counts":
            heatmap_data = crosstab
            title_suffix = "(Raw Counts)"
            text_format = ".0f"
        elif norm_option == "Row Percentages":
            heatmap_data = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            title_suffix = "(Row %)"
            text_format = ".1f"
        elif norm_option == "Column Percentages":
            heatmap_data = crosstab.div(crosstab.sum(axis=0), axis=1) * 100
            title_suffix = "(Column %)"
            text_format = ".1f"
        else:  # Overall Percentages
            total = crosstab.sum().sum()
            heatmap_data = crosstab / total * 100
            title_suffix = "(Overall %)"
            text_format = ".1f"
        
        fig = px.imshow(
            heatmap_data,
            text_auto=text_format,
            aspect="auto",
            title=f"Heatmap of {y_column} vs {x_column} {title_suffix}",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig)
    
    elif viz_type == "Stacked Bar":
        # Orientation option
        orientation = st.radio("Orientation:", ["Vertical", "Horizontal"], horizontal=True)
        
        # Normalization option
        normalize = st.checkbox("Show percentages", value=True)
        
        # Prepare data based on orientation and normalization
        if orientation == "Vertical":
            if normalize:
                # Normalize by row
                bar_data = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
                y_title = "Percentage"
            else:
                bar_data = crosstab
                y_title = "Count"
                
            fig = px.bar(
                bar_data,
                barmode="stack",
                title=f"Stacked Bar Chart of {y_column} by {x_column}",
                labels={"value": y_title}
            )
        else:  # Horizontal
            if normalize:
                # Normalize by column
                bar_data = crosstab.div(crosstab.sum(axis=0), axis=1) * 100
                x_title = "Percentage"
            else:
                bar_data = crosstab.T
                x_title = "Count"
                
            fig = px.bar(
                bar_data,
                barmode="stack",
                title=f"Stacked Bar Chart of {x_column} by {y_column}",
                labels={"value": x_title}
            )
        
        st.plotly_chart(fig)
    
    elif viz_type == "Grouped Bar":
        # Orientation option
        orientation = st.radio("Orientation:", ["Vertical", "Horizontal"], horizontal=True)
        
        # Convert to long format for grouped bars
        if orientation == "Vertical":
            # Melt the crosstab for plotting
            melted = crosstab.reset_index().melt(id_vars=x_column)
            
            fig = px.bar(
                melted,
                x=x_column,
                y="value",
                color=y_column,
                barmode="group",
                title=f"Grouped Bar Chart of {y_column} by {x_column}"
            )
        else:  # Horizontal
            # Transpose and melt
            melted = crosstab.T.reset_index().melt(id_vars=y_column)
            
            fig = px.bar(
                melted,
                x="value",
                y=y_column,
                color=x_column,
                barmode="group",
                title=f"Grouped Bar Chart of {x_column} by {y_column}"
            )
        
        st.plotly_chart(fig)
    
    elif viz_type == "Mosaic Plot":
        try:
            # Calculate proportions for mosaic plot
            from statsmodels.graphics.mosaicplot import mosaic
            
            # Create a figure
            plt.figure(figsize=(10, 8))
            mosaic(df, [x_column, y_column])
            plt.title(f"Mosaic Plot of {x_column} vs {y_column}")
            
            # Save and display
            st.pyplot(plt)
        except ImportError:
            st.error("Could not create mosaic plot: statsmodels not available.")
            # Show heatmap as fallback
            fig = px.imshow(
                crosstab,
                text_auto=True,
                title=f"Heatmap of {y_column} vs {x_column}"
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating mosaic plot: {str(e)}")
            # Show heatmap as fallback
            fig = px.imshow(
                crosstab,
                text_auto=True,
                title=f"Heatmap of {y_column} vs {x_column}"
            )
            st.plotly_chart(fig)
    
    elif viz_type == "Parallel Categories":
        try:
            # Create parallel categories diagram
            # Convert to long format
            parallel_data = df[[x_column, y_column]].dropna()
            
            fig = px.parallel_categories(
                parallel_data,
                dimensions=[x_column, y_column],
                color_continuous_scale="Viridis",
                title=f"Parallel Categories Plot of {x_column} vs {y_column}"
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating parallel categories plot: {str(e)}")
            # Show heatmap as fallback
            fig = px.imshow(
                crosstab,
                text_auto=True,
                title=f"Heatmap of {y_column} vs {x_column}"
            )
            st.plotly_chart(fig)
    
    elif viz_type == "Pie Charts":
        # For each value in x_column, show distribution of y_column
        # Limit to top 6 categories for x-axis to avoid too many pie charts
        if crosstab.shape[0] > 6:
            top_x = df[x_column].value_counts().nlargest(6).index
            st.warning(f"Showing pie charts for top 6 {x_column} categories only.")
        else:
            top_x = crosstab.index
        
        # Create multiple columns for pie charts
        cols = st.columns(2)  # 2 pie charts per row
        
        for i, x_val in enumerate(top_x):
            # Get distribution for this x value
            if x_val in crosstab.index:
                dist = crosstab.loc[x_val]
                
                # Create pie chart
                fig = px.pie(
                    names=dist.index,
                    values=dist.values,
                    title=f"Distribution of {y_column} for {x_column}={x_val}"
                )
                
                # Show in alternating columns
                with cols[i % 2]:
                    st.plotly_chart(fig)
    
    # Chi-square test
    with st.expander("Statistical Tests"):
        try:
            from scipy.stats import chi2_contingency
            
            # Perform chi-square test
            chi2, p, dof, expected = chi2_contingency(crosstab)
            
            st.write(f"Chi-square Test for Independence:")
            st.write(f"Chi-square statistic: {chi2:.4f}")
            st.write(f"p-value: {p:.4f}")
            st.write(f"Degrees of freedom: {dof}")
            
            if p < 0.05:
                st.write("There is a statistically significant association between variables (p < 0.05).")
            else:
                st.write("There is no statistically significant association between variables (p >= 0.05).")
            
            # Cramér's V for effect size
            n = crosstab.sum().sum()
            min_dim = min(crosstab.shape[0]-1, crosstab.shape[1]-1)
            if min_dim > 0:  # Avoid division by zero
                cramers_v = np.sqrt(chi2 / (n * min_dim))
                st.write(f"Cramér's V (effect size): {cramers_v:.4f}")
                
                # Interpret Cramér's V
                if cramers_v < 0.1:
                    strength = "negligible"
                elif cramers_v < 0.3:
                    strength = "weak"
                elif cramers_v < 0.5:
                    strength = "moderate"
                else:
                    strength = "strong"
                
                st.write(f"Association strength: {strength}")
                
                # Pie chart of explained vs unexplained variation
                explained = cramers_v**2 * 100
                unexplained = 100 - explained
                
                fig = px.pie(
                    names=['Explained Variation', 'Unexplained Variation'],
                    values=[explained, unexplained],
                    title="Variation Explained by Association"
                )
                st.plotly_chart(fig)
            
            # Expected vs observed
            if st.checkbox("Show expected vs observed counts"):
                # Create dataframe of expected frequencies
                expected_df = pd.DataFrame(
                    expected,
                    index=crosstab.index,
                    columns=crosstab.columns
                )
                
                st.write("Expected counts (if no association):")
                st.dataframe(expected_df.round(2))
                
                # Calculate residuals (observed - expected)
                residuals = crosstab - expected_df
                
                st.write("Residuals (observed - expected):")
                st.dataframe(residuals.round(2))
                
                # Heatmap of residuals
                fig = px.imshow(
                    residuals,
                    title="Residuals Heatmap (blue = more than expected, red = less than expected)",
                    color_continuous_scale="RdBu",
                    text_auto=".1f"
                )
                st.plotly_chart(fig)
                
        except ImportError:
            st.write("SciPy not available for chi-square test.")
        except Exception as e:
            st.write(f"Could not perform chi-square test: {str(e)}")


def analyze_datetime_vs_numeric(df, x_column, y_column):
    """Analyze relationship between a datetime and a numeric column"""
    st.write(f"### Time Analysis: {y_column} by {x_column}")
    
    # Add a small sleep time for UI responsiveness
    import time
    time.sleep(0.1)
    
    # Ensure column is datetime
    if df[x_column].dtype != 'datetime64[ns]':
        df[x_column] = pd.to_datetime(df[x_column], errors='coerce')
    
    # Time unit selection for grouping
    time_unit = st.selectbox(
        "Group by time unit:",
        options=["Day", "Week", "Month", "Quarter", "Year"]
    )
    
    # Aggregate method
    agg_method = st.selectbox(
        "Aggregation method:", 
        options=["Mean", "Median", "Sum", "Min", "Max", "Count"]
    )
    
    # Resample based on selected time unit
    if time_unit == "Day":
        period = "D"
    elif time_unit == "Week":
        period = "W"
    elif time_unit == "Month":
        period = "M"
    elif time_unit == "Quarter":
        period = "Q"
    else:  # Year
        period = "Y"
    
    # Ensure index is datetime for resampling
    temp_df = df[[x_column, y_column]].copy().dropna()
    temp_df.set_index(x_column, inplace=True)
    
    # Apply aggregation
    if agg_method == "Mean":
        resampled = temp_df.resample(period).mean()
    elif agg_method == "Median":
        resampled = temp_df.resample(period).median()
    elif agg_method == "Sum":
        resampled = temp_df.resample(period).sum()
    elif agg_method == "Min":
        resampled = temp_df.resample(period).min()
    elif agg_method == "Max":
        resampled = temp_df.resample(period).max()
    else:  # Count
        resampled = temp_df.resample(period).count()
    
    # Reset index for plotting
    resampled = resampled.reset_index()
    
    # Plot selection
    plot_type = st.radio(
        "Plot type:",
        options=["Line", "Bar", "Area", "Scatter", "Calendar Heatmap"]
    )
    
    time.sleep(0.1)  # Small delay for UI responsiveness
    
    if plot_type == "Line":
        # Line customization options
        markers = st.checkbox("Show markers", value=True)
        smoother = st.checkbox("Add smoother line", value=False)
        
        # Create figure
        fig = px.line(
            resampled,
            x=x_column,
            y=y_column,
            markers=markers,
            title=f"{agg_method} of {y_column} by {time_unit}"
        )
        
        # Add smoother if requested
        if smoother and len(resampled) > 5:
            # Moving average smoother
            window = max(3, min(11, len(resampled) // 5))
            window = window if window % 2 == 1 else window + 1  # ensure odd window
            
            resampled['smoothed'] = resampled[y_column].rolling(window=window, center=True).mean()
            
            fig.add_scatter(
                x=resampled[x_column],
                y=resampled['smoothed'],
                mode='lines',
                line=dict(color='red', width=3),
                name=f'Smoothed ({window}-point MA)'
            )
        
        st.plotly_chart(fig)
    
    elif plot_type == "Bar":
        fig = px.bar(
            resampled,
            x=x_column,
            y=y_column,
            title=f"{agg_method} of {y_column} by {time_unit}"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Area":
        fig = px.area(
            resampled,
            x=x_column,
            y=y_column,
            title=f"{agg_method} of {y_column} by {time_unit}"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Scatter":
        fig = px.scatter(
            resampled,
            x=x_column,
            y=y_column,
            size=y_column,
            title=f"{agg_method} of {y_column} by {time_unit}"
        )
        st.plotly_chart(fig)
    
    elif plot_type == "Calendar Heatmap":
        # Only applicable for daily, monthly data
        if time_unit in ["Day", "Month"]:
            try:
                # Extract year and month/day components
                if time_unit == "Day":
                    resampled['year'] = resampled[x_column].dt.year
                    resampled['month'] = resampled[x_column].dt.month
                    resampled['day'] = resampled[x_column].dt.day
                    
                    # Create pivot table
                    pivot = resampled.pivot_table(
                        index='day',
                        columns=['year', 'month'],
                        values=y_column,
                        aggfunc='mean'
                    )
                    
                    # Select specific year/month if too many
                    if pivot.shape[1] > 12:
                        # Get the most recent year with data
                        recent_year = resampled['year'].max()
                        
                        # Filter for most recent year
                        resampled_filtered = resampled[resampled['year'] == recent_year]
                        
                        pivot = resampled_filtered.pivot_table(
                            index='day',
                            columns='month',
                            values=y_column,
                            aggfunc='mean'
                        )
                        
                        title = f"Calendar Heatmap of {y_column} for Year {recent_year}"
                    else:
                        title = f"Calendar Heatmap of {y_column}"
                    
                else:  # Month
                    resampled['year'] = resampled[x_column].dt.year
                    resampled['month'] = resampled[x_column].dt.month
                    
                    # Create pivot table
                    pivot = resampled.pivot_table(
                        index='month',
                        columns='year',
                        values=y_column,
                        aggfunc='mean'
                    )
                    
                    title = f"Monthly Heatmap of {y_column} by Year"
                
                fig = px.imshow(
                    pivot,
                    labels=dict(
                        x='Month' if time_unit == 'Day' else 'Year',
                        y='Day' if time_unit == 'Day' else 'Month',
                        color=y_column
                    ),
                    title=title,
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error creating calendar heatmap: {str(e)}")
                # Fall back to line chart
                fig = px.line(
                    resampled,
                    x=x_column,
                    y=y_column,
                    markers=True,
                    title=f"{agg_method} of {y_column} by {time_unit}"
                )
                st.plotly_chart(fig)
        else:
            st.warning("Calendar heatmap is only available for daily or monthly data. Showing line chart instead.")
            fig = px.line(
                resampled,
                x=x_column,
                y=y_column,
                markers=True,
                title=f"{agg_method} of {y_column} by {time_unit}"
            )
            st.plotly_chart(fig)
    
    # Show rolling average option
    add_rolling = st.checkbox("Add rolling average")
    if add_rolling:
        window = st.slider("Rolling window size:", 2, min(20, len(resampled)//2), 7)
        
        # Calculate rolling average
        resampled['Rolling Avg'] = resampled[y_column].rolling(window=window).mean()
        
        # Create plot with both series
        fig = px.line(
            resampled,
            x=x_column,
            y=[y_column, 'Rolling Avg'],
            title=f"{agg_method} of {y_column} by {time_unit} with Rolling Average (window={window})",
            markers=True
        )
        st.plotly_chart(fig)
    
    # Seasonality analysis
    if st.checkbox("Show seasonality analysis") and len(resampled) >= 4:
        st.write("### Seasonality Analysis")
        
        try:
            # Get appropriate time component based on unit
            if time_unit == "Day":
                resampled['component'] = resampled[x_column].dt.dayofweek
                component_name = "Day of Week"
                component_labels = {
                    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                    3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                }
            elif time_unit == "Month":
                resampled['component'] = resampled[x_column].dt.month
                component_name = "Month"
                component_labels = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 
                    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
            elif time_unit == "Quarter":
                resampled['component'] = resampled[x_column].dt.quarter
                component_name = "Quarter"
                component_labels = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
            else:
                # For week and year, use month as seasonal component
                resampled['component'] = resampled[x_column].dt.month
                component_name = "Month"
                component_labels = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 
                    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
            
            # Group by seasonal component
            seasonal = resampled.groupby('component')[y_column].mean().reset_index()
            
            # Map labels if defined
            if component_labels:
                seasonal['label'] = seasonal['component'].map(component_labels)
                if None not in seasonal['label'].values:  # Check if mapping was successful
                    seasonal = seasonal.sort_values('component')
                else:  # Fall back to numeric if mapping failed
                    seasonal['label'] = seasonal['component'].astype(str)
            else:
                seasonal['label'] = seasonal['component'].astype(str)
            
            # Create bar chart of seasonal pattern
            fig = px.bar(
                seasonal,
                x='label',
                y=y_column,
                title=f"Average {y_column} by {component_name}",
                labels={"label": component_name, "y": f"Average {y_column}"}
            )
            st.plotly_chart(fig)
            
            # Add pie chart of seasonal distribution
            fig = px.pie(
                seasonal,
                names='label',
                values=y_column,
                title=f"Distribution of {y_column} by {component_name}"
            )
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error in seasonality analysis: {str(e)}")
    
    # Show trend analysis
    with st.expander("Trend Analysis"):
        try:
            from scipy import stats
            
            # Calculate trend (linear regression)
            # Convert to numeric for regression
            resampled['numeric_time'] = np.arange(len(resampled))
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                resampled['numeric_time'], resampled[y_column]
            )
            
            # Calculate percentage change
            first_value = resampled[y_column].iloc[0]
            last_value = resampled[y_column].iloc[-1]
            
            # Avoid division by zero
            if first_value != 0:
                pct_change = ((last_value - first_value) / first_value) * 100
            else:
                pct_change = float('inf') if last_value > 0 else float('-inf') if last_value < 0 else 0
            
            st.write(f"Linear Trend Analysis:")
            st.write(f"Slope: {slope:.4f} (per time unit)")
            st.write(f"R-squared: {r_value**2:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            st.write(f"Total change: {last_value - first_value:.4f}")
            st.write(f"Percentage change: {pct_change:.2f}%")
            
            if p_value < 0.05:
                st.write("There is a statistically significant trend (p < 0.05).")
                
                # Add trend direction
                if slope > 0:
                    st.write(f"The trend is **increasing** over time.")
                else:
                    st.write(f"The trend is **decreasing** over time.")
                
                # Add trend line to visualization
                resampled['trend'] = intercept + slope * resampled['numeric_time']
                
                fig = px.scatter(
                    resampled,
                    x=x_column,
                    y=y_column,
                    title=f"Trend Analysis of {y_column} over Time"
                )
                
                # Add trend line
                fig.add_scatter(
                    x=resampled[x_column],
                    y=resampled['trend'],
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2)
                )
                
                st.plotly_chart(fig)
                
                # Pie chart of explained vs unexplained variance
                explained = r_value**2 * 100
                unexplained = 100 - explained
                
                fig = px.pie(
                    names=['Explained by Trend', 'Unexplained Variation'],
                    values=[explained, unexplained],
                    title="Variation Explained by Time Trend"
                )
                st.plotly_chart(fig)
                
            else:
                st.write("There is no statistically significant trend (p >= 0.05).")
        except ImportError:
            st.write("Required libraries not available for trend analysis.")
        except Exception as e:
            st.write(f"Error in trend analysis: {str(e)}")


def generate_correlation_heatmap(df, method='pearson', figsize=(12, 10), 
                                cmap='coolwarm', annot=True, mask_upper=False,
                                title='Correlation Matrix'):
    """
    Generate a correlation heatmap for a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    method : str, default 'pearson'
        The correlation method to use ('pearson', 'kendall', 'spearman')
    figsize : tuple, default (12, 10)
        Figure size (width, height) in inches
    cmap : str or matplotlib colormap, default 'coolwarm'
        The colormap to use for the heatmap
    annot : bool, default True
        Whether to annotate the heatmap with correlation values
    mask_upper : bool, default False
        Whether to mask the upper triangle of the heatmap
    title : str, default 'Correlation Matrix'
        The title of the heatmap
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Create mask for upper triangle if requested
    mask = np.zeros_like(corr_matrix, dtype=bool)
    if mask_upper:
        mask[np.triu_indices_from(mask, k=1)] = True
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask if mask_upper else None,
                cmap=cmap, annot=annot, fmt='.2f', square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    # Set title and adjust layout
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    return fig, ax

def generate_missing_data_heatmap(df, figsize=(12, 10), cmap='viridis'):
    """
    Generate a heatmap showing missing values in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to visualize missing values for
    figsize : tuple, default (12, 10)
        Figure size (width, height) in inches
    cmap : str or matplotlib colormap, default 'viridis'
        The colormap to use for the heatmap
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Create a boolean mask where True indicates missing values
    missing = df.isnull()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    sns.heatmap(missing, cmap=cmap, yticklabels=False, 
                cbar_kws={'label': 'Missing Data'}, ax=ax)
    
    # Set title and labels
    plt.title('Missing Value Heatmap', fontsize=16, pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.tight_layout()
    
    return fig, ax

def generate_feature_density_heatmap(df, features=None, bins=30, figsize=(14, 12)):
    """
    Generate a density heatmap for selected features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    features : list or None, default None
        List of features to include in the heatmap. If None, all numeric features are used.
    bins : int, default 30
        Number of bins for the histogram
    figsize : tuple, default (14, 12)
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib figure object
    """
    # Select numeric columns if features not specified
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Ensure all specified features exist in the dataframe
        for feature in features:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found in the DataFrame")
    
    # Determine number of rows and columns for subplot grid
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Generate density heatmaps for each pair of features
    for i, feature1 in enumerate(features):
        for j, ax in enumerate(axes):
            if j == i:
                # Create a histogram for the feature on the diagonal
                sns.histplot(df[feature1].dropna(), kde=True, ax=ax)
                ax.set_title(f'Distribution of {feature1}')
            elif j < i:
                # Calculate 2D histogram
                feature2 = features[j]
                hist, x_edges, y_edges = np.histogram2d(
                    df[feature1].dropna(), 
                    df[feature2].dropna(), 
                    bins=bins
                )
                # Plot heatmap
                sns.heatmap(
                    hist.T, 
                    ax=ax,
                    cmap='viridis',
                    cbar_kws={'label': 'Count'}
                )
                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
            else:
                # Hide unused axes
                ax.axis('off')
    
    plt.tight_layout()
    return fig

def generate_categorical_count_heatmap(df, categorical_columns=None, figsize=(12, 10)):
    """
    Generate a heatmap showing counts of categorical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    categorical_columns : list or None, default None
        List of categorical columns to include. If None, all object and category dtypes are used.
    figsize : tuple, default (12, 10)
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Select categorical columns if not specified
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        raise ValueError("No categorical columns found in the DataFrame")
    
    # Create a dataframe to store value counts
    count_df = pd.DataFrame()
    
    # Calculate value counts for each categorical column
    for col in categorical_columns:
        # Get top 10 most common values
        value_counts = df[col].value_counts().nlargest(10)
        count_df[col] = value_counts
        count_df.index = pd.MultiIndex.from_product([[col], value_counts.index])
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    sns.heatmap(count_df, annot=True, fmt='g', cmap='YlGnBu', ax=ax)
    
    # Set title and adjust layout
    plt.title('Categorical Variable Counts', fontsize=16, pad=20)
    plt.tight_layout()
    
    return fig, ax

# Example usage
"""
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'education_years': np.random.normal(16, 3, 1000),
        'satisfaction': np.random.normal(7, 2, 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'gender': np.random.choice(['Male', 'Female', 'Other'], 1000),
        'customer_type': np.random.choice(['New', 'Returning', 'Loyal'], 1000)
    }
    
    # Introduce some missing values
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'satisfaction'] = np.nan
    
    # Create correlation between some variables
    df['spending'] = df['income'] * 0.3 + np.random.normal(0, 5000, 1000)
    df['health_score'] = 100 - df['age'] * 0.5 + np.random.normal(0, 10, 1000)
    
    # Generate and show heatmaps
    # 1. Correlation heatmap
    fig1, ax1 = generate_correlation_heatmap(df, mask_upper=True, 
                                            title='Correlation Heatmap of Numeric Features')
    plt.savefig('correlation_heatmap.png')
    
    # 2. Missing data heatmap
    fig2, ax2 = generate_missing_data_heatmap(df)
    plt.savefig('missing_data_heatmap.png')
    
    # 3. Feature density heatmap for selected features
    fig3 = generate_feature_density_heatmap(df, 
                                           features=['age', 'income', 'satisfaction', 'spending'],
                                           bins=25)
    plt.savefig('feature_density_heatmap.png')
    
    # 4. Categorical count heatmap
    fig4, ax4 = generate_categorical_count_heatmap(df)
    plt.savefig('categorical_count_heatmap.png')
    
    plt.show()"
    """