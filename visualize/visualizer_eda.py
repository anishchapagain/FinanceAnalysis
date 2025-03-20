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
    
    # Select analysis type
    # analysis_type = st.radio(
    #     "Select Analysis Type:",
    #     options=["Exploratory Data Analysis", "Univariate Analysis", "Bivariate Analysis", 
    #             "Correlation Analysis", "Time Series Analysis"]
    # )
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
    st.write("### Data Distribution Overview")
    
    # For numeric columns - show distribution
    if numeric_columns:
        # Option to select which numeric columns to view
        selected_numeric = st.multiselect(
            "Select numeric columns to visualize:",
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
    
    # For categorical columns - show value counts
    if categorical_columns:
        # Option to select which categorical columns to view
        st.divider()
        selected_categorical = st.multiselect(
            "**Select categorical columns to visualize:**",
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
            st.write("**Categorical Summary Statistics:**")
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
        selected_datetime = st.selectbox(
            "Select datetime column to visualize:",
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
                st.write(f"Distribution by time components:")
                
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
    
    # Outlier detection for numeric columns
    if numeric_columns:
        st.divider()
        st.write("### Outlier Detection")
        
        # Option to select which numeric column to view
        outlier_col = st.selectbox(
            "Select numeric column for outlier detection:",
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
            bin_count = st.slider("Number of bins:", min_value=5, max_value=50, value=20, key="histogram_bins")
            
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
                title=f"Box Plot of {selected_column}"
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
            
            st.write(f"Outliers: {len(outliers)} ({len(outliers)/len(valid_data)*100:.2f}%)")
            
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


def analyze_numeric_vs_numeric(df, x_column, y_column):
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


def analyze_categorical_vs_numeric(df, x_column, y_column):
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


def analyze_categorical_vs_categorical(df, x_column, y_column):
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


def analyze_datetime_vs_numeric(df, x_column, y_column):
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


def perform_bivariate_analysis(df, numeric_columns, categorical_columns, datetime_columns):
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