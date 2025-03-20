def visualize_numeric_columns(df):
    """
    Creates interactive charts for numeric columns in a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing data to visualize
    
    Returns:
    None: Displays charts directly in the Streamlit app
    """
    # Get all numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_columns:
        st.warning("No numeric columns found in the provided DataFrame.")
        return
    
    # Let user select a numeric column to visualize
    selected_column = st.selectbox(
        "Select a numeric column to visualize:",
        options=numeric_columns
    )
    
    if selected_column:
        st.subheader(f"Distribution of {selected_column}")
        
        # Choose visualization type
        viz_type = st.radio(
            "Select visualization type:",
            options=["Histogram", "Box Plot", "Bar Chart (Top Values)"]
        )
        
        # Filter out NaN values
        valid_data = df[~df[selected_column].isna()]
        
        if viz_type == "Histogram":
            # Create histogram with adjustable bins
            bin_count = st.slider("Number of bins:", min_value=5, max_value=50, value=20)
            
            fig = px.histogram(
                valid_data, 
                x=selected_column, 
                nbins=bin_count,
                color_discrete_sequence=['#3366CC'],
                title=f"Histogram of {selected_column}"
            )
            st.plotly_chart(fig)
            
            # Add basic statistics
            st.write("Basic Statistics:")
            stats = df[selected_column].describe()
            st.write(stats)
            
        elif viz_type == "Box Plot":
            fig = px.box(
                valid_data, 
                y=selected_column,
                title=f"Box Plot of {selected_column}"
            )
            st.plotly_chart(fig)
            
            # Show outliers info
            q1 = np.percentile(valid_data[selected_column], 25)
            q3 = np.percentile(valid_data[selected_column], 75)
            iqr = q3 - q1
            outlier_cutoff_low = q1 - 1.5 * iqr
            outlier_cutoff_high = q3 + 1.5 * iqr
            
            outliers = valid_data[(valid_data[selected_column] < outlier_cutoff_low) | 
                                 (valid_data[selected_column] > outlier_cutoff_high)]
            
            if not outliers.empty:
                st.write(f"Number of outliers: {len(outliers)}")
                if st.checkbox("Show outliers"):
                    st.write(outliers)
            
        elif viz_type == "Bar Chart (Top Values)":
            # For bar charts, limit to top N values
            top_n = st.slider("Show top N values:", min_value=5, max_value=50, value=10)
            
            # Check if we need to group values
            if len(valid_data[selected_column].unique()) > 100:
                st.warning(f"Column has {len(valid_data[selected_column].unique())} unique values. Using bins for visualization.")
                
                # Create bins for large number of unique values
                counts, bins = np.histogram(valid_data[selected_column], bins=top_n)
                bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]
                binned_data = pd.DataFrame({
                    'bin': bin_labels,
                    'count': counts
                })
                
                fig = px.bar(
                    binned_data,
                    x='bin',
                    y='count',
                    title=f"Distribution of {selected_column} (Binned)",
                    color='count',
                    color_continuous_scale='Blues'
                )
            else:
                # Get value counts for this column
                value_counts = valid_data[selected_column].value_counts().nlargest(top_n).reset_index()
                value_counts.columns = ['Value', 'Count']
                
                fig = px.bar(
                    value_counts,
                    x='Value',
                    y='Count',
                    title=f"Top {top_n} values for {selected_column}",
                    color='Count',
                    color_continuous_scale='Blues'
                )
            
            st.plotly_chart(fig)
    
    # Option to explore correlations if multiple numeric columns exist
    if len(numeric_columns) > 1 and st.checkbox("Explore correlations between numeric columns"):
        st.subheader("Correlation Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("X-axis:", options=numeric_columns, key="x_column")
        with col2:
            y_column = st.selectbox("Y-axis:", options=numeric_columns, index=1 if len(numeric_columns) > 1 else 0, key="y_column")
        
        if x_column and y_column:
            fig = px.scatter(
                df, 
                x=x_column, 
                y=y_column,
                title=f"{x_column} vs {y_column}",
                opacity=0.6,
                trendline="ols" if st.checkbox("Show trend line") else None
            )
            st.plotly_chart(fig)
            
            # Show correlation coefficient
            correlation = df[[x_column, y_column]].corr().iloc[0,1]
            st.write(f"Correlation coefficient: {correlation:.4f}")


def analyze_dataframe(df):
    """
    Comprehensive visualization and analysis tool for pandas DataFrames.
    Handles numeric, categorical, and datetime fields with correlation analysis.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame to analyze
    
    Returns:
    None: Displays analysis directly in the Streamlit app
    """
    st.title("DataFrame Analysis Dashboard")
    
    # Display basic info about dataframe
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
        st.dataframe(df.head())
        
        # Display column types
        st.write("Column Data Types:")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtypes_df)
    
    # Identify column types
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Try to detect datetime columns (both actual datetime dtype and string dates)
    datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Check for string columns that may contain dates
    for col in categorical_columns:
        if col not in datetime_columns:
            try:
                # Try to convert a sample to datetime
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample and pd.to_datetime(sample, errors='coerce') is not pd.NaT:
                    datetime_columns.append(col)
            except (ValueError, TypeError):
                pass
    
    # Remove detected datetime columns from categorical list
    categorical_columns = [col for col in categorical_columns if col not in datetime_columns]
    
    # Select analysis type
    analysis_type = st.radio(
        "Select Analysis Type:",
        options=["Univariate Analysis", "Bivariate Analysis", "Correlation Analysis", "Time Series Analysis"]
    )
    
    if analysis_type == "Univariate Analysis":
        st.subheader("Univariate Analysis")
        
        # Allow user to select column type first
        column_type = st.radio(
            "Select Column Type:",
            options=["Numeric", "Categorical", "Datetime"]
        )
        
        if column_type == "Numeric" and numeric_columns:
            selected_column = st.selectbox(
                "Select a numeric column:",
                options=numeric_columns
            )
            
            # Numeric visualization type
            viz_type = st.selectbox(
                "Visualization type:",
                options=["Histogram", "Box Plot", "Violin Plot", "KDE Plot"]
            )
            
            # Filter out NaN values
            valid_data = df[~df[selected_column].isna()]
            
            if viz_type == "Histogram":
                # Create histogram with adjustable bins
                bin_count = st.slider("Number of bins:", min_value=5, max_value=50, value=20)
                
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
                fig = ff.create_distplot(
                    [valid_data[selected_column].dropna()],
                    [selected_column],
                    show_hist=False,
                    curve_type='kde'
                )
                fig.update_layout(title=f"KDE Plot of {selected_column}")
                st.plotly_chart(fig)
            
            # Display statistics
            with st.expander("Statistical Summary"):
                stats = df[selected_column].describe()
                st.write(stats)
                
                # Additional stats
                st.write(f"Skewness: {df[selected_column].skew():.4f}")
                st.write(f"Kurtosis: {df[selected_column].kurtosis():.4f}")
                
                # Check for outliers
                q1 = np.percentile(valid_data[selected_column], 25)
                q3 = np.percentile(valid_data[selected_column], 75)
                iqr = q3 - q1
                outlier_low = q1 - 1.5 * iqr
                outlier_high = q3 + 1.5 * iqr
                outliers = valid_data[(valid_data[selected_column] < outlier_low) | 
                                     (valid_data[selected_column] > outlier_high)]
                
                st.write(f"Outliers: {len(outliers)} ({len(outliers)/len(valid_data)*100:.2f}%)")
        
        elif column_type == "Categorical" and categorical_columns:
            selected_column = st.selectbox(
                "Select a categorical column:",
                options=categorical_columns
            )
            
            # Get value counts
            value_counts = df[selected_column].value_counts()
            
            # Limit categories shown if there are too many
            max_categories = st.slider("Max categories to display:", 5, 50, 20)
            if len(value_counts) > max_categories:
                value_counts = value_counts.nlargest(max_categories)
                st.warning(f"Showing only top {max_categories} categories out of {df[selected_column].nunique()}.")
            
            # Plot bar chart
            fig = px.bar(
                value_counts,
                title=f"Distribution of {selected_column}",
                labels={"index": selected_column, "value": "Count"},
                color=value_counts.values,
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig)
            
            # Show pie chart if categories are not too many
            if len(value_counts) <= 10:
                fig = px.pie(
                    names=value_counts.index,
                    values=value_counts.values,
                    title=f"Proportion of {selected_column}"
                )
                st.plotly_chart(fig)
            
            # Show statistics
            with st.expander("Category Statistics"):
                st.write(f"Total categories: {df[selected_column].nunique()}")
                st.write(f"Most common: {df[selected_column].mode()[0]} ({value_counts.iloc[0]} occurrences)")
                st.write(f"Missing values: {df[selected_column].isna().sum()} ({df[selected_column].isna().sum()/len(df)*100:.2f}%)")
                
                # Show full table
                st.write("Full category distribution:")
                st.dataframe(pd.DataFrame({
                    'Category': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / value_counts.sum() * 100).round(2)
                }))
        
        elif column_type == "Datetime" and datetime_columns:
            selected_column = st.selectbox(
                "Select a datetime column:",
                options=datetime_columns
            )
            
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
                options=["Year", "Month", "Day", "Hour"]
            )
            
            if time_unit == "Year":
                date_counts = valid_dates[selected_column].dt.year.value_counts().sort_index()
            elif time_unit == "Month":
                date_counts = valid_dates[selected_column].dt.to_period('M').value_counts().sort_index()
                date_counts.index = date_counts.index.astype(str)
            elif time_unit == "Day":
                date_counts = valid_dates[selected_column].dt.date.value_counts().sort_index()
                date_counts.index = date_counts.index.astype(str)
            else:  # Hour
                date_counts = valid_dates[selected_column].dt.hour.value_counts().sort_index()
            
            # Plot the distribution
            fig = px.bar(
                date_counts,
                title=f"Distribution of {selected_column} by {time_unit}",
                labels={"index": time_unit, "value": "Count"},
                color=date_counts.values,
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig)
            
            # Show line chart for time trends
            fig = px.line(
                x=date_counts.index,
                y=date_counts.values,
                markers=True,
                title=f"Trend of {selected_column} by {time_unit}"
            )
            st.plotly_chart(fig)
            
            # Show statistics
            with st.expander("Datetime Statistics"):
                st.write(f"Date range: {df[selected_column].min()} to {df[selected_column].max()}")
                st.write(f"Time span: {(df[selected_column].max() - df[selected_column].min()).days} days")
                st.write(f"Missing values: {df[selected_column].isna().sum()} ({df[selected_column].isna().sum()/len(df)*100:.2f}%)")
        
        else:
            st.warning(f"No {column_type.lower()} columns found in the dataframe.")
    
    elif analysis_type == "Bivariate Analysis":
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
            
            elif (x_type == "Categorical" and y_type == "Numeric") or (x_type == "Numeric" and y_type == "Categorical"):
                # Ensure categorical is x and numeric is y
                if x_type == "Numeric" and y_type == "Categorical":
                    x_column, y_column = y_column, x_column
                    x_type, y_type = y_type, x_type
                
                # Offer multiple plot types
                plot_type = st.radio(
                    "Select plot type:",
                    options=["Box Plot", "Violin Plot", "Bar Chart"]
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
            
            elif x_type == "Categorical" and y_type == "Categorical":
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
            
            elif (x_type == "Datetime" and y_type == "Numeric") or (y_type == "Datetime" and x_type == "Numeric"):
                # Make sure datetime is on x-axis
                if y_type == "Datetime":
                    x_column, y_column = y_column, x_column
                
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
                        
                        # Seasonality detection if we have enough data points
                        if len(resampled) >= 24:  # Need enough data for seasonal analysis
                            # Decompose the time series
                            from statsmodels.tsa.seasonal import seasonal_decompose
                            
                            # Ensure no missing values for decomposition
                            decomp_df = resampled.copy()
                            decomp_df[y_column] = decomp_df[y_column].fillna(method='ffill').fillna(method='bfill')
                            
                            # Try to decompose with appropriate period
                            if time_unit == "Day":
                                period = 7  # Weekly seasonality
                            elif time_unit == "Week":
                                period = 4  # Monthly seasonality
                            elif time_unit == "Month":
                                period = 12  # Yearly seasonality
                            else:
                                period = 4  # Quarterly seasonality
                            
                            try:
                                result = seasonal_decompose(decomp_df[y_column], period=period)
                                
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
                                
                                # Calculate seasonality strength
                                seasonal_strength = 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)
                                st.write(f"Seasonality strength: {seasonal_strength:.4f}")
                                
                                if seasonal_strength > 0.6:
                                    st.write("Strong seasonal pattern detected.")
                                elif seasonal_strength > 0.3:
                                    st.write("Moderate seasonal pattern detected.")
                                else:
                                    st.write("Weak or no seasonal pattern detected.")
                            except Exception as e:
                                st.write(f"Could not perform seasonal decomposition: {str(e)}")
                    except ImportError:
                        st.write("Required libraries not available for trend analysis.")
                    except Exception as e:
                        st.write(f"Error in trend analysis: {str(e)}")
            
            else:
                st.warning("The selected column combination doesn't have a predefined visualization method.")
        else:
            st.warning("Please select valid columns for both axes.")
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        if len(numeric_columns) < 2:
            st.warning("Need at least two numeric columns for correlation analysis.")
        else:
            # Select correlation method
            corr_method = st.radio(
                "Correlation method:",
                options=["Pearson", "Spearman", "Kendall"]
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
            
            # Feature selection based on correlation
            st.subheader("Feature Selection")
            
            # Target variable for feature importance
            target = st.selectbox(
                "Select target variable:",
                options=numeric_columns
            )
            
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
                        st.dataframe(high_corr_df)
                else:
                    st.write(f"No feature pairs with correlation > {threshold}")
                
                # Option to show scatter plots
                if st.checkbox("Show scatter plots for top correlated features"):
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
    
    elif analysis_type == "Time Series Analysis":
        st.subheader("Time Series Analysis")
        
        if not datetime_columns:
            st.warning("No datetime columns detected in the dataframe.")
        else:
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
            
            if target_col:
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
                
                # Decomposition
                with st.expander("Time Series Decomposition"):
                    # Check if we have enough data points
                    if len(resampled) < 2 * max(7, 12, 4):  # Minimum 2 periods
                        st.warning("Not enough data points for decomposition.")
                    else:
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
                        except Exception as e:
                            st.error(f"Error in decomposition: {str(e)}")
                
                # Autocorrelation
                with st.expander("Autocorrelation Analysis"):
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
                                    st.write("The differenced series is still not stationary (p > 0.05).")
                    except ImportError:
                        st.error("Required libraries not available for autocorrelation analysis.")
                    except Exception as e:
                        st.error(f"Error in autocorrelation analysis: {str(e)}")
                
                # Rolling statistics
                with st.expander("Rolling Statistics"):
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
                
                # Simple forecasting
                with st.expander("Simple Forecasting"):
                    st.write("Simple forecasting models:")
                    
                    try:
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing
                        
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
                            
                            if forecast_type == "Exponential Smoothing":
                                # Fit exponential smoothing model
                                model = ExponentialSmoothing(
                                    train[target_col],
                                    trend='add',
                                    seasonal='add',
                                    seasonal_periods=12 if time_unit == "Month" else 4
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
                                
                                # Calculate error metrics
                                from sklearn.metrics import mean_absolute_error, mean_squared_error
                                
                                mae = mean_absolute_error(test[target_col], forecast)
                                rmse = np.sqrt(mean_squared_error(test[target_col], forecast))
                                mape = np.mean(np.abs((test[target_col] - forecast) / test[target_col])) * 100
                                
                                st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                                st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                            
                            elif forecast_type == "Moving Average":
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
                                
                                # Calculate error metrics
                                from sklearn.metrics import mean_absolute_error, mean_squared_error
                                
                                mae = mean_absolute_error(test[target_col], ma_forecast)
                                rmse = np.sqrt(mean_squared_error(test[target_col], ma_forecast))
                                mape = np.mean(np.abs((test[target_col].values - np.array(ma_forecast)) / test[target_col].values)) * 100
                                
                                st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                                st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                            
                            else:  # Last Value
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
                                from sklearn.metrics import mean_absolute_error, mean_squared_error
                                
                                mae = mean_absolute_error(test[target_col], lv_forecast)
                                rmse = np.sqrt(mean_squared_error(test[target_col], lv_forecast))
                                mape = np.mean(np.abs((test[target_col].values - np.array(lv_forecast)) / test[target_col].values)) * 100
                                
                                st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                                st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                        else:
                            st.warning("Not enough data for test set after splitting.")
                    except ImportError:
                        st.error("Required libraries not available for forecasting.")
                    except Exception as e:
                        st.error(f"Error in forecasting: {str(e)}")