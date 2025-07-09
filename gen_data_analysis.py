import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.llm_client = None
        self.api_key = None
        
    def set_openai_key(self, api_key):
        """Set the OpenAI API key for LLM features"""
        self.api_key = api_key
        self.llm_client = OpenAI(api_key="sk-or-v1-...")
    
    def load_data(self, file):
        """Load data from various file formats"""
        if file.name.endswith('.csv'):
            self.df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            self.df = pd.read_json(file)
        return self.df
    
    def get_data_info(self):
        """Get basic information about the dataset"""
        if self.df is None:
            return None
        
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'unique_values': {col: self.df[col].nunique() for col in self.df.columns},
            'info': info_str,
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': self.df.select_dtypes(include=['datetime']).columns.tolist(),
        }
        return info
    
    def get_summary_statistics(self):
        """Calculate summary statistics for numerical columns"""
        if self.df is None:
            return None
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
        
        return numeric_df.describe()
    
    def get_correlation_matrix(self):
        """Calculate correlation matrix for numerical columns"""
        if self.df is None:
            return None
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
        
        return numeric_df.corr()
    
    def plot_histogram(self, column):
        """Create histogram for specified column"""
        if self.df is None or column not in self.df.columns:
            return None
        
        fig = px.histogram(self.df, x=column, title=f'Histogram of {column}')
        return fig
    
    def plot_boxplot(self, column):
        """Create boxplot for specified column"""
        if self.df is None or column not in self.df.columns:
            return None
        
        fig = px.box(self.df, y=column, title=f'Boxplot of {column}')
        return fig
    
    def plot_scatter(self, x_col, y_col, color_col=None):
        """Create scatter plot for specified columns"""
        if self.df is None or x_col not in self.df.columns or y_col not in self.df.columns:
            return None
        
        if color_col and color_col in self.df.columns:
            fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, 
                           title=f'Scatter Plot: {x_col} vs {y_col}')
        else:
            fig = px.scatter(self.df, x=x_col, y=y_col, 
                           title=f'Scatter Plot: {x_col} vs {y_col}')
        return fig
    
    def plot_correlation_heatmap(self):
        """Create correlation heatmap for numerical columns"""
        if self.df is None:
            return None
        
        corr_matrix = self.get_correlation_matrix()
        if corr_matrix is None:
            return None
        
        fig = px.imshow(corr_matrix, 
                       title='Correlation Heatmap',
                       color_continuous_scale='RdBu_r')
        return fig
    
    def plot_pairplot(self, columns=None, n_max=5):
        """Create pair plot for specified columns (limited to n_max)"""
        if self.df is None:
            return None
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None
        
        if columns:
            plot_cols = [col for col in columns if col in numeric_cols]
        else:
            plot_cols = numeric_cols
        
        # Limit to max n_max columns to avoid excessive computation
        if len(plot_cols) > n_max:
            plot_cols = plot_cols[:n_max]
            
        if len(plot_cols) < 2:
            return None
        
        # Create a figure with subplots
        fig = go.Figure()
        rows = len(plot_cols)
        cols = len(plot_cols)
        
        # Fill in scatter plots in the upper triangle and histograms on diagonal
        for i, col1 in enumerate(plot_cols):
            for j, col2 in enumerate(plot_cols):
                if i == j:  # Diagonal - histogram
                    fig.add_trace(go.Histogram(
                        x=self.df[col1],
                        name=col1,
                        showlegend=False
                    ))
                elif i < j:  # Upper triangle - scatter plot
                    fig.add_trace(go.Scatter(
                        x=self.df[col1],
                        y=self.df[col2],
                        mode='markers',
                        name=f'{col1} vs {col2}',
                        showlegend=False,
                        marker=dict(
                            size=6,
                            opacity=0.5
                        )
                    ))
        
        # Create a grid layout and update the figure
        fig.update_layout(
            title='Pair Plot Matrix',
            grid=dict(rows=rows, columns=cols, pattern='independent'),
            height=200 * rows,
            width=200 * cols
        )
        
        return fig
    
    def clean_data(self, columns_to_drop=None, handle_missing='drop'):
        """Clean data by removing columns and handling missing values"""
        if self.df is None:
            return None
        
        # Create a copy to avoid modifying the original
        cleaned_df = self.df.copy()
        
        # Drop columns if specified
        if columns_to_drop:
            cleaned_df = cleaned_df.drop(columns=[col for col in columns_to_drop if col in cleaned_df.columns])
        
        # Handle missing values
        if handle_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
        elif handle_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif handle_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif handle_missing == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        
        self.df = cleaned_df
        return self.df
    
    def feature_engineering(self, operations=None):
        """Perform feature engineering operations"""
        if self.df is None:
            return None
        
        if not operations:
            return self.df
        
        # Create a copy to avoid modifying the original
        engineered_df = self.df.copy()
        
        for op in operations:
            if op['type'] == 'scaling' and op.get('columns'):
                columns = [col for col in op['columns'] if col in engineered_df.columns and pd.api.types.is_numeric_dtype(engineered_df[col])]
                if not columns:
                    continue
                
                if op.get('method') == 'standard':
                    scaler = StandardScaler()
                    engineered_df[columns] = scaler.fit_transform(engineered_df[columns])
                elif op.get('method') == 'minmax':
                    scaler = MinMaxScaler()
                    engineered_df[columns] = scaler.fit_transform(engineered_df[columns])
            
            elif op['type'] == 'encoding' and op.get('columns'):
                columns = [col for col in op['columns'] if col in engineered_df.columns]
                if not columns:
                    continue
                
                if op.get('method') == 'onehot':
                    for col in columns:
                        one_hot = pd.get_dummies(engineered_df[col], prefix=col, drop_first=op.get('drop_first', False))
                        engineered_df = pd.concat([engineered_df, one_hot], axis=1)
                        if op.get('drop_original', True):
                            engineered_df = engineered_df.drop(col, axis=1)
                
                elif op.get('method') == 'label':
                    for col in columns:
                        engineered_df[f'{col}_encoded'] = engineered_df[col].astype('category').cat.codes
                        if op.get('drop_original', False):
                            engineered_df = engineered_df.drop(col, axis=1)
            
            elif op['type'] == 'binning' and op.get('column') and op.get('bins'):
                column = op['column']
                bins = op['bins']
                labels = op.get('labels')
                
                if column in engineered_df.columns and pd.api.types.is_numeric_dtype(engineered_df[column]):
                    engineered_df[f'{column}_binned'] = pd.cut(engineered_df[column], bins=bins, labels=labels)
                    if op.get('drop_original', False):
                        engineered_df = engineered_df.drop(column, axis=1)
            
            elif op['type'] == 'pca' and op.get('columns') and op.get('n_components'):
                columns = [col for col in op['columns'] if col in engineered_df.columns and pd.api.types.is_numeric_dtype(engineered_df[col])]
                if not columns or len(columns) < 2:
                    continue
                
                n_components = min(op['n_components'], len(columns))
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(engineered_df[columns])
                
                for i in range(n_components):
                    engineered_df[f'PCA_{i+1}'] = pca_result[:, i]
                
                if op.get('drop_original', False):
                    engineered_df = engineered_df.drop(columns, axis=1)
        
        self.df = engineered_df
        return self.df
    
    def llm_data_insights(self, specific_question=None):
        """Generate insights about the data using LLM"""
        if self.df is None or self.llm_client is None:
            return "Please load data and set up the OpenAI API key first."
        
        # Sample the data to avoid sending too much data
        sample_rows = min(20, len(self.df))
        sample_df = self.df.sample(sample_rows)
        
        # Get basic data info to help the LLM understand the dataset structure
        info = self.get_data_info()
        stats = self.get_summary_statistics()
        
        # Prepare the prompt
        prompt = f"Dataset overview: {len(self.df)} rows and {len(self.df.columns)} columns.\n"
        prompt += f"Columns: {', '.join(self.df.columns.tolist())}\n\n"
        prompt += "Sample data:\n"
        prompt += sample_df.to_string() + "\n\n"
        
        if stats is not None:
            prompt += "Summary statistics:\n"
            prompt += stats.to_string() + "\n\n"
        
        if specific_question:
            prompt += f"Question: {specific_question}\n"
            prompt += "Please provide a detailed analysis and answer to the question based on the data."
        else:
            prompt += "Please provide key insights about this dataset, including:\n"
            prompt += "1. Potential patterns or trends you observe\n"
            prompt += "2. Interesting correlations or relationships\n"
            prompt += "3. Potential feature engineering suggestions\n"
            prompt += "4. Data quality issues that might need attention\n"
            prompt += "5. Recommended visualizations or analyses to explore further"
        
        # Get response from OpenAI
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",  # or another appropriate model
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant. Analyze the provided dataset and generate insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def suggest_visualizations(self):
        """Use LLM to suggest appropriate visualizations based on the data"""
        if self.df is None or self.llm_client is None:
            return "Please load data and set up the OpenAI API key first."
        
        info = self.get_data_info()
        
        prompt = f"Dataset has {len(self.df)} rows and {len(self.df.columns)} columns.\n"
        prompt += f"Numeric columns: {info['numeric_columns']}\n"
        prompt += f"Categorical columns: {info['categorical_columns']}\n"
        prompt += f"Based on this information, suggest 3-5 specific and appropriate data visualizations that would be most insightful for this dataset. For each suggestion, explain what insights it might reveal."
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert. Suggest appropriate visualizations for the given dataset."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating visualization suggestions: {str(e)}"
    
    def get_csv_download_link(self):
        """Generate a download link for the current dataframe"""
        if self.df is None:
            return None
        
        csv = self.df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'data:file/csv;base64,{b64}'


def main():
    st.set_page_config(page_title="Data Analysis App", page_icon="📊", layout="wide")
    
    st.title("🔍 Data Analysis App with LLM Capabilities")
    st.markdown("---")
    
    # Initialize the analyzer
    analyzer = DataAnalyzer()
    
    # Session state initialization
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = analyzer
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json'])
        
        if uploaded_file is not None:
            try:
                with st.spinner('Loading data...'):
                    st.session_state.df = analyzer.load_data(uploaded_file)
                    st.session_state.analyzer = analyzer
                st.success(f"Successfully loaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        st.header("🤖 LLM Settings")
        with st.expander("OpenAI API Key"):
            api_key = st.text_input("Enter OpenAI API Key", type="password")
            if st.button("Set API Key"):
                try:
                    analyzer.set_openai_key(api_key)
                    st.session_state.api_key_set = True
                    st.success("API key set successfully")
                except Exception as e:
                    st.error(f"Error setting API key: {str(e)}")
    
    # Main area
    if st.session_state.df is not None:
        df = st.session_state.df
        analyzer = st.session_state.analyzer
        
        # Tabs for different functions
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🧹 Data Cleaning", "🔧 Feature Engineering", 
                                              "📈 Visualizations", "🤖 LLM Insights"])
        
        with tab1:
            st.header("Dataset Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Sample")
                st.dataframe(df.head())
                
                st.subheader("Shape")
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                
                if analyzer.get_csv_download_link():
                    st.markdown(f"[Download Current Data as CSV]({analyzer.get_csv_download_link()})")
            
            with col2:
                st.subheader("Data Types and Missing Values")
                info = analyzer.get_data_info()
                
                missing_df = pd.DataFrame({
                    'Column': info['missing_values'].keys(),
                    'Missing Values': info['missing_values'].values(),
                    'Type': [info['dtypes'][col] for col in info['missing_values'].keys()]
                })
                st.dataframe(missing_df)
            
            st.subheader("Summary Statistics")
            st.dataframe(analyzer.get_summary_statistics())
            
            st.subheader("Correlation Matrix")
            corr_matrix = analyzer.get_correlation_matrix()
            if corr_matrix is not None and not corr_matrix.empty:
                fig = analyzer.plot_correlation_heatmap()
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Data Cleaning")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Drop Columns")
                columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
            
            with col2:
                st.subheader("Handle Missing Values")
                missing_method = st.radio("Method", ["drop", "mean", "median", "mode"], horizontal=True)
            
            if st.button("Clean Data"):
                with st.spinner("Cleaning data..."):
                    cleaned_df = analyzer.clean_data(columns_to_drop, missing_method)
                    st.success("Data cleaned successfully!")
                    st.dataframe(cleaned_df.head())
            
        with tab3:
            st.header("Feature Engineering")
            
            feature_ops = []
            
            with st.expander("Scaling", expanded=False):
                scale_cols = st.multiselect("Select columns to scale", 
                                          df.select_dtypes(include=[np.number]).columns.tolist(),
                                          key="scale_cols")
                scale_method = st.radio("Scaling method", ["standard", "minmax"], horizontal=True)
                
                if st.button("Apply Scaling"):
                    feature_ops.append({
                        'type': 'scaling',
                        'method': scale_method,
                        'columns': scale_cols
                    })
            
            with st.expander("Encoding", expanded=False):
                cat_cols = st.multiselect("Select categorical columns to encode", 
                                        df.select_dtypes(include=['object', 'category']).columns.tolist(),
                                        key="cat_cols")
                encode_method = st.radio("Encoding method", ["onehot", "label"], horizontal=True)
                drop_original = st.checkbox("Drop original columns", value=True)
                
                if st.button("Apply Encoding"):
                    feature_ops.append({
                        'type': 'encoding',
                        'method': encode_method,
                        'columns': cat_cols,
                        'drop_original': drop_original,
                        'drop_first': True if encode_method == 'onehot' else False
                    })
            
            with st.expander("Binning", expanded=False):
                num_cols_bin = st.selectbox("Select column for binning", 
                                          df.select_dtypes(include=[np.number]).columns.tolist())
                
                num_bins = st.slider("Number of bins", min_value=2, max_value=10, value=4)
                
                if num_cols_bin:
                    min_val = float(df[num_cols_bin].min())
                    max_val = float(df[num_cols_bin].max())
                    
                    bin_edges = np.linspace(min_val, max_val, num_bins+1)
                    bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
                    
                    st.write("Bin edges:", bin_edges)
                    st.write("Bin labels:", bin_labels)
                    
                    if st.button("Apply Binning"):
                        feature_ops.append({
                            'type': 'binning',
                            'column': num_cols_bin,
                            'bins': bin_edges.tolist(),
                            'labels': bin_labels,
                            'drop_original': st.checkbox("Drop original binning column", value=False)
                        })
            
            with st.expander("PCA", expanded=False):
                pca_cols = st.multiselect("Select columns for PCA", 
                                        df.select_dtypes(include=[np.number]).columns.tolist(),
                                        key="pca_cols")
                
                if pca_cols:
                    n_components = st.slider("Number of components", min_value=1, max_value=min(len(pca_cols), 10), value=2)
                    
                    if st.button("Apply PCA"):
                        feature_ops.append({
                            'type': 'pca',
                            'columns': pca_cols,
                            'n_components': n_components,
                            'drop_original': st.checkbox("Drop original PCA columns", value=False)
                        })
            
            if feature_ops and st.button("Apply All Feature Engineering"):
                with st.spinner("Engineering features..."):
                    engineered_df = analyzer.feature_engineering(feature_ops)
                    st.success("Features engineered successfully!")
                    st.dataframe(engineered_df.head())
        
        with tab4:
            st.header("Data Visualizations")
            
            viz_type = st.selectbox("Select Visualization Type", 
                                  ["Histogram", "Boxplot", "Scatter Plot", "Correlation Heatmap", "Pair Plot"])
            
            if viz_type == "Histogram":
                hist_col = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns.tolist())
                if hist_col:
                    fig = analyzer.plot_histogram(hist_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Boxplot":
                box_col = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns.tolist())
                if box_col:
                    fig = analyzer.plot_boxplot(box_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Scatter Plot":
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X Column", df.select_dtypes(include=[np.number]).columns.tolist())
                with col2:
                    y_col = st.selectbox("Y Column", df.select_dtypes(include=[np.number]).columns.tolist())
                with col3:
                    color_options = ["None"] + df.columns.tolist()
                    color_col = st.selectbox("Color By", color_options)
                
                if x_col and y_col:
                    color = None if color_col == "None" else color_col
                    fig = analyzer.plot_scatter(x_col, y_col, color)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Correlation Heatmap":
                fig = analyzer.plot_correlation_heatmap()
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough numeric columns for correlation analysis")
            
            elif viz_type == "Pair Plot":
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                selected_cols = st.multiselect("Select Columns (max 5)", num_cols, 
                                             default=num_cols[:min(3, len(num_cols))])
                
                if selected_cols:
                    fig = analyzer.plot_pairplot(selected_cols)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Select at least 2 numeric columns for pair plot")
        
        with tab5:
            st.header("LLM-Powered Data Insights")
            
            if not st.session_state.api_key_set:
                st.warning("Please set your OpenAI API key in the sidebar to use LLM features")
            else:
                st.subheader("Ask about your data")
                question = st.text_input("Ask a specific question about your data")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Data Insights"):
                        with st.spinner("Generating insights..."):
                            insights = analyzer.llm_data_insights()
                            st.markdown(insights)
                
                with col2:
                    if st.button("Suggest Visualizations"):
                        with st.spinner("Generating visualization suggestions..."):
                            viz_suggestions = analyzer.suggest_visualizations()
                            st.markdown(viz_suggestions)
                
                if question and st.button("Answer Question"):
                    with st.spinner("Analyzing..."):
                        answer = analyzer.llm_data_insights(question)
                        st.markdown(answer)
    
    else:
        st.info("👈 Please upload a data file to get started.")
        
        st.markdown("""
        ## Welcome to the Data Analysis App!
        
        This application allows you to:
        
        - 📊 Explore your data with automatic summaries and visualizations
        - 🧹 Clean your data by handling missing values and dropping unnecessary columns
        - 🔧 Apply feature engineering techniques like scaling, encoding, binning, and PCA
        - 📈 Create various visualizations to gain insights
        - 🤖 Use LLM capabilities to get AI-powered insights and suggestions
        
        ### Getting Started
        
        1. Upload your dataset using the file uploader in the sidebar
        2. (Optional) Set your OpenAI API key to use the LLM features
        3. Explore the different tabs to analyze and transform your data
        
        ### Supported File Types
        
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - JSON (.json)
        """)


if __name__ == "__main__":
    main()