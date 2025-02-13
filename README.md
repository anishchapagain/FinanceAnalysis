<!-- @format -->

# Financial Data Analysis Tool
A powerful Python-based application for analyzing financial and business data through interactive visualizations, AI-powered insights, and automated reporting.

## Features
- 📊 Interactive data visualizations
- 🤖 AI-powered data analysis
- 📝 Automated report generation
- 💬 Natural language querying
- 📈 Multi-file data processing
- 🔄 Real-time data updates

## Prerequisites
1. **Python Environment**

   - Python 3.8 or higher
   - pip package manager

2. **OpenAI API Key**

   - Required for default GPT-4 model
   - Get it from [OpenAI Platform](https://platform.openai.com)

3. **Ollama (Optional)**
   - Required for local Llama3.2 model
   - Follow installation steps below

## Installation
**Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

**Install Ollama (Optional, for local Llama2)**

   ```bash
   # macOS or Linux
   curl https://ollama.ai/install.sh | sh

   # Windows
   # Download from https://ollama.ai/download
   ```

5. **Pull Llama3.2 model (Optional)**
   ```bash
   ollama pull llama3.2
   ```

## Usage

1. **Start the application**

   ```bash
   streamlit run app.py
   ```

2. **Upload CSV files**
   - Click "Browse files" button
   - Select one or more CSV files
   - Wait for processing to complete

3. **Explore visualizations**
   - View time series plots
   - Analyze category distributions
   - Compare monthly trends

4. **Chat with your data**
   - Ask questions in natural language
   - Get AI-powered insights
   - Generate detailed reports

## Data Format Requirements

### CSV Structure

- Must include at least one date column
- Must include numeric columns for analysis
- Optional categorical columns for grouping

### Column Types
1. **Date Columns**
   - ISO format (YYYY-MM-DD)
   - Consistent formatting

2. **Numeric Columns**
   - Decimal numbers
   - No currency symbols
   - Percentages as decimals

3. **Categorical Columns**
   - Text data
   - Consistent naming
   - Case-sensitive

## Features in Detail

### 1. Data Processing
- Automatic type inference
- Missing value handling
- Data validation
- Aggregation management

### 2. Visualizations
- Time series analysis
- Category distributions
- Monthly comparisons
- Interactive charts

### 3. AI Integration
- Natural language processing
- Context-aware responses
- Multi-model support
- Automated insights

### 4. Report Generation
- PDF format
- Executive summaries
- Key metrics
- Downloadable reports

## Model Options

### 1. OpenAI GPT-4 (Default)

- Requires API key
- Cloud-based processing
- Best for complex analysis

### 2. Local Llama2 (Optional)

- Runs locally
- No API key needed
- Privacy-focused option

## Performance Optimization

1. **Data Management**

- Batch processing
- Efficient storage
- Memory optimization
- Cache management

2. **Query Processing**

- Indexed searches
- Result caching
- Parallel processing
- Optimized aggregations

## Security

1. **API Security**

- Secure key storage
- Regular rotation
- Access control
- Rate limiting

2. **Data Privacy**

- Local processing
- No external storage
- Session isolation
- Secure cleanup