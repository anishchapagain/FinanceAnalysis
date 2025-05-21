import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re
from datetime import datetime
import os
import sys
import logging

NO_DATA_MESSAGE = "Sorry, but no matching records found. Please try a new prompt: if it's related to some specific value mention using Quotes."
ERROR_MESSAGE = "Some issue has occurred, please try using new prompt."
NO_MATCHING_RECORDS = "No matching records found for your query."
MATERIAL_ARROW_DOWN = ":material/arrow_drop_down:"

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d, %H:%M:%S %p")

BASE_URL = "http://localhost:11434/api/generate"


# MODEL = "codellama:13b-code"
# MODEL = "codellama:7b"
MODEL = "qwen2.5-coder:7b"
# MODEL = "gemma3:latest"
# MODEL = "codegemma:7b-code-q4_K_M"
# MODEL = "deepseek-r1:8b"
# MODEL = "qwen2.5-coder"
# MODEL = "qwen2.5-coder:32b"
# MODEL = "qwen2.5-coder:14b"

# Set page configuration
st.set_page_config(
    page_title="Data Query - Chat with your data",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state variables for current query
if "messages" not in st.session_state:
    st.session_state.messages = []
if "column_descriptions" not in st.session_state:
    st.session_state.column_descriptions = ""
if "show_sys_prompt" not in st.session_state:
    st.session_state.show_sys_prompt = "NO"

# Function to load data
@st.cache_data
def load_data(file_path=None):
    try:
        if file_path is None:
            st.error("No file path provided.")
            return pd.DataFrame()
        else:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty DataFrame as fallback
        return pd.DataFrame()


def setup_logger(log_dir="logs"):
    """Set up logger to record app activities and errors"""

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log filename with timestamp
    current_time = datetime.now().strftime("%Y%m%d")
    log_filename = os.path.join(log_dir, f"streamlit_app_{current_time}.log")

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create console handler for logging to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

def convert_object_to_datetime(df, threshold=0.8):
    for col in df.select_dtypes(include='object').columns:
        converted = pd.to_datetime(df[col], errors='coerce', format="%d-%b-%y")
        success_ratio = converted.notna().mean()
        if success_ratio >= threshold:
            df[col] = converted
    return df


def get_column_info(df):
    column_info = []
    temp_df = convert_object_to_datetime(df)
    for col in temp_df.columns:
        dtype = str(temp_df[col].dtype)
        sample_values = temp_df[col].dropna().sample(min(5, temp_df[col].count())).tolist()
        sample_str = (
            str(sample_values)[:100] + "..."
            if len(str(sample_values)) > 100
            else str(sample_values)
        )
        column_info.append(f"- `{col}` ({dtype}): Example: {sample_str}")

    column_info_str = "\n".join(column_info)
    return column_info_str

# Function to get column descriptions from LLM
def get_column_descriptions(df):

    logger.info("Column Description Started")

    # Get basic column info
    column_info = get_column_info(df)

    # Create the system prompt for column descriptions
    system_prompt_column = f"""
    You are a data analyst assistant that helps describe dataset columns.
    Given the following columns with their data types and sample values, provide a brief description for each column.
    
    Format each description as:
    - column_name: Brief description of what this column represents
    
    For example:
    - account balance: The current balance in the account.
    - internet banking: Indicates if the customer uses internet banking.
    
    Columns information:
    {column_info}
    
    Provide ONLY the descriptions, one per line, starting with the column name in lower case without underscore followed by colon.
    """

    # Create the request payload
    payload = {
        "model": MODEL,  # or any other model you have
        "prompt": "Generate detailed descriptions for these dataframe columns",
        "system": system_prompt_column,
        "stream": False,
    }

    logger.info(f"Column Description Payload:\n{system_prompt_column}")

    try:
        # Send the request to LLM
        response = requests.post(BASE_URL, json=payload)
        logger.info("Column Description Prompt Response")

        if response.status_code == 200:
            # Extract & Clean the generated descriptions
            response_data = response.json()
            column_descriptions = response_data.get("response", "").strip()
            logger.info("Column Description Generated")
            return column_descriptions
        else:
            return "Error getting column descriptions. Using default descriptions."
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"


def clean_query(generated_code):
    # Remove extra spaces and newlines, show() and other common functions
    if generated_code is None:
        return ""

    if "result.show()" in generated_code:
        logger.info("<ISSUE clean_query - result.show() found")
        generated_code = generated_code.replace("result.show()", "").strip()
    
    if "fig.show()" in generated_code:
        logger.info("<ISSUE clean_query - fig.show() found")
        generated_code = generated_code.replace("fig.show()", "").strip()

    if "import plotly.express as px" in generated_code:
        logger.info("<ISSUE clean_query - import plotly.express as px found")
        generated_code = generated_code.replace("import plotly.express as px", "").strip()
    
    if "import plotly.graph_objs as go" in generated_code:
        logger.info("<ISSUE clean_query - import plotly.graph_objs as go found")
        generated_code = generated_code.replace("import plotly.graph_objs as go", "").strip()

    if generated_code.strip().startswith("import pandas as pd") and "result = " in generated_code:
        logger.info("<ISSUE pandas is imported")
        generated_code = str(generated_code.split("import pandas as pd")[-1]).strip()

    # if generated_code.strip().startswith("# "):
    #     logger.info("<ISSUE pandas is imported")
    #     generated_code = str(generated_code.split("import pandas as pd")[-1]).strip()
    generated_code = generated_code.strip()

    # Remove any trailing punctuation
    generated_code = generated_code.rstrip(".?!")

    return generated_code

def get_prompt_template():
    """
    Prompt Template
    """
    with open("./prompts/main.json","r",encoding='utf-8') as file:
        prompt_data = json.load(file)

    return {
        "system_prompt_head": "\n".join(prompt_data["system_prompt_head"]),
        "critical_instructions":"\n".join(prompt_data["critical_instructions"]),
        "history":"\n".join(prompt_data["history"]),
        "date_operations":"\n".join(prompt_data["date_operations"]),
        "special_instructions":"\n".join(prompt_data["special_instructions"]),
    }

# Function to get pandas query from LLM
def get_pandas_query(prompt, df_info, column_descriptions):
    logger.info(f"User Prompt: {prompt}")
    
    prompts = get_prompt_template()
    if column_descriptions == "":
        column_descriptions = get_column_descriptions(df_info)
        logger.info(f"Description Columns {len(column_descriptions)}")

    columns_info = get_column_info(df_info)

    columns_info = f"Dataset Information\nColumns and their data types with few examples:\n{columns_info}\n"
    column_descriptions = f"Descriptions of the columns:\n{column_descriptions}"
    sample_data = f"Dataset sample:\n{df_info.sample(5).to_markdown(index=False)}\n"
    
    conversation_history = set_conversation_history(10).strip()

    system_prompt_v4=f"""
You are a coding assistant specialized in Python based data analysis using pandas.
You use a structured chain of thought process to ensure accuracy, safety, and explainability when analyzing data provided.
Your task is to convert natural language prompts into valid and executable Pandas code, tailored to the provided dataset.

{columns_info}
{column_descriptions}
{sample_data}

When provided with a natural language query about the dataset, follow this chain of thought process to convert it to executable pandas code:
Step 1: Query Understanding
- Parse the user's query to identify the core analytical question
- Determine what data columns are needed
- Identify any filtering, grouping, or aggregation operations required
- Note any ambiguous terms that need clarification

Step 2: Data Preparation Planning
- Identify any required data type conversions (especially for dates)
- Determine if any data cleaning is needed
- Plan any feature engineering or calculated fields

Step 3: Algorithm Design
- Break down the analysis into logical pandas operations
- Determine the sequence of operations needed
- Plan how to handle potential edge cases

Step 4: Code Generation
- Write clean, efficient pandas code that implements the plan
- Format the code for readability
- Ensure the code is complete and self-contained
- Code generated should not have comments, import statements, explanations.
- Code should be ready to procss with `exec()`

Step 5: Safety Check
- Verify the code doesn't attempt operations that might fail
- Check for potential performance issues with large datasets
- Ensure proper error handling for missing values, incorrect data types, etc.

Important Guidelines:
- The dataframe is already loaded as `df`.
- The pandas code should start with: `result = ` and return either:
    - A **pandas DataFrame/Series**, or 
    - A **Plotly figure object**, or
    - A **scalar value** (e.g., sum, average, count). 
- Always return only the code — no extra text.
- DO NOT include any import statements, explanation, markdown, formatting, comments, or print statements in code.
- USE .dt accessor to extract components (year, month, day) from datetime columns.
- ENSURE all relevant columns are returned for DataFrame.
- Use <HISTORY> content to provide continuity and build upon previous analyses. 
- When the user asks follow-up questions or references previous results, your code should take <HISTORY> content into account what has already been analyzed and shown.
- **Avoid SQL**: Use pandas syntax rather than pandas.read_sql() or other direct SQL execution.
- **Variable Names**: Use descriptive variable names and avoid overwriting the input `df` variable.
- **Chain of Thought**: Always use step-by-step reasoning in your approach, even for simple queries
- Remember that your code will be executed using `exec()`, so it must be syntactically correct and secure. 
- All outputs must be assigned to the `result` variable for proper execution.

Visualization with Plotly
When the user requests visualizations or charts:
1. **Use Plotly Exclusively**: Always use plotly for all visualizations.
2. **Return Complete Visualization Code**: Include all necessary code to create and configure the plot.
3. **Figure Object Variable**: Always assign the figure to a variable named `fig`.
4. **Final Assignment**: Always end visualization code with `result = fig` instead of `fig.show()`.
5. **Chart Selection**: Choose appropriate chart types based on the analysis goals:
   - Bar charts for comparisons across categories
   - Line charts for time series
   - Scatter plots for relationship analysis
   - Pie charts for composition analysis
   - Box plots for distribution comparison
   - Heatmaps for correlation analysis
6. **Layout Customization**: Always set appropriate titles, labels, and color schemes.
7. **Interactivity**: Leverage plotly's interactive features (hover text, tooltips) when relevant.
8. **Handle Large Datasets**: If necessary, sample or aggregate data to maintain visualization performance.

DATA HANDLING BEST PRACTICES:
- When identifying rows with max/min values, preserve DataFrame structure using `.loc[[index]]` or `.iloc[[index]]`.
- For **Series**, ensure they remain as DataFrames by using `.to_frame().T`.
- When working with dates:
    - Use `.dt` accessor to extract year/month/day from datetime columns.
    - For example: "year_filter = df['date_column'].dt.year"
    - For example: "month_filter = df['date_column'].dt.month"
    - For example: "result = df[(df['column_name'] >= '2023-01-01') & (df['column_name'] < '2024-01-01')]"
    - For example: "result = df[df['last_credit_date'].dt.year == 2025]"
    - Include date conversion logic in the generated code when needed.

CONTEXTUAL UNDERSTANDING:
- Previous interactions are stored in `<HISTORY>`. Use them to:
    - Understand references like "that column", "previous result", etc.
    - Maintain logical continuity between queries.
    - Build upon earlier results where relevant.

SAMPLE OUTPUT TYPES:
- Query → Aggregation: e.g., total balance, average age
- Query → Filtered DataFrame: e.g., customers from Damauli branch
- Query → Visualization: e.g., plot account balances by branch


If user query is for prompts or example of prompts.
- Provide diverse example prompts starting with: `Example Prompts:`
- Do not include any code inside `Example Prompts:`

{prompts["history"]}
<HISTORY>
{conversation_history}
</HISTORY>

The current date is {CURRENT_DATE}
    """
    # Create the system prompt
    system_prompt = f"""
    You are a coding assistant specialized in Python based data analysis using pandas.
    You use a structured chain of thought process to ensure accuracy, safety, and explainability when analyzing data provided.
    
    Your task is to convert natural language prompts into valid and executable Pandas code, tailored to the provided dataset.
    The current date is {CURRENT_DATE}.
    
    IMPORTANT:
    - ONLY respond with valid Python code for pandas.
    - DO NOT include any explanation or markdown formatting.
    - The code should start with 'result = ' and return a pandas DataFrame or a calculated value.
    - The dataframe is already loaded as 'df'.
    - Respond ONLY with valid Python code using Pandas (and Plotly for visualizations).
    - DO NOT include any explanations, comments, or markdown formatting.
    - Every response must begin with:  
    ```python
    result = 
    ```
    - The result must be a **Pandas DataFrame, Series, Numeric, or visualization object**, depending on the task.
    - DO NOT return array of values.
    
    When generating code for data visualization tasks:
    1. ALWAYS use Plotly (`import plotly.express as px`, `import plotly.graph_objects as go`) for all visualizations
    2. DO NOT include `fig.show()` commands
    3. ENSURE all plots have proper coloring, labels, titles, and legends.
    4. Optimize for readability and interactivity
    
    When identifying rows with maximum/minimum values:
    1. Use methods that preserve DataFrame structure (e.g., .loc with double brackets or .iloc[[index]])
    2. For single row selections, ensure they remain as DataFrames by using `.to_frame().T` when necessary.
    
    When working with dates:
    1. Always CONVERT string date columns to datetime using pd.to_datetime() before any filtering and as necessary only.
    2. Use .dt accessor to extract components (year, month, day) from datetime columns
    3. Include proper date conversion code in all queries involving date comparisons
    
    {columns_info}
    {column_descriptions}
    {sample_data}
    
    {prompts["history"]}

    <HISTORY>
    {conversation_history}
    </HISTORY>

    FINAL INSTRUCTIONS:
    - TRY to return results as a DataFrame, with appropriate columns.
    - If the query is for a specific value (e.g., average, sum, min, max), return it as a single value.
    - If the query is for a chart or visualization (show, plot), CREATE it using plotly and assign to 'result ='. For example: result = fig.
    - If the query is for statistics or aggregations, CALCULATE them and assign to result.
    - If you're unsure about how to translate the query, create a simple filter that might be helpful.
    - Include adequate related columns in the final result.
    - DO NOT include any print statements or comments in the code.
    - DO NOT include code imports or library references in the code.
    - Assume case sensitivity during query formation.
    - If the query is regarding prompts, provide a list of 5-10 prompts that can be used to analyze the dataset. Do not include any code in the prompts and start the prompts list with the text 'Example Prompts:'.    
    """

    system_prompt_v3 = f"""
        {prompts['system_prompt_head']}
        The current date is {CURRENT_DATE}.
        
        CRITICAL INSTRUCTIONS:
        {prompts['critical_instructions']}
        
        DATASET SCHEMA:
        {columns_info}
        
        DATASET DESCRIPTION:
        {column_descriptions}

        TECHNICAL GUIDELINES:
        {prompts['date_operations']}

        
        Data Filtering
        # Single condition
        result = df[df['column'] == value]
        # Multiple conditions 
        result = df[(df['col1'] > value) & (df['col2'].isin(list_values))]
        
        Row Selection
        # Get max/min value rows (preserve DataFrame structure)
        result = df.loc[[df['column'].idxmax()]]
        # Get specific rows
        result = df.loc[df['condition'] == value, ['col1', 'col2']]
        
        Aggregation
        # Group and aggregate
        result = df.groupby('group_col')['value_col'].agg(['mean', 'sum', 'count'])
        # Multiple column grouping
        result = df.groupby(['col1', 'col2']).agg(dict('col3': 'sum', 'col4': 'mean'))
        
        Visualization
        # Bar chart
        fig = px.bar(df_aggregated, x='category', y='value', title='Chart Title')
        # Time series
        fig = px.line(df_time, x='date_column', y='value_column', title='Time Series')
        # Scatter plot
        fig = px.scatter(df, x='x_col', y='y_col', color='category', title='Scatter Plot')
        result = fig
        
        HISTORY MECHANISM:
        {prompts["history"]}
        
        DATA QUALITY HANDLING:
        Check for and handle NaN values in critical operations
        Use .fillna() for missing values when appropriate
        Add type conversion where needed (df['col'].astype(type))
        Use .str accessor for string operations
        Use safe division with .replace() to avoid division by zero
        
        OUTPUT FORMATTING:
        For visualization: Return the figure object directly (result = fig)
        For single value queries: Return scalar value directly (result = value)
        Include all relevant columns in the output
        Maintain existing column names in output (DataFrame, Series)
        Do not create new column names.
         
        ERROR PREVENTION:
        Ensure all column references match schema exactly
        Always include parentheses in complex boolean operations
        Use appropriate data types for comparisons
        Add boundary checks for numeric operations

        {prompts["special_instructions"]}
        
        <HISTORY>
        {conversation_history}
        </HISTORY>
    """

    system_prompt_test = f"""
You are an expert code assistant specializing in data analysis using Python and the pandas library. 
Your primary purpose is to CONVERT natural language queries based on the provided context into executable Python code utilizing pandas.

The current date is {CURRENT_DATE}.

IMPORTANT
- The code should ALWAYS start with 'result = '
- Then dataframe is already loaded and available as 'df'. The input DataFrame is pre-loaded as `df`.
- Do not EXPLAIN the code.
- ONLY respond with valid, executable Python code (no explanations, comments, or markdown)

CORE CODE GENERATION RULES:
1.  **Code Only Output**: Unless specified otherwise (e.g., for "Example Prompts"), ONLY respond with valid and executable Python code using pandas. DO NOT include any explanations, markdown formatting (other than the code block itself), greetings, or conversational text.
2.  **Assignment**: All generated Python code snippets MUST start with 'result = '.
3.  **Return Types & Structure**:
    * If the query asks for data retrieval from 'df' that results in a table or multiple rows/columns, 'result = ' should be a pandas DataFrame. Include relevant identifier columns along with the specifically requested data, unless the query is purely for aggregation.
    * If the query asks for a single specific calculated value from 'df' (e.g., a sum, average, count, minimum, maximum of a particular field), 'result' should be that scalar value.
    * For statistics or aggregations on 'df', CALCULATE them and assign to 'result = '.
    * For charts or visualizations of 'df', 'result = ' should be the Plotly figure object (see "DATA VISUALIZATION (PLOTLY)" section).
    * For queries about prompt content or conversational history, see the "HANDLING QUERIES ABOUT PROMPT CONTENT OR HISTORY" section.
4.  **No Imports, Prints, or Comments**: DO NOT include `import` statements, `print()` statements, or any comments within the generated code.
5.  **Case Sensitivity**: Assume string comparisons are case-sensitive unless the user's prompt implies otherwise (e.g., "ignore case").
6.  **Contextual Awareness for `df` Operations**: Refer to previous queries and their results on 'df' (provided within the `<HISTORY>` block) when the current user prompt is a follow-up or refers to those previous `df` results (e.g., "show me the top 5 of those").

### Core Guidelines
- ONLY respond with executable Python code using pandas.
- DO NOT include any explanation, markdown formatting, or comments.
- Return either a **pandas DataFrame** or a **single calculated value** as output.
- ASSUME 'df' is the pre-loaded DataFrame.

### Data Visualization Rules
1. Use **Plotly Express (`px`) or Plotly Graph Objects (`go`)** for all visualizations.
2. **No `fig.show()`**: DO NOT include `fig.show()` commands. Assign the figure object to `result` (e.g., `result = fig`).
3. Ensure all plots have proper:
   - Titles
   - Axis labels
   - Legends (where applicable)
   - Color schemes for clarity
4. Optimize for readability and interactivity.
5. For charts, return the figure as `result = fig`.

### Handling Extremum Values (Max/Min)
- Preserve DataFrame structure when identifying rows with max/min values.
- Use `.loc[[index]]` or `.iloc[[index]]` for row selection.
- For single-row outputs, ensure they remain as DataFrames by using `.to_frame().T` or double brackets `df[[row]]`.

### Date Operations
- Convert string-based date columns to datetime format before filtering or comparison:  
  `df['date_col'] = pd.to_datetime(df['date_col'])`
- Use the `.dt` accessor to extract year/month/day components.
- Include necessary date conversion logic in all date-related queries.

When working with dates in `df`:
1.  **Mandatory Datetime Conversion**: ALWAYS convert string date columns in `df` to datetime objects using `pd.to_datetime()` before any filtering or date-based operations. Use `format='%d-%b-%y'` and `errors='coerce'` for all date columns in this dataset (e.g., `account_open_date`, `last_debit_date`, `last_credit_date`, `date_of_birth`). Example: `df['account_open_date'] = pd.to_datetime(df['account_open_date'], format='%d-%b-%y', errors='coerce')`.
2.  **Date Components**: Use the `.dt` accessor to extract date components from datetime columns (e.g., `df['date_column'].dt.year`, `df['date_column'].dt.month`, `df['date_column'].dt.day_name()`).
3.  **Date Comparisons**: Ensure all date values are in datetime format before conducting comparisons. Use `pd.to_datetime('YYYY-MM-DD')` for fixed date strings in comparisons. Include proper date conversion code in all queries involving date comparisons against `df` columns.

### Dataset Schema Overview
{columns_info}

Column Descriptions:
{column_descriptions}

Sample Data:
{sample_data}

<HISTORY>
{conversation_history}
</HISTORY>

### Output Format Rules
- EXCEPT for charts or plots, ALWAYS return results as a DataFrame.
- If the query asks for a scalar value (e.g., sum, average, count), return it directly.
- For charts, return the figure as `result = fig`.
- If the query is for a chart or visualization (plot), CREATE it using plotly and assign to 'result ='. For example: result = fig.
- For statistics or aggregations, compute and assign the result.
- If unsure, generate a relevant filter or subset of the data.

### Context Awareness
- Previous queries and results are stored in <HISTORY> tags
- Use history to understand context and build upon previous analyses
- Maintain consistency with code patterns established in history
- Analyze history before generating new code to ensure relevance

### Final Instructions
- If the query relates to example prompts, return a list starting with `Example Prompts:` containing 5-10 meaningful prompts for analyzing the dataset (do not include any code).
- DO NOT include comments, import or print statements in the generated code.
- SINCE data in the dataset are related to columns availbale, INCLUDE all relevant columns as needed to answer the question clearly.

Your goal is to provide accurate, clean, and functional pandas code that answers the user's question effectively based on the dataset schema and history.
"""
    # Create the request payload
    payload = {
        "model": MODEL,  # or any other model you have
        "prompt": f"{prompt.strip()}",
        "system": system_prompt.strip(),
        "stream": False,
    }

    # if st.session_state.get("show_sys_prompt") == "No":
    # logger.info(f"System Prompt:\n {system_prompt}") # CHECK
    logger.info(f"HISTORY :\n{conversation_history}")

    try:
        # Send the request to LLM
        response = requests.post(BASE_URL, json=payload)
        logger.info("Waiting for response..")

        if response.status_code == 200:  # Extract the generated code from the response
            response_data = response.json()
            generated_code = response_data.get("response", "")

            # Clean up the generated code
            generated_code = generated_code.strip()
            # Remove any markdown code blocks if present
            generated_code = re.sub(r"```python\s*", "", generated_code)
            generated_code = re.sub(r"```\s*", "", generated_code)

            generated_code = clean_query(generated_code)

            return generated_code
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"


# Function to execute the pandas query
def execute_pandas_query(query_code, df):
    try:
        # Create a local scope with the dataframe
        result_type = None
        local_vars = {"df": df}
        global_vars = {"px": px, "go":go, "pd": pd, "np": np, "re": re} # go added

        # Execute the code in the local scope
        logger.info(f"Before exec >> {query_code[:15]}")
        # clean_query
        if query_code.startswith("Example Prompts:"):
            # If the query is for prompts, return the generated code as is
            result = query_code
            result_type = "text"

        if query_code.strip().startswith("result = ") and "Example Prompts:" in query_code: # both result and prompts are added
            logger.info("<ISSUE example prompt>")
            query_code = str(query_code.split("Example Prompts:")[0]).strip()
        # '>=' not supported between instances of 'str' and 'Timestamp'
            
        if query_code.strip().startswith("Example Prompts:") and "result = " in query_code: 
            logger.info("<ISSUE prompt example>")
            query_code = str(query_code.split("result =")[-1]).strip()

        if query_code.strip().startswith("result = ") and "fig = px." in query_code:
            logger.info("<ISSUE pandas and figures>")
            if "result = fig" not in query_code: 
                query_code +="\nresult = fig"

        if query_code.startswith("result = "):
            exec(query_code, global_vars, local_vars)

            # Get the result
            result = local_vars.get("result", None)

            # logger.info(f"Local vars Keys: {local_vars.keys()}")  # df, result:Figure,
            logger.info(f"Local vars Type: {type(local_vars['result'])}")
            logger.info(f"Local vars Result: {local_vars['result']}")

        # logger.info(f"After exec >> type:{type(result)}")

        # Check if result exists or is NaN/None
        if result is None:
            logger.info("Result is None")
            return None, NO_DATA_MESSAGE, None

        # For numeric results, check if NaN
        if isinstance(result, (float, int)) and (pd.isna(result) or result is None):
            logger.info("Result is NaN")
            return None, NO_DATA_MESSAGE, None

        # For dataframes, check if empty
        # if isinstance(result, pd.DataFrame) and result.empty:
            # logger.info("Result is empty DataFrame")
            # return None, NO_DATA_MESSAGE, None
        if isinstance(result, pd.DataFrame):
            logger.info("Result is dataframe")
            result_type = "dataframe"
        # elif isinstance(result, pd.Series):
            # result_type = "series" # Convert Series to DataFrame for display            result = result.to_frame()
        elif hasattr(result, "update_layout") and hasattr(result, "data"): # isinstance(result, px.Figure):
            logger.info("Result is plotly_figure1")
            result_type = "plotly_figure"
        elif isinstance(result, (float, int, np.integer, np.floating)):
            logger.info("Result is numeric")
            result_type = "numeric"
        elif isinstance(result, str): #  What is the most common customer industry among active accounts?
            logger.info("Result is text")
            result_type = "text"
        elif isinstance(result, pd.Series):
            logger.info("Result is Series")
            result_type = "series"
        elif isinstance(result, np.ndarray):
            logger.info("Result is nDarray")
            result_type = "ndarray"
        else:
            logger.info("Result is other")
            result_type = "other"

        return result, None, result_type
    except Exception as e:
        logger.error(str(e))
        logger.warning(f"<Exception> {str(e)}") 
        return None, ERROR_MESSAGE, None


# Function to generate summary statistics
def visualize_numeric_columns(df):
    """
    Creates interactive charts for numeric columns in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing data to visualize

    Returns:
    None: Displays charts directly in the Streamlit app
    """
    # Get all numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime"]).columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns found in the provided DataFrame.")
        return
    # if not categorical_columns:
    #     st.warning("No categorical columns found in the provided DataFrame.")
    #     return
    # if not datetime_columns:
    #     st.warning("No datetime columns found in the provided DataFrame.")
    #     return

    # Let user select a numeric column to visualize
    selected_column = st.selectbox(
        "**Select a column to visualize:**", options=numeric_columns
    )

    if selected_column:
        st.subheader(f"Distribution of {selected_column}")

        # Choose visualization type
        viz_type = st.selectbox(
            "Select visualization type:",
            ("Bar", "Scatter", "Histogram", "Box Plot", "Bar Chart (Top Values)"),
        )
        # viz_type = st.radio(
        #     "Select visualization type:",
        #     options=["Bar", "Scatter", "Histogram", "Box Plot", "Bar Chart (Top Values)"]
        # )

        # Filter out NaN values
        valid_data = df[~df[selected_column].isna()]
        bin_count = st.slider(
            "Number of bins (interval of grouped data):",
            min_value=0,
            max_value=20,
            value=5,
        )
        value_counts = valid_data[selected_column].value_counts().nlargest(bin_count)

        if viz_type == "Histogram":
            # Create histogram with adjustable bins

            fig = px.histogram(
                valid_data,
                x=selected_column,
                nbins=bin_count,
                color_discrete_sequence=["#3366CC"],
                title=f"Histogram of {selected_column}",
            )
            st.plotly_chart(fig)

            # Add basic statistics
            st.write(f"**Basic Statistics:** {selected_column}")
            stats = df[selected_column].describe()
            st.write(stats)

        elif viz_type == "Bar":
            # For categorical data, show the top N categories

            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                color=value_counts.values,
                color_continuous_scale="Sunset",
                labels={"x": selected_column, "y": "Count"},
                title=f"Bar Chart of {selected_column}",
            )
            st.plotly_chart(fig)

            # Add basic statistics
            st.info(f"**General Statistics:** {selected_column}")
            stats = df[selected_column].describe()
            st.dataframe(stats, use_container_width=True)
            # st.write(stats)

        elif viz_type == "Scatter":
            # For categorical data, show the top N categories

            fig = px.scatter(
                x=value_counts.index,
                y=value_counts.values,
                color=value_counts.values,
                color_continuous_scale="Viridis",
                labels={"x": selected_column, "y": "Count"},
                title=f"Scatter Plot of {selected_column}",
            )
            st.plotly_chart(fig)

            # Add basic statistics
            st.write(f"**Basic Statistics:** {selected_column}")
            stats = df[selected_column].describe()
            st.write(stats)

        elif viz_type == "Box Plot":
            fig = px.box(
                valid_data, y=selected_column, title=f"Box Plot of {selected_column}"
            )
            st.plotly_chart(fig)

            # Show outliers info
            q1 = np.percentile(valid_data[selected_column], 25)
            q3 = np.percentile(valid_data[selected_column], 75)
            iqr = q3 - q1
            outlier_cutoff_low = q1 - 1.5 * iqr
            outlier_cutoff_high = q3 + 1.5 * iqr

            outliers = valid_data[
                (valid_data[selected_column] < outlier_cutoff_low)
                | (valid_data[selected_column] > outlier_cutoff_high)
            ]

            if not outliers.empty:
                st.write(f"Number of outliers: {len(outliers)}")
                if st.checkbox("Show outliers"):
                    st.write(outliers)

        elif viz_type == "Bar Chart (Top Values)":
            # For bar charts, limit to top N values
            top_n = st.slider("Show top N values:", min_value=5, max_value=20, value=5)

            # Check if we need to group values
            if len(valid_data[selected_column].unique()) > 100:
                st.warning(
                    f"Column has {len(valid_data[selected_column].unique())} unique values. Using bins for visualization."
                )

                # Create bins for large number of unique values
                counts, bins = np.histogram(valid_data[selected_column], bins=top_n)
                bin_labels = [
                    f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins) - 1)
                ]
                binned_data = pd.DataFrame({"bin": bin_labels, "count": counts})

                fig = px.bar(
                    binned_data,
                    x="bin",
                    y="count",
                    title=f"**Distribution of** {selected_column} (Binned)",
                    color=binned_data.values,
                    color_continuous_scale="Blues",
                )
            else:
                # Get value counts for this column
                value_counts = (
                    valid_data[selected_column]
                    .value_counts()
                    .nlargest(top_n)
                    .reset_index()
                )
                value_counts.columns = ["Value", "Count"]

                fig = px.bar(
                    value_counts,
                    x="Value",
                    y="Count",
                    title=f"Top {top_n} values for {selected_column}",
                    color="Count",
                    color_continuous_scale="Blues",
                )

            st.plotly_chart(fig)

    # Option to explore correlations if multiple numeric columns exist
    if len(numeric_columns) > 1 and st.checkbox(
        "*Explore correlations between columns*"
    ):
        st.divider()
        st.subheader("Correlation Analysis")

        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("X-axis:", options=numeric_columns, key="x_column")
        with col2:
            y_column = st.selectbox(
                "Y-axis:",
                options=numeric_columns,
                index=1 if len(numeric_columns) > 1 else 0,
                key="y_column",
            )

        if x_column and y_column:
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=f"{x_column} vs {y_column}",
                opacity=0.6,
                trendline="ols" if st.checkbox("Show trend line") else None,
            )
            st.plotly_chart(fig)

            # Show correlation coefficient
            correlation = df[[x_column, y_column]].corr().iloc[0, 1]
            st.write(f"Correlation coefficient: {correlation:.4f}")


# Function to display basic info
def show_basic_info(df):
    import time

    time.sleep(0.2)
    columns_description = get_column_descriptions(df)
    # Display basic info about dataframe
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    st.divider()
    # Show a sample of the data
    with st.expander("**Data Preview**", expanded=True, icon=MATERIAL_ARROW_DOWN):
        st.dataframe(df.head(1), use_container_width=True)
    st.divider()
    # Display sample data in the main area
    with st.expander(
        "**View Sample Data, by Columns**", expanded=True, icon=MATERIAL_ARROW_DOWN
    ):
        # Show column selector dropdown
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=all_columns,
            default=all_columns[:5],  # Default to first 5 columns
        )
    st.divider()
    # Show sample data with selected columns
    sample_size = st.slider("Number of sample rows:", 3, 30, 5)
    st.dataframe(df[selected_columns].head(sample_size), use_container_width=True)
    st.divider()
    # Display column data types
    with st.expander("**Column Data Types**", expanded=True, icon=MATERIAL_ARROW_DOWN):
        st.markdown(
            "Column name, data types and sample values: these information will be valubale for Prompt writing."
        )
        col_types = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [str(df[col].dtype) for col in df.columns],
                "Sample Value": [
                    str(df[col].iloc[0]) if len(df) > 0 else "" for col in df.columns
                ],
                "Null Count": df.isna().sum().values,
                "Non-Null Count": df.count().values,
                "Unique Values": [df[col].nunique() for col in df.columns],
            }
        )
        st.dataframe(col_types, use_container_width=True)
    st.divider()

    # Display basic info
    with st.expander("**Column Description**", expanded=True, icon=MATERIAL_ARROW_DOWN):
        st.info(columns_description)
    st.divider()

    # Show distributions of numeric columns
    with st.expander("**Visualize Columns**", expanded=True, icon=MATERIAL_ARROW_DOWN):
        visualize_numeric_columns(df)

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.metric("Total Records", len(df))
    #     st.metric("Total Balance (NPR)", f"{df['account_balance'].sum():,.2f}")

    # with col2:
    #     active_accounts = len(df[df['account_inactive'] == False]) if 'account_inactive' in df.columns else "N/A"
    #     inactive_accounts = len(df[df['account_inactive'] == True]) if 'account_inactive' in df.columns else "N/A"

    #     st.metric("Active Accounts", active_accounts)
    #     st.metric("Inactive Accounts", inactive_accounts)

    # # Show distribution of account types if column exists
    # if 'account_type' in df.columns:
    #     st.subheader("Account Types")
    #     account_types = df['account_type'].value_counts().reset_index()
    #     account_types.columns = ['Account Type', 'Count']
    #     fig = px.bar(account_types, x='Account Type', y='Count', color='Count')
    #     st.plotly_chart(fig)


# Function to check if LLM is running
def llm_connection_status():
    try:
        logger.info("LLM Connection")
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            st.success("Connected to LocalLLM")
            return True
        else:
            st.error("LocalLLM is running but returned an error")
            return False
    except:
        st.error("? Cannot connect to LocalLLM")
        st.info("Start LocalLLM service to enable query functionality")
        return False


def format_indian_currency(amount):
    """Formats a float value as Indian currency (e.g., 1,65,514.68)."""
    try:
        amount_str = "{:.2f}".format(amount)  # Format to 2 decimal places
        integer_part, decimal_part = amount_str.split(".")

        # Process for Indian numbering system (lakhs, crores)
        s = integer_part[::-1]  # Reverse the string
        groups = [s[0:3]]  # First group of 3 digits

        i = 3
        while i < len(s):
            groups.append(s[i : i + 2] if i + 2 <= len(s) else s[i:])
            i += 2

        formatted_integer = ",".join(groups)[::-1]  # Reverse back and join with commas

        return f"{formatted_integer}.{decimal_part}"
    except:
        return str(amount)  # Return original amount if formatting fails


# Function to convert DataFrame to image
def df_to_image(df, filename="dataframe.png"):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("tight")
    ax.axis("off")
    sns.heatmap(df.isnull(), cbar=False, ax=ax)  # Example formatting
    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.savefig(filename, dpi=300, bbox_inches="tight")


def set_conversation_history(max_messages=10):
    """
    Format the last max_messages from session state into a structured conversation history
    for the system prompt.
    Role: assistant
    Type: ['plotly_figure','numeric','text','dataframe','series']
    Content: Found \d matching records : Found 294 matching records
    Prompt: Prompt from user
    Query: result = df[df['account_type'] == 'SAMMAN BACHAT KHATA']
    Data: 
    DataType:

    """
    history = []
    temp_history_messages = []
    historic_data = ''
    temp_content = ''

    logger.info("History")
   
    # Get the most recent messages up to max_messages
    temp_history_messages = [msg for msg in st.session_state.messages if msg.get('role') == 'assistant' and 'data' in msg][-max_messages:]
    # logger.info(f" <> {temp_history_messages}")
    if len(temp_history_messages)>0:
        for message in temp_history_messages:
            temp_content = message.get("content").strip()
            if re.match(r"Found\s+([1-9]+)\s+matching",temp_content):
                logger.info("OTHER resdatault")
                # Format the assistant response with data
                df_data = message.get('data','')
                
                if len(df_data) > 10:
                    df_data = df_data.sample(10)

                # Convert dataframe to markdown table
                if isinstance(df_data, pd.DataFrame):
                    historic_data = df_data.to_markdown(index=False)
                elif isinstance(df_data, pd.Series):
                    historic_data = df_data.to_markdown(index=False)
                else:
                    logger.info("OTHER resdatault>>")
                    historic_data = str(df_data)
                
                response = f"""
                Query: {message.get("prompt","").strip()}
                Code: {message.get('query', '').strip()}
                Output:
                {historic_data}
                """
                history.append(response)
            if re.match(r"The\s+result\s+is\:\s*(\d+)",temp_content):
                logger.info("OTHER result")
                response = f"""
                Query: {message.get("prompt","").strip()}
                Code: {message.get('query', "").strip()}
                Output: {message.get('data',"")}
                """
                history.append(response)
            if re.match(r"The\s+result\s+is\:\s*(\w+)",temp_content): # True, False, Head Office
                logger.info("OTHER HIstoRY")
                response = f"""
                Query: {message.get("prompt","").strip()}
                Code: {message.get('query', "").strip()}
                Output: {message.get('data',"")}
                """
                history.append(response)
            # if "Error" not in temp_content:
            #     logger.info("OTHER Example")
            #     if "Example Prompts:" not in temp_content:
            #         response = f"""
            #         Query: {message.get("prompt","").strip()}
            #         Code: {message.get('query', "").strip()}
            #         Output: {message.get('data',"")}
            #         """
            #         history.append(response)
            
    return "\n".join(history)


def set_session_state(assistant_message, error_message, content, prompt, pandas_query, result):
    """
    Role: assistant
    Type: ['plotly_figure','numeric','text','dataframe','series']
    Content: Found \d matching records
    Prompt: Prompt from user
    Query: result = df[df['account_type'] == 'SAMMAN BACHAT KHATA']
    Data: 
    DataType: 

    st.session_state.messages.append({"role": "user", "content": query})
    """
    logger.info(f"--- {type(result)}")
    st.session_state.messages.append(
        {
            "role": assistant_message,  # assistant
            "type": error_message,      # data
            "content": content,
            'prompt': prompt,
            "query": pandas_query,
            "data": result,
            "data_type": type(result)
        }
    )

# Main app
def main():
    st.title("Chat FinanceLLM - Account Transactions")
    logger.info("Chat FinanceLLM")
    # Upload file
    uploaded_file = st.sidebar.file_uploader(
        "**Upload your CSV file**",
        type="csv",
        help="Upload a CSV file with data to be processed",
    )

    df = pd.DataFrame()

    if uploaded_file:
        # Check file extension
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "csv":
            df = load_data(uploaded_file)
            if df.empty:
                st.error("Failed to load data. Please check the file path and format.")
                return
            logger.info(f"File uploaded {df.shape}")
            logger.info(f"SESSION: \n<....session.........>\n{st.session_state}\n</....session..........>")
            if len(st.session_state.get("column_descriptions")) == 0:
                logger.info("SESSION: Column Description")
                st.session_state.column_descriptions = get_column_descriptions(df)

    # Initialize session state for selection
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = "Query"

    # Sidebar navigation buttons
    if st.sidebar.button(
        "Data Overview",
        use_container_width=True,
        help="View the data overview",
        key="overview",
        icon=":material/dashboard:",
    ):
        st.session_state.selected_option = "Overview"

    if st.sidebar.button(
        "Chat with Data",
        use_container_width=True,
        help="Query the bank data",
        key="query",
        icon=":material/chat:",
    ):
        st.session_state.selected_option = "Query"

    if st.sidebar.button(
        "Analytic Dashboard",
        use_container_width=True,
        help="View the dashboard overview with analysis",
        key="dashboard",
        icon=":material/health_metrics:",
    ):
        st.session_state.selected_option = "Dashboard"

    # Dashboard and Query buttons
    # option_dashboard = st.sidebar.button("Dashboard Overview", use_container_width=True, help="View the dashboard overview")
    # option_query = st.sidebar.button("Query Data", use_container_width=True,help="Query the bank data")

    # Add LLM connection status indicator
    with st.sidebar:
        st.subheader("", divider=True)
        llm_connected = llm_connection_status()

    # Main content based on selected option DASHBOARD
    # if st.session_state.selected_option == "Dashboard":
    # st.header("Dashboard / Analysis")
    # st.caption("Overview of the data with detailed analysis")

    # visual.analyze_dataframe(df)  # Dashboard

    # if st.session_state.selected_option == "Overview":
    #  st.header("General Data Overview")
    #  st.caption("Overview of the data with data introspection")

    # show_basic_info(df) # Basic info

    # Query mode is the default view
    # if option_query or not option_dashboard:
    if st.session_state.selected_option == "Query":
        st.header("Query/Chat with Your Data")

        # Show data sample in sidebar
        with st.expander("**Sample Data**", expanded=False, icon=MATERIAL_ARROW_DOWN):
            st.info("""Displays the sample data from the choosen file.""")
            if df.columns.size > 0:
                st.dataframe(df.sample(6), use_container_width=True)
            # st.dataframe(df.head(1), use_container_width=True)

        with st.expander(
            "**Sample Prompts**", expanded=False, icon=MATERIAL_ARROW_DOWN
        ):
            st.info(
                """Enter questions in natural language to analyze the data uploaded.."""
            )
            st.code(
                """            
            Example queries:
            - provide me some prompts to analyze current dataset using sector
            - provide me some prompts to analyze current dataset
            - Show top 10 customers with highest balance
            - Describe data or give some information about data
            - Data Sample
            - Calculate the total account balance for each economic sector and find the average, minimum, and maximum balance
            - provide me some prompts to analyze current dataset using sector
            - Describe data
            - Plot the distribution of account balances across different sectors
            - list me all inactive account with balance > 50000 and account opened before 2015
            - list some data with balance > 50000 and internet banking disabled
            - Plot bar chart with average balance from industry, sector and branch
            - Show inactive accounts with a balance greater than 100000
            - Generate a chart showing the distribution of account types across different economic sectors
            - list me all inactive account with balance > 50000
            - Create a bar chart showing the number of active accounts per bank branch
            - Show me max, min, average account balance for year 2015 by industry
            - how many sector are there and which is the most common
            - Show customers from branch Damauli
            - Plot active vs inactive accounts
            - Create a bar chart showing the distribution of account types by customer industry.
            - What is the average balance of the accounts?
            - What is the average balance for branch Damauli?
            - Show average amount for sector 'LOCAL - PERSONS' in Head Office
            - list 5 account holder from damauli with amount > than 500000
            - What is the ratio of active to inactive account
            - how many account categories are there provide me with their average amount
            - Show the distribution of account types
            """
            )

        # Display chat messages: For chat session
        # CHAT Window
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Display text messages
                st.markdown(message["content"]) 
                
                # Prompt and Result line (user|assistant: content)
                # Display data frames, figures, etc.
                if "data" in message:
                    if message["type"] == "dataframe" or message["type"] == "series":
                        # logger.info("DF SER")
                        st.dataframe(message["data"])
                    if message["type"] == "plotly_figure":
                        # logger.info("FIG")
                        st.plotly_chart(message["data"])

                    # if message["type"] == "numeric":
                    #     st.info(f"Result: {message['data']}")
                    # if message["type"] == "text":
                    #     st.info(f"Result: {message['data']}")
                    # if message["type"] == "other":
                    #     st.info(f"Result: {message['data']}")

        # Input for the query
        if query := st.chat_input(
            "Talk to your data...LLM will generate the output based on your prompts. Ex:'prompts to analyze data'. Try to be specific as required."
        ):
            # Check if LLM is connected
            if not llm_connected:
                st.error("Cannot process query because LocalLLM is not connected.")
                return

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})

            # Display user message
            with st.chat_message("user"):
                st.markdown(query)

            prompt=''
            with st.chat_message("assistant"):
                with st.spinner("Processing your query..."):
                    prompt = query.strip()
                    # Get the pandas query from LLM
                    pandas_query = get_pandas_query(prompt, df, st.session_state.column_descriptions)
                    if st.session_state.get("show_sys_prompt") == "No":
                        st.session_state.get("show_sys_prompt") == "Yes" # TEMP log

                    logger.info(
                        f"Generated Pandas Query:\n{pandas_query}\n------------"
                    )
                    # # Display the generated code
                    # with st.expander("View Generated Code", icon=MATERIAL_ARROW_DOWN):
                    #     st.code(pandas_query, language="python")

                    # Execute the query
                    # Result Type: DataFrame, Plotly Figure, Numeric, Other, Text

                    result, error, result_type = execute_pandas_query(pandas_query, df)
                    if result_type == "plotly_figure":
                        logger.info(
                            f">> main FIGURE Type: {result_type}|{type(result)} - {error} - Results: {result['data'][0]['name']} - {result['data'][0]['type']}"
                        )
                    else:
                        logger.info(
                            f">> main RESULTS Type: {result_type}|{type(result)} - {error} - Results: {result}"
                        )

                    # Do not display "View Generated Code"
                    # Display the generated code
                    with st.expander("View Generated Code", icon=MATERIAL_ARROW_DOWN):
                        st.code(pandas_query, language="python")  

                    if error:
                        logger.error(f"MAIN Error: {error}")

                        expected_texts = [
                            "Which",
                            "What",
                            "How",
                            "List",
                            "Visualize",
                            "Create",
                            "Calculate",
                            "Identify",
                            "Generate",
                            "Display",
                            "Find",
                            "Show",
                        ]
                        if not any(elem in pandas_query for elem in expected_texts):
                            logger.error(f"1.After execution error: {error}")
                            st.error(error)
                            # set_session_state(
                            #     "assistant",
                            #     "error",
                            #     NO_MATCHING_RECORDS,
                            #     pandas_query,
                            #     result,
                            # )

                        else:
                            logger.error(f"2.After execution error: {error}")
                            st.error(error)
                            # set_session_state(
                            #     "assistant",
                            #     "error",
                            #     f"Some issue has occurred, rewrite your prompt. Error: {error}",
                            #     pandas_query,
                            #     result,
                            # )
                    else:
                        logger.info("<MAIN>")

                        if result is None:
                            st.info(
                                "No matching records found for your query. Please try again with a different query."
                            )
                            response_content = NO_MATCHING_RECORDS

                        elif (result_type == "dataframe"):  # Result_type: DataFrame, Plotly Figure, Numeric, Other, Text
                            logger.info("<DATAFRAME>")
                            total_results = len(result)
                            response_content = f"Found {total_results} matching records"
                            st.success(response_content)
                            logger.info("<DF>...")
                            st.dataframe(result, use_container_width=True, hide_index=False)

                            csv = result.to_csv(index=False)
                            st.download_button(
                                label="Download results as CSV",
                                data=csv,
                                file_name="query_results.csv",  # TODO: filename with prompt
                                mime="text/csv",
                            )
                            # Store the result for message history 'role': 'assistant'
                            if total_results > 0:
                                set_session_state(
                                    "assistant",
                                    "dataframe",
                                    response_content,
                                    prompt,
                                    pandas_query,
                                    result, # TEMP
                                )
                            else:
                                set_session_state(
                                    "assistant",
                                    "dataframe",
                                    response_content,
                                    prompt,
                                    pandas_query,
                                    result, # TEMP
                                )
                        elif (result_type == "series"):
                            logger.info("<SERIES>")
                            # series to df
                            temp_df = result.to_frame().T.reset_index() # .T.reset_index()
                            total_results = len(temp_df)
                            response_content = f"Found {total_results} matching records"
                            st.success(response_content)
                            st.dataframe(temp_df, use_container_width=True, hide_index=True)
                            if total_results > 0:
                                set_session_state(
                                    "assistant",
                                    "series",
                                    response_content,
                                    prompt,
                                    pandas_query,
                                    temp_df, # TEMP
                                )
                        elif (result_type == "ndarray"):
                            logger.info("<NDARRAY>")
                            temp_df = pd.Series(result)
                            total_results = len(temp_df)
                            response_content = f"Found {total_results} matching records"
                            st.success(response_content)
                            st.dataframe(temp_df, use_container_width=True, hide_index=True)
                            if total_results > 0:
                                set_session_state(
                                    "assistant",
                                    "ndarray",
                                    response_content,
                                    prompt,
                                    pandas_query,
                                    temp_df, # TEMP
                                )
                        elif result_type == "plotly_figure":
                            logger.info("<FIGURE>")
                            st.plotly_chart(result)
                            response_content = "Here's the visualization that you have requested"
                            set_session_state(
                                "assistant",
                                "plotly_figure",
                                response_content,
                                prompt,
                                pandas_query,
                                result,
                            )

                        elif result_type == "numeric":
                            logger.info("<NUMERIC>")
                            if isinstance(result, float):
                                formatted_result = format_indian_currency(result)
                                st.info(f"Result: {formatted_result}")
                                response_content = f"The result is: {formatted_result}"
                            else:
                                st.info(f"Result: {result}")
                                response_content = f"The result is: {result}"

                            set_session_state(
                                "assistant",
                                "numeric",
                                response_content,
                                prompt,
                                pandas_query,
                                result,
                            )
                        elif result_type == "other":
                            response_content = ''
                            error_message = ''
                            logger.info("<OTHER>")
                            if result == "True" or result == "False":
                                logger.info("<other bool>")
                                st.info(result)
                                set_session_state(
                                "assistant",
                                "bool",
                                f"The result is: {result}",
                                prompt,
                                pandas_query,
                                result,
                            )

                            if isinstance(result, (pd.DataFrame, pd.Series)) and len(result) > 0:
                                if "result = df" in pandas_query:
                                    error_message = 'dataframe'
                                    st.dataframe(result)
                                    response_content = f"Found {len(result)} matching records"
                                else:
                                    error_message = 'Other_dataframe'
                                    st.info(result)
                                    response_content = result
                            else:
                                error_message = 'Other_else_dataframe'
                                logger.info("<OTHER-ELSE>")
                                st.info(
                                    "No matching records found for your query. Please try again with a different query."
                                )
                                response_content = NO_MATCHING_RECORDS

                            # Store the result for message history
                            # set_session_state(
                            #     "assistant",
                            #     error_message,
                            #     response_content,
                            #     prompt,
                            #     pandas_query,
                            #     result,
                            # )
                        elif result_type == "text":
                            logger.info("<TEXT>")
                            # If the result is a text prompt, display it                                
                            if result.strip().startswith("Example Prompts:"):
                                st.info(result)
                                set_session_state(
                                    "assistant",
                                    "text",
                                    str(result),
                                    prompt,
                                    pandas_query,
                                    result,
                                )
                            else: # Which branch has the most number of active accounts?
                                st.info(result)
                                set_session_state(
                                    "assistant",
                                    "text",
                                    f"The result is: {str(result)}",
                                    prompt,
                                    pandas_query,
                                    result,
                                )
                        else:
                            logger.info(f"<ELSE>->{result}")
                            st.info(result)
                            # set_session_state(
                            #     "assistant",
                            #     "text",
                            #     str(result),
                            #     prompt,
                            #     pandas_query,
                            #     result,
                            # )


if __name__ == "__main__":
    main()


    # History:
    # st.session_state.messages.append(
    #     {
    #         "role": assistant_message,
    #         "type": error_message,
    #         "content": content,
    #         'prompt': prompt,
    #         "query": pandas_query,
    #         "data": result,
    #         "data_type": type(result)
    #     }
    # )
    # if "role" == "assistant":
    #     if "type" == "dataframe": # Found 1 matching records  
    #     if "type" == "numeric": # The result is: \d
    #     if "type" == "series":  # Found 1 matching records
            # query
            # prompt
            # data
    # session_state.messages and History to be make different
    
    




    # st.rerun()
    # st.session_state.messages = []
    # st.session_state.selected_option = ""
    # st.session_state.column_descriptions = []
    # st.cache.clear()
    