import streamlit as st
import pandas as pd
from data_processor import DataProcessor
from visualization import Visualizer
from data_embedder import DataEmbedder
from conversation_handler import ConversationHandler
from report_generator import ReportGenerator
from utils.logger import Logger
from models.model_manager import ModelManager

from ai_analyzer import AIAnalyzer 


def initialize_session_state():
    """Initialize all session state variables."""
    try:
        # Initialize all required session state variables
        st.session_state.initialized = True
        st.session_state.data_processors = {}
        st.session_state.data_embedder = DataEmbedder()
        st.session_state.model_manager = ModelManager()
        st.session_state.conversation_handler = ConversationHandler(
            st.session_state.data_embedder, st.session_state.model_manager
        )
        st.session_state.report_generator = ReportGenerator()
        st.session_state.logger = Logger("streamlit_app")
        st.session_state.messages = []
        st.session_state.uploaded_files = set()
        st.session_state.current_file = None  # Track the most recently uploaded file
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        st.session_state.initialized = False


def process_uploaded_file(uploaded_file):
    """Process a single uploaded file."""
    try:
        # Create a new data processor for this file
        data_processor = DataProcessor()
        success, message = data_processor.load_data(uploaded_file)

        if success:
            print(f"====> {data_processor.df.head()}")
            """
            # Store the processor with the filename as key
            st.session_state.data_processors[uploaded_file.name] = data_processor

            # Always embed the data for new files
            column_info = data_processor.get_column_info()
            print(f"====> {column_info}")

            embedding_success = st.session_state.data_embedder.embed_data(
                data_processor.df, column_info, source_name=uploaded_file.name
            )

            if embedding_success:
                st.session_state.uploaded_files.add(uploaded_file.name)
                st.session_state.current_file = (
                    uploaded_file.name
                )  # Update current file
                st.success(f"Successfully processed and embedded {uploaded_file.name}")
            else:
                st.error(f"Failed to embed data from {uploaded_file.name}")"
            """
            return data_processor.df,success
        else:
            st.error(message)

        return success
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return False


def main():
    st.set_page_config(page_title="Finance-AI-LLM", layout="wide")
    st.title("Finance-AI-LLM")

    # Initialize session state first
    initialize_session_state()

    if not st.session_state.initialized:
        st.error(
            "Failed to initialize application. Please refresh the page or contact support."
        )
        return

    # Model Selection and Database Reset in sidebar
    with st.sidebar:
        st.header("Settings")

        # Model Settings
        st.subheader("Model Settings")
        if st.session_state.model_manager:
            available_models = st.session_state.model_manager.get_available_models()
            current_model = st.session_state.conversation_handler.get_current_model()

            selected_model = st.selectbox(
                "Select AI Model",
                available_models,
                index=(
                    available_models.index(current_model)
                    if current_model in available_models
                    else 0
                ),
                key="model_selector",
                help="Choose between OpenAI GPT-4 (default) or Local Finance Model",
            )

            # Update model if changed
            if selected_model != current_model:
                success, message = st.session_state.conversation_handler.set_model(
                    selected_model
                )
                if success:
                    st.success(f"Switched to {selected_model}")
                else:
                    st.error(message)

        # Database Reset
        st.subheader("Database Management")
        if st.button(
            "Reset Database",
            type="primary",
            help="Clear all embedded data and start fresh",
        ):
            try:
                st.session_state.data_embedder.clear_data()
                st.session_state.uploaded_files.clear()
                st.session_state.data_processors.clear()
                st.session_state.messages.clear()
                st.session_state.current_file = None
                st.success("Database successfully reset!")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting database: {str(e)}")

    # File upload section
    st.header("Data Upload")
    uploaded_files = st.file_uploader(
        "Upload your CSV files", type=["csv"], accept_multiple_files=True,help="Currently CSV only Supported. TODO"
    )

    if uploaded_files:
        df = None
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.data_processors:
                df, success = process_uploaded_file(uploaded_file)

        # Display currently available datasets
        st.header("Available Datasets")
        st.info(
            f"Currently loaded datasets: {', '.join(st.session_state.uploaded_files)}"
        )
        if st.session_state.current_file:
            st.success(
                f"Currently viewing analysis for: {st.session_state.current_file}"
            )
        
        # Chat interface
        st.markdown("---")
        st.header("Chat with Data")

        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask about your data"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your question..."):
                        try:
                            ai_analyzer = AIAnalyzer()
                            code_response = ai_analyzer.generate_query(prompt,'df')
                            response = eval(code_response, {"df": df, "pd": pd})
                            # response = (
                            #     st.session_state.conversation_handler.process_query(
                            #         prompt
                            #     )
                            # )
                            print(f"====>{response}")
                            st.markdown(response)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response}
                            )
                        except Exception as e:
                            error_msg = "Sorry, I couldn't process your question. Please try again."
                            st.error(error_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_msg}
                            )


def display_metrics(metrics):
    num_metrics = len(metrics)
    num_rows = (num_metrics + 1) // 2

    for row in range(num_rows):
        col1, col2 = st.columns(2)

        idx = row * 2
        if idx < num_metrics:
            metric_name = list(metrics.keys())[idx]
            value = metrics[metric_name]
            with col1:
                st.metric(
                    label=metric_name.replace("_", " ").title(),
                    value=(
                        f"${value:,.2f}"
                        if "amount" in metric_name.lower()
                        or "budget" in metric_name.lower()
                        else f"{value:,.2f}"
                    ),
                    help=f"Full value: {value:,.2f}",
                )

        idx = row * 2 + 1
        if idx < num_metrics:
            metric_name = list(metrics.keys())[idx]
            value = metrics[metric_name]
            with col2:
                st.metric(
                    label=metric_name.replace("_", " ").title(),
                    value=(
                        f"${value:,.2f}"
                        if "amount" in metric_name.lower()
                        or "budget" in metric_name.lower()
                        else f"{value:,.2f}"
                    ),
                    help=f"Full value: {value:,.2f}",
                )

# Few-Shot Code

if __name__ == "__main__":
    main()
