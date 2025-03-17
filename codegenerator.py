import streamlit as st
import requests
import json
from datetime import datetime
import os


# Configure page settings
st.set_page_config(
    page_title="LocalLLM Chat Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M_%S")

# Function to save conversation history
def save_conversation(conversation_id, messages):
    os.makedirs("conversations", exist_ok=True)
    with open(f"conversations/{conversation_id}.json", "w") as f:
        json.dump(messages, f)

# Function to load conversation history
def load_conversation(conversation_id):
    try:
        with open(f"conversations/{conversation_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Function to generate code using LocalLLM
def generate_code_with_local_llm(prompt, model="qwen2.5-coder:7b"):
    api_url = "http://localhost:11434/api/generate"
    
    # Get full conversation context
    conversation_context = ""
    for msg in st.session_state.messages:
        role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
        conversation_context += f"{role_prefix}{msg['content']}\n\n"
    
    # Add current prompt
    conversation_context += f"User: {prompt}\n\nAssistant: "
    
    payload = {
        "model": model,
        "prompt": conversation_context,
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }

    # """
    # payload = {
    #     "model": model,
    #     "prompt": conversation_context,
    #     "stream": False,
    #     "options": {
    #         "temperature": 0.2,
    #         "top_p": 0.95,
    #         "top_k": 40
    #     }
    # }
    # """
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "Sorry, I couldn't generate a response.")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to LocalLLM: {str(e)}\n\nMake sure LocalLLM is running !"

# Sidebar for settings and new chat
with st.sidebar:
    st.title("Chat Assistant-💻")
    # Model selection
    model_options = ["codellama:7b", "codellama:13b-code", "qwen2.5-coder:7b", "codellama:latest","deepseek-r1:7b-qwen-distill-q8_0","gemma3:latest"]
    st.subheader("", divider=True)
    selected_model = st.selectbox("Select Model", model_options)
    
    # Temperature slider for creativity
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                           help="Higher values make output more creative but less precise")
    
    # prompt_words = prompt.split()
    # conversation_prompt_len = len(prompt_words)
    # if conversation_prompt_len >= 3:
    #     conversation_prompt = "_".join(prompt_words[-3:])
    # else:
    #     conversation_prompt = prompt
    if st.button("New Chat", use_container_width=True, help="Start a new chat session", icon=":material/chat:"):
        st.session_state.messages = []
        st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M_%S")
        st.rerun()
    
    # Display saved conversations
    st.subheader("Saved Conversations", divider=True)
    if not os.path.exists("conversations"):
        os.makedirs("conversations")
    saved_conversations = [f.replace(".json", "") for f in os.listdir("conversations") if f.endswith(".json")]
    
    if saved_conversations:
        selected_conversation = st.selectbox("Conversation History", saved_conversations)
        if st.button("Load Conversation", type='secondary', use_container_width=True, help="Load a saved conversation", icon=":material/folder_open:"):
            st.session_state.messages = load_conversation(selected_conversation)
            st.session_state.current_conversation_id = selected_conversation
            st.rerun()
    
    # Add LocalLLM connection status indicator
    st.subheader("", divider=True)
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            st.success("✅ Connected to LocalLLM")
            
            # List available models
            # models = response.json().get("models", [])
            # if models:
            #     st.write("Available models:")
            #     for model in models:
            #         st.write(f"- {model['name']}")
            # else:
            #     st.info("No models found.")
            
        else:
            st.error("❌ LocalLLM is running but returned an error")
    except:
        st.error("❌ Cannot connect to LocalLLM")
        st.info("Start LocalLLM")

# Main chat interface
st.title("LocalLLM Chat Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for user message
if prompt := st.chat_input("What code would you like me to generate?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})  #Prompt is the user input
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_placeholder = st.empty()
            response = generate_code_with_local_llm(prompt, model=selected_model)
            
            # Format code blocks properly
            formatted_response = ""
            in_code_block = False
            lines = response.split("\n")
            
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    formatted_response += line + "\n"
                else:
                    formatted_response += line + "\n"
            
            response_placeholder.markdown(formatted_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
    
    # # Save conversation
    # prompt_words = prompt.split()
    # conversation_prompt_len = len(prompt_words)
    # if conversation_prompt_len >= 4:
    #     current_conversation_id = "_".join(prompt_words[-4:])
    # else:
    #     current_conversation_id = "_".join(prompt_words)

    # Rename Conversation Id    
    # st.session_state.current_conversation_id = current_conversation_id
    save_conversation(st.session_state.current_conversation_id, st.session_state.messages)

# Footer with instructions
st.caption("""
---
### How to use this LocalLLM Chat Assistant:
1. Make sure LocalLLM is installed and connected.
2. Ask for code generation, chat completion or necessary details by describing what you need
3. The assistant will generate code, provide assistanship based on your prompt]
""")