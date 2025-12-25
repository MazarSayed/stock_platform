"""Streamlit application for testing LangGraph-based stock trading platform agent via FastAPI."""

import streamlit as st
import requests
import uuid


API_BASE_URL = "http://localhost:8000" 

# Page config
st.set_page_config(
    page_title="Stock Trading Platform Assistant",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "api_url" not in st.session_state:
    st.session_state.api_url = API_BASE_URL

st.title("📈 Stock Trading Platform Assistant")
st.markdown("Ask questions about the platform, execute trades, or get market insights!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # API URL configuration
    api_url = st.text_input("API URL", value=st.session_state.api_url, help="FastAPI server URL")
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
        st.rerun()
    
    st.markdown("---")
    st.header("Agent Information")
    st.markdown("""
    **Available Agents:**
    - **FAQ Agent**: Platform questions & support
    - **Task Agent**: Execute trades & operations  
    - **Market Insights Agent**: Market analysis & research
    """)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    # Test API connection
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{st.session_state.api_url}/", timeout=5)
            if response.status_code == 200:
                st.success("✅ API connected successfully!")
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent" in message:
            st.caption(f"Handled by: {message['agent']}")

# Chat input
if prompt := st.chat_input("Ask a question or request an action..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Invoke API
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                # Make API request
                response = requests.post(
                    f"{st.session_state.api_url}/chat",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("response", "No response generated.")
                    agent_used = data.get("agent", "supervisor")
                    
                    # Display response
                    st.markdown(response_text)
                    if agent_used:
                        st.caption(f"Handled by: {agent_used}")
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "agent": agent_used
                    })
                    
                    # Update session ID if returned
                    if "session_id" in data:
                        st.session_state.session_id = data["session_id"]
                else:
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
            except requests.exceptions.ConnectionError:
                error_msg = f"❌ Cannot connect to API at {st.session_state.api_url}. Make sure the FastAPI server is running."
                st.error(error_msg)
                st.info("💡 Start the FastAPI server with: `python main.py` or `uvicorn main:app --reload`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
