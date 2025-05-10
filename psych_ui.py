import streamlit as st
from psych_assistant import process_request
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'app' not in st.session_state:
    st.session_state.app = process_request

st.title("Psychologist Assistant")
st.write("Generate content, draft emails, and get research summaries")

# Task selection
task = st.radio(
    "Select a task:",
    ["1", "2", "3"],
    format_func=lambda x: {
        "1": "Content Generation",
        "2": "Email Drafting",
        "3": "Research Summary"
    }[x]
)

# Initialize state
state = {
    "messages": [],
    "task": task,
    "topic": None,
    "content_type": None,
    "email_type": None,
    "details": None
}

# Task-specific inputs
if task == "1":
    state["topic"] = st.text_input("Enter the topic for content generation:")
    state["content_type"] = st.selectbox(
        "Select content type:",
        ["LinkedIn post", "Blog post", "Social media post"]
    )
elif task == "2":
    state["email_type"] = st.selectbox(
        "Select email type:",
        ["intake", "follow-up", "referral", "termination"]
    )
    state["details"] = st.text_area("Enter email details:")
elif task == "3":
    state["topic"] = st.text_input("Enter the research topic:")

# Generate button
if st.button("Generate"):
    if (task == "1" and state["topic"]) or \
       (task == "2" and state["details"]) or \
       (task == "3" and state["topic"]):
        
        with st.spinner("Generating..."):
            result = st.session_state.app(state)
            st.write("### Result:")
            st.write(result["result"])
    else:
        st.error("Please fill in all required fields")

# Add a footer
st.markdown("---")
st.markdown("Built with Streamlit and LangChain")