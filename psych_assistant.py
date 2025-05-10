from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def process_task(state: Dict[str, Any]) -> Dict[str, Any]:
    task = state.get('task')
    
    if task == '1':  # Content generation
        topic = state.get("topic", "")
        content_type = state.get("content_type", "LinkedIn post")
        messages = [
            SystemMessage(content=f"You are a professional content writer for psychologists. Create {content_type} content that is engaging, credible, and tailored for mental health professionals. Be specific and use a warm, expert tone."),
            HumanMessage(content=f"Topic: {topic}")
        ]
        model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        response = model.invoke(messages)
        state['result'] = response.content
        
    elif task == '2':  # Email drafting
        email_type = state.get("email_type", "intake")
        details = state.get("details", "")
        messages = [
            SystemMessage(content=f"You are an expert psychologist writing a {email_type} email to a client. Be clear, compassionate, and professional. Use a warm, supportive tone."),
            HumanMessage(content=f"Details: {details}")
        ]
        model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
        response = model.invoke(messages)
        state['result'] = response.content
        
    elif task == '3':  # Research summary
        topic = state.get("topic", "")
        messages = [
            SystemMessage(content="You are a research assistant for a psychologist. Summarize the latest research and best practices on the given topic. Be concise, evidence-based, and cite reputable sources if possible."),
            HumanMessage(content=f"Topic: {topic}")
        ]
        model = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        response = model.invoke(messages)
        state['result'] = response.content
    
    return state

def process_request(state: Dict[str, Any]) -> Dict[str, Any]:
    """Simple wrapper function to process requests"""
    return process_task(state)

if __name__ == "__main__":
    # Test the function
    test_state = {
        "messages": [],
        "task": "1",
        "topic": "Mindfulness in therapy",
        "content_type": "LinkedIn post",
        "email_type": None,
        "details": None
    }
    result = process_request(test_state)
    print(result["result"]) 