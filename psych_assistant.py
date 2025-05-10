from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Define a simple state type
StateType = Dict[str, Any]

def create_messages(task_type: str, content: str, **kwargs) -> List[Any]:
    """Create messages based on task type"""
    if task_type == "content":
        return [
            SystemMessage(content=f"You are a professional content writer for psychologists. Create {kwargs.get('content_type', 'LinkedIn post')} content that is engaging, credible, and tailored for mental health professionals. Be specific and use a warm, expert tone."),
            HumanMessage(content=f"Topic: {content}")
        ]
    elif task_type == "email":
        return [
            SystemMessage(content=f"You are an expert psychologist writing a {kwargs.get('email_type', 'intake')} email to a client. Be clear, compassionate, and professional. Use a warm, supportive tone."),
            HumanMessage(content=f"Details: {content}")
        ]
    else:  # research
        return [
            SystemMessage(content="You are a research assistant for a psychologist. Summarize the latest research and best practices on the given topic. Be concise, evidence-based, and cite reputable sources if possible."),
            HumanMessage(content=f"Topic: {content}")
        ]

def process_task(state: StateType) -> StateType:
    """Process the task based on state"""
    task = state.get('task')
    
    if task == '1':  # Content generation
        messages = create_messages(
            "content",
            state.get("topic", ""),
            content_type=state.get("content_type", "LinkedIn post")
        )
        model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        response = model.invoke(messages)
        state['result'] = response.content
        
    elif task == '2':  # Email drafting
        messages = create_messages(
            "email",
            state.get("details", ""),
            email_type=state.get("email_type", "intake")
        )
        model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
        response = model.invoke(messages)
        state['result'] = response.content
        
    elif task == '3':  # Research summary
        messages = create_messages("research", state.get("topic", ""))
        model = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        response = model.invoke(messages)
        state['result'] = response.content
    
    return state

def process_request(state: StateType) -> StateType:
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