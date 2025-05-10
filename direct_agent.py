from typing import Dict, TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define state
class ChatState(TypedDict):
    messages: List
    
# Define node
def respond(state: Dict) -> Dict:
    """Generate a response from the chatbot."""
    # Get or initialize messages
    messages = state.get("messages", [])
    
    # The specific user message to respond to
    user_message = state.get("user_message", "")
    
    # Clear out existing messages and start fresh
    messages = []
    
    # Add system message
    system_message = SystemMessage(content="""
    You are a helpful, intelligent AI assistant specialized in creating professional content.
    When asked to build content, create something specific, detailed, and tailored to the 
    exact request. If asked about psychologist content, focus on mental health expertise, 
    professional credentials, and compassionate care.
    """)
    messages.append(system_message)
    
    # Add user message
    if user_message:
        messages.append(HumanMessage(content=user_message))
    
    # Set up the model
    model = ChatOpenAI(temperature=0.7)
    
    # Generate response
    response = model.invoke(messages)
    
    # Add the response to messages
    messages.append(response)
    
    # Return updated state
    return {"messages": messages}

# Build the graph
def create_chat_graph():
    """Create a simple graph with one node that responds to the user."""
    workflow = StateGraph(ChatState)
    
    # Add node
    workflow.add_node("respond", respond)
    
    # Set entry point
    workflow.set_entry_point("respond")
    
    # Add edge to END
    workflow.add_edge("respond", END)
    
    # Compile
    return workflow.compile()

def main():
    # Create the graph
    app = create_chat_graph()
    
    # Get user input
    user_query = input("Enter your request: ")
    
    # Initial state
    state = {"messages": [], "user_message": user_query}
    
    # Run the graph
    result = app.invoke(state)
    
    # Print result
    ai_responses = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    if ai_responses:
        print(f"AI: {ai_responses[-1].content}")
    else:
        print("No response generated.")

if __name__ == "__main__":
    main() 