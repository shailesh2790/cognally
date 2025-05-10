from typing import Dict, TypedDict, List, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define state
class ChatState(TypedDict):
    messages: List
    
# Define nodes
def respond(state: ChatState) -> ChatState:
    """Generate a response from the chatbot."""
    messages = state["messages"]
    model = ChatOpenAI(temperature=0)
    response = model.invoke(messages)
    messages.append(response)
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
    
    # User query
    user_query = input("Enter your question: ")
    
    # Initial state
    state = {"messages": [HumanMessage(content=user_query)]}
    
    # Run the graph
    result = app.invoke(state)
    
    # Print result
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"AI: {message.content}")

if __name__ == "__main__":
    main() 