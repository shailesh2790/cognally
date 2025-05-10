import os
from typing import List, Dict, TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define state
class AgentState(TypedDict):
    messages: List
    next: Optional[str]

# Define nodes
def user_node(state: AgentState, inputs: Dict) -> AgentState:
    """Process user input."""
    messages = state.get("messages", [])
    question = inputs.get("question", "")
    messages.append(HumanMessage(content=question))
    
    return {
        "messages": messages,
        "next": "assistant"
    }

def assistant_node(state: AgentState) -> AgentState:
    """Generate assistant response."""
    messages = state.get("messages", [])
    
    # Set up model
    model = ChatOpenAI(temperature=0)
    
    # Generate response
    response = model.invoke(messages)
    
    # Add response to messages
    messages.append(response)
    
    return {
        "messages": messages,
        "next": None
    }

# Create router
def router(state: AgentState) -> str:
    """Route to the next node or end."""
    if state.get("next") == "assistant":
        return "assistant"
    else:
        return END

# Build graph
def build_graph():
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("user", user_node)
    graph.add_node("assistant", assistant_node)
    
    # Add edges
    graph.add_conditional_edges("user", router, {
        "assistant": "assistant",
        END: END
    })
    
    graph.add_conditional_edges("assistant", router, {
        "assistant": "assistant",
        END: END
    })
    
    # Set entry point
    graph.set_entry_point("user")
    
    # Compile graph
    return graph.compile()

def main():
    # Build graph
    agent = build_graph()
    
    # Initialize state
    state = {
        "messages": [],
        "next": None
    }
    
    # Run agent
    question = input("Enter your question: ")
    result = agent.invoke({
        "question": question,
        **state
    })
    
    # Print result
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"AI: {message.content}")

if __name__ == "__main__":
    main() 