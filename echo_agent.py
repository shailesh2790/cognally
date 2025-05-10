from typing import Dict, TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# Define state
class EchoState(TypedDict):
    input: str
    output: Optional[str]
    
# Define nodes
def echo(state: EchoState) -> EchoState:
    """Echo the input."""
    return {"input": state["input"], "output": f"Echo: {state['input']}"}

# Build the graph
def create_echo_graph():
    """Create a simple graph with one node that echoes the input."""
    workflow = StateGraph(EchoState)
    
    # Add node
    workflow.add_node("echo", echo)
    
    # Set entry point
    workflow.set_entry_point("echo")
    
    # Add edge to END
    workflow.add_edge("echo", END)
    
    # Compile
    return workflow.compile()

def main():
    # Create the graph
    app = create_echo_graph()
    
    # User input
    user_input = input("Enter something to echo: ")
    
    # Initial state
    state = {"input": user_input, "output": None}
    
    # Run the graph
    result = app.invoke(state)
    
    # Print result
    print(result["output"])

if __name__ == "__main__":
    main() 