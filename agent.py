import os
from typing import List, Dict, TypedDict, Annotated, Literal, Union, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import StateGraph

# Define END constant
END = "end"

# Load environment variables
load_dotenv()

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for the given query."""
    # Mock implementation
    return f"Results for: {query}\n- Found information about {query}\n- Additional details: This is a simulation of web search"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Define the state
class AgentState(TypedDict):
    messages: List
    current_node: str
    function_calls: Optional[List]
    pending_function_calls: Optional[List]
    function_results: Optional[List]

# Define models for each node's input and output
class FunctionCallOutput(BaseModel):
    name: str
    arguments: Dict

class ToolChoice(BaseModel):
    type: Literal["function"]
    function: Dict[str, str]

# Node implementations
def route_node(state: AgentState) -> str:
    """Determine the next node to visit."""
    if state.get("pending_function_calls"):
        return "function_node"
    else:
        return state["current_node"]

def user_node(state: AgentState, user_message: str) -> AgentState:
    """Process user input."""
    messages = state.get("messages", [])
    messages.append(HumanMessage(content=user_message))
    
    return {
        "messages": messages,
        "current_node": "agent_node",
        "function_calls": [],
        "pending_function_calls": [],
        "function_results": []
    }

def agent_node(state: AgentState) -> AgentState:
    """Core agent logic."""
    messages = state.get("messages", [])
    
    # Create system message
    system_message = SystemMessage(content="""
    You are a helpful assistant with access to tools. 
    When asked a question, determine if you need to use a tool.
    If you need to use a tool, call the appropriate function.
    If you have the answer, respond directly.
    """)
    
    # Set up the agent model
    agent_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Prepare the messages for the model
    prompt_messages = [system_message] + messages
    
    # Available tools
    tools = [search_web, calculator]
    
    # Invoke the model
    response = agent_model.invoke(
        prompt_messages,
        tools=tools
    )
    
    # Check if the model wants to call a function
    if response.additional_kwargs.get("tool_calls"):
        tool_calls = response.additional_kwargs["tool_calls"]
        pending_calls = []
        
        for tool_call in tool_calls:
            function_call = {
                "name": tool_call["function"]["name"],
                "arguments": tool_call["function"]["arguments"],
                "id": tool_call["id"]
            }
            pending_calls.append(function_call)
        
        return {
            "messages": messages,
            "current_node": "function_node",
            "function_calls": state.get("function_calls", []) + pending_calls,
            "pending_function_calls": pending_calls,
            "function_results": state.get("function_results", [])
        }
    else:
        # No function call, just add the response to messages
        messages.append(response)
        return {
            "messages": messages,
            "current_node": END,
            "function_calls": state.get("function_calls", []),
            "pending_function_calls": [],
            "function_results": state.get("function_results", [])
        }

def function_node(state: AgentState) -> AgentState:
    """Execute function calls."""
    messages = state.get("messages", [])
    pending_calls = state.get("pending_function_calls", [])
    
    available_tools = {
        "search_web": search_web,
        "calculator": calculator
    }
    
    results = []
    
    for call in pending_calls:
        tool_name = call["name"]
        tool_args = eval(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
        
        if tool_name in available_tools:
            tool_fn = available_tools[tool_name]
            try:
                result = tool_fn(**tool_args)
                results.append({
                    "name": tool_name,
                    "result": result,
                    "id": call["id"]
                })
                messages.append(FunctionMessage(
                    name=tool_name,
                    content=result
                ))
            except Exception as e:
                error_message = f"Error executing {tool_name}: {str(e)}"
                results.append({
                    "name": tool_name,
                    "result": error_message,
                    "id": call["id"]
                })
                messages.append(FunctionMessage(
                    name=tool_name,
                    content=error_message
                ))
    
    return {
        "messages": messages,
        "current_node": "agent_node",
        "function_calls": state.get("function_calls", []),
        "pending_function_calls": [],
        "function_results": state.get("function_results", []) + results
    }

# Create and compile the graph
def create_agent_graph():
    # Initialize the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("user_node", user_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("function_node", function_node)
    
    # Add conditional edges
    graph.add_conditional_edges(
        "user_node",
        route_node,
        {
            "agent_node": "agent_node",
            "function_node": "function_node"
        }
    )
    
    graph.add_conditional_edges(
        "agent_node",
        route_node,
        {
            "agent_node": "agent_node",
            "function_node": "function_node",
            END: END
        }
    )
    
    graph.add_conditional_edges(
        "function_node",
        route_node,
        {
            "agent_node": "agent_node",
            "function_node": "function_node"
        }
    )
    
    # Set the entry point
    graph.set_entry_point("user_node")
    
    return graph.compile()

# Helper function to run the agent
def run_agent(user_input: str):
    agent = create_agent_graph()
    
    # Initialize state
    state = {
        "messages": [],
        "current_node": "user_node",
        "function_calls": [],
        "pending_function_calls": [],
        "function_results": []
    }
    
    # Run the agent
    result = agent.invoke({"user_message": user_input, **state})
    
    # Extract and return AI messages
    return [msg for msg in result["messages"] if isinstance(msg, AIMessage)]

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    responses = run_agent(user_query)
    
    for response in responses:
        print(f"AI: {response.content}")
        print("---") 