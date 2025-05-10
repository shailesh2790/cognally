import os
from typing import Dict, List, TypedDict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define the state
class AgentState(TypedDict):
    messages: List
    agent_scratchpad: Optional[str]
    next: Optional[str]

# Define nodes
def user_node(state: Dict) -> Dict:
    """Handle user input"""
    messages = state.get("messages", [])
    user_message = state.get("user_message", "")
    
    # Add system message if it's the first message
    if not messages:
        messages.append(SystemMessage(content="You are a helpful AI assistant that can help with various tasks. Be concise and specific in your responses."))
    
    messages.append(HumanMessage(content=user_message))
    
    return {
        "messages": messages,
        "agent_scratchpad": "",
        "next": "planner"
    }

def planner_node(state: Dict) -> Dict:
    """Plan the next steps"""
    messages = state.get("messages", [])
    
    # Create a planner prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a thoughtful planner. 
        Your job is to analyze the user's request and create a step-by-step plan to address it effectively.
        Be specific and tailor your plan to exactly what the user has asked for.
        Focus only on the user's current request."""),
        ("human", "User request: {input}\n\nCreate a clear, step-by-step plan to address this specific request:")
    ])
    
    # Set up the planner model
    planner_model = ChatOpenAI(temperature=0)
    
    # Get the last user message
    last_message = messages[-1].content
    
    # Generate a plan
    plan = planner_model.invoke(prompt.format(input=last_message))
    
    # Update the state
    return {
        "messages": messages,
        "agent_scratchpad": plan.content,
        "next": "executor"
    }

def executor_node(state: Dict) -> Dict:
    """Execute the plan"""
    messages = state.get("messages", [])
    scratchpad = state.get("agent_scratchpad", "")
    
    # Create an executor prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an executor AI. 
        Your job is to carefully follow the provided plan to address the user's specific request.
        Provide a helpful, complete response that directly answers what the user asked for.
        Stay focused on the current request and don't introduce unrelated topics."""),
        ("human", "User request: {input}\n\nPlan to follow:\n{plan}\n\nPlease execute this plan and provide a complete response:")
    ])
    
    # Set up the executor model
    executor_model = ChatOpenAI(temperature=0)
    
    # Get the last user message
    last_message = messages[-1].content
    
    # Execute the plan
    result = executor_model.invoke(prompt.format(plan=scratchpad, input=last_message))
    
    # Add the result to messages
    messages.append(AIMessage(content=result.content))
    
    # Update the state
    return {
        "messages": messages,
        "agent_scratchpad": "",
        "next": None
    }

# Create router
def router(state: Dict) -> str:
    """Route to the next node or end."""
    if state.get("next") == "planner":
        return "planner"
    elif state.get("next") == "executor":
        return "executor"
    else:
        return END

# Build the graph
def build_graph():
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("user", user_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    
    # Add edges
    graph.add_conditional_edges("user", router, {
        "planner": "planner",
        "executor": "executor",
        END: END
    })
    
    graph.add_conditional_edges("planner", router, {
        "planner": "planner",
        "executor": "executor",
        END: END
    })
    
    graph.add_conditional_edges("executor", router, {
        "planner": "planner",
        "executor": "executor",
        END: END
    })
    
    # Set entry point
    graph.set_entry_point("user")
    
    # Compile graph
    return graph.compile()

def main():
    # Build graph
    agent = build_graph()
    
    # Get user input
    user_message = input("Enter your request: ")
    
    # Initialize state
    state = {
        "messages": [],
        "agent_scratchpad": "",
        "next": None,
        "user_message": user_message
    }
    
    # Run agent
    result = agent.invoke(state)
    
    # Print result
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"AI: {message.content}")

if __name__ == "__main__":
    main() 