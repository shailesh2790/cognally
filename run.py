import os
from dotenv import load_dotenv
from agent import run_agent

# Load environment variables
load_dotenv()

def main():
    print("======================================")
    print("LangGraph Agent with Tool Usage")
    print("======================================")
    print("Type 'exit' to quit the chat")
    print()
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        print("\nProcessing...\n")
        
        try:
            responses = run_agent(user_input)
            
            for response in responses:
                print(f"AI: {response.content}")
                
            print("\n--------------------------------------\n")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("\n--------------------------------------\n")

if __name__ == "__main__":
    main() 