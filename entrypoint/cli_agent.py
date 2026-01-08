# src/test_agent.py
from agents.agent_runtime import run_agent_text, run_agent_image

def main():
    while True:
        user_input = input("\n\033[1;36mYou:\033[0m ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        result = run_agent_text(user_input)
        
        # Clean output - just the response
        print(f"\n\033[1;32mAgent:\033[0m {result['response']}")
        
        # Optional: show metadata in gray
        print(f"\033[90m[{result['tool_calls']} tool call(s)]\033[0m\n")

