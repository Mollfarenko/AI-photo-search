# src/test_agent.py
from agents.agent_runtime import run_agent_text, run_agent_image

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    response = run_agent_text(user_input)
    print("Agent:", response)
