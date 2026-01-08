# entrypoint/cli_agent.py
from agents.agent_runtime import run_agent_text, run_agent_image

CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
GRAY = "\033[90m"
RESET = "\033[0m"

def main():
    while True:
        user_input = input(f"\n{CYAN}You:{RESET} ")

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        result = run_agent_text(user_input)

        print(f"\n{GREEN}Agent:{RESET} {result['response']}")
        print(f"{GRAY}[{result.get('tool_calls', 0)} tool call(s)]{RESET}\n")


if __name__ == "__main__":
    main()


