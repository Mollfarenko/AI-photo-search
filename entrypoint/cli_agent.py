# entrypoint/cli_agent.py
from agents.agent_runtime import run_agent_text, run_agent_image

CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
GRAY = "\033[90m"
RESET = "\033[0m"

def extract_tool_calls(messages):
    """Extract tool calls made by the LLM."""
    tool_calls = []
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(tc)
    return tool_calls


def main(show_tools=True):
    while True:
        user_input = input(f"\n{CYAN}You:{RESET} ")

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        result = run_agent_text(user_input)

        # Main response
        print(f"\n{GREEN}Agent:{RESET} {result['response']}")

        # Tool call trace
        if show_tools:
            tool_calls = extract_tool_calls(result["messages"])
            if tool_calls:
                print(f"\n{GRAY}[LLM → Tool decision]{RESET}")
                for tc in tool_calls:
                    print(
                        f"{GRAY}- {tc['name']}("
                        + ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                        + f"){RESET}"
                    )

        print(f"{GRAY}[{result.get('tool_calls', 0)} tool call(s)]{RESET}\n")


if __name__ == "__main__":
    main()



