# entrypoint/cli_agent.py
from agents.agent_runtime import run_agent_text
from tools.tool_message_extractor import extract_photos
from utilities.url_generator import S3PhotoResolver
from utilities.photo_viewer import PhotoViewer

CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
GRAY = "\033[90m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"

def main():
    print("\n" + "="*60)
    print("     Photo Search CLI")
    print("="*60)

    photo_viewer = PhotoViewer()

    while True:
        user_input = input(f"\n{CYAN}You:{RESET} ")

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        result = run_agent_text(user_input)

        print(f"\n{GREEN}Agent:{RESET} {result['response']}")

        photos = result.get("photos", [])

        stats = f"[{result.get('tool_calls', 0)} tool call(s)]"
        if photos:
            stats += f" • {len(photos)} photo(s) found"
        print(f"{GRAY}{stats}{RESET}")

        if photos:
            choice = input(f"\n{YELLOW}Open photos in browser? (y/n):{RESET} ").lower()
            if choice in ("y", "yes"):
                photo_viewer.show_photos(photos, url_generator)

if __name__ == "__main__":
    main()





