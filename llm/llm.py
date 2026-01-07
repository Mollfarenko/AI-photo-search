from langchain_openai import ChatOpenAI

def load_llm():
    """
    Load OpenAI LLM for agent reasoning.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=300,
    )
