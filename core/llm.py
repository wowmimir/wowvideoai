from langchain_ollama import ChatOllama

_llm = None

def get_llm():

    global _llm

    if _llm is None:

        _llm = ChatOllama(
            model="gemma4:31b-cloud",
            temperature=0.2
        )

    return _llm