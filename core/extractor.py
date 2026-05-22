from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda



def get_llm():
    return ChatOllama(
        model="gemma4:latest",
        temperature=0.2
    )

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def build_chain(system_prompt: str):
    llm = get_llm()

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}")
            ]
        )
        | llm
        | StrOutputParser()
    )


def process_transcript(
    transcript: str,
    map_prompt: str,
    combine_prompt: str
) -> str:

    chunks = split_transcript(transcript)

    map_chain = build_chain(map_prompt)

    partial_outputs = [
        map_chain.invoke(chunk)
        for chunk in chunks
    ]

    combined_text = "\n\n".join(partial_outputs)

    combine_chain = build_chain(combine_prompt)

    return combine_chain.invoke(combined_text)


def extract_actions(transcript: str) -> str:

    map_prompt = (
        "Extract all action items from this portion of the meeting transcript.\n\n"
        "For each action item provide:\n"
        "- Task description\n"
        "- Owner\n"
        "- Deadline (or 'Not specified')\n\n"
        "If none found say 'No action items found.'"
    )

    combine_prompt = (
        "Combine and deduplicate these extracted action items into one final list.\n"
        "Merge similar items.\n"
        "Keep the final output concise and structured."
    )

    return process_transcript(
        transcript,
        map_prompt,
        combine_prompt
    )


def extract_key_decisions(transcript: str) -> str:

    map_prompt = (
        "Extract all key decisions made in this portion of the meeting transcript.\n"
        "Format as a numbered list.\n"
        "If none found say 'No key decisions found.'"
    )

    combine_prompt = (
        "Combine and deduplicate these key decisions into one final clean numbered list."
    )

    return process_transcript(
        transcript,
        map_prompt,
        combine_prompt
    )


def extract_questions(transcript: str) -> str:

    map_prompt = (
        "Extract unresolved questions or follow-up topics from this portion "
        "of the meeting transcript.\n"
        "Format as a numbered list.\n"
        "If none found say 'No open questions found.'"
    )

    combine_prompt = (
        "Combine and deduplicate these unresolved questions into one final list."
    )

    return process_transcript(
        transcript,
        map_prompt,
        combine_prompt
    )