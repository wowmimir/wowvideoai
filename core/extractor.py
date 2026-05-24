from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm
import json


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


EXTRACTION_PROMPT = """
You are an expert meeting analyst.

Extract the following from this meeting transcript chunk:

1. Action Items
2. Key Decisions
3. Open Questions

Return ONLY valid JSON.

Format:

{{
  "action_items": [
    {{
      "task": "",
      "owner": "",
      "deadline": ""
    }}
  ],
  "key_decisions": [],
  "open_questions": []
}}

If a section has no items, return an empty list.

Transcript:
{text}
"""



def extract_meeting_insights(transcript: str):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        EXTRACTION_PROMPT
    )

    chain = prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    all_actions = []
    all_decisions = []
    all_questions = []

    for chunk in chunks:

        try:

            result = chain.invoke({"text": chunk})

            print("\nRAW LLM OUTPUT:")
            print(result)
            print("=" * 60)

            result = result.strip()

            result = result.replace("```json", "")
            result = result.replace("```", "")

            result = result.strip()

            parsed = json.loads(result)



            all_actions.extend(
                parsed.get("action_items", [])
            )

            all_decisions.extend(
                parsed.get("key_decisions", [])
            )

            all_questions.extend(
                parsed.get("open_questions", [])
            )

        except Exception as e:

            print(f"Extraction failed: {e}")

    return {
        "action_items": dedupe_actions(all_actions),
        "key_decisions": dedupe_list(all_decisions),
        "open_questions": dedupe_list(all_questions)
    }

def dedupe_list(items):

    seen = set()
    result = []

    for item in items:

        normalized = item.strip().lower()

        if normalized not in seen:

            seen.add(normalized)
            result.append(item)

    return result

def dedupe_actions(actions):

    seen = set()
    result = []

    for action in actions:

        key = (
            action.get("task", "").strip().lower(),
            action.get("owner", "").strip().lower()
        )

        if key not in seen:

            seen.add(key)
            result.append(action)

    return result