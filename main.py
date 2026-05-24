from dotenv import load_dotenv

from utils.audio import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_meeting_insights
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:

    print("🚀 Starting AI Video Assistant...\n")

    # STEP 1 — Audio Processing
    chunks = process_input(source)

    # STEP 2 — Transcription
    transcript = transcribe_all(chunks, language)

    print(
        f"\n📝 Raw transcription "
        f"(first 300 chars):\n{transcript[:300]}\n"
    )

    # STEP 3 — Metadata Generation
    title = generate_title(transcript)

    summary = summarize(transcript)

    # STEP 4 — Structured Extraction
    insights = extract_meeting_insights(transcript)

    # STEP 5 — Build RAG Chain
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,

        "action_items": insights.get("action_items", []),

        "key_decisions": insights.get("key_decisions", []),

        "open_questions": insights.get("open_questions", []),

        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    source = input(
        "Enter YouTube URL or local file path: "
    ).strip()

    language = (
        input("Language (english/hinglish): ").strip()
        or "english"
    )

    result = run_pipeline(source, language)

    print("\n" + "=" * 60)

    print(f"📌 TITLE: {result['title']}")

    print("\n📋 SUMMARY")
    print("-" * 60)
    print(result["summary"])

    print("\n✅ ACTION ITEMS")
    print("-" * 60)

    if result["action_items"]:

        for item in result["action_items"]:

            print(f"- Task: {item.get('task', 'N/A')}")
            print(f"  Owner: {item.get('owner', 'N/A')}")
            print(
                f"  Deadline: "
                f"{item.get('deadline', 'Not specified')}"
            )
            print()

    else:

        print("No action items found.")

    print("\n🔑 KEY DECISIONS")
    print("-" * 60)

    if result["key_decisions"]:

        for decision in result["key_decisions"]:

            print(f"- {decision}")

    else:

        print("No key decisions found.")

    print("\n❓ OPEN QUESTIONS")
    print("-" * 60)

    if result["open_questions"]:

        for question in result["open_questions"]:

            print(f"- {question}")

    else:

        print("No open questions found.")

    print("\n" + "=" * 60)

    # STEP 6 — Interactive RAG Chat
    print("\n💬 Chat with your transcript")
    print("Type 'exit' to quit.\n")

    rag_chain = result["rag_chain"]

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:

            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)

        print(f"\n🤖 Assistant:\n{answer}\n")