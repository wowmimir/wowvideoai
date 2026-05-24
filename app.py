import streamlit as st
from dotenv import load_dotenv

from utils.audio import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_meeting_insights
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
    --bg:#0b0b11;
    --surface:#11111a;
    --surface2:#181824;
    --border:#2b2b3b;
    --accent:#7c3aed;
    --accent2:#06b6d4;
    --text:#f5f5ff;
    --muted:#8f90aa;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background: var(--bg);
}

[data-testid="stSidebar"]{
    background: var(--surface) !important;
    border-right:1px solid var(--border);
}

.hero-title{
    font-family:'Syne', sans-serif;
    font-size:3rem;
    font-weight:800;
    line-height:1;
    background:linear-gradient(135deg, white 0%, #b794ff 50%, #67e8f9 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:0.4rem;
}

.hero-sub{
    color:var(--muted);
    font-size:0.8rem;
    letter-spacing:0.2em;
    text-transform:uppercase;
}

/* CARD LABEL — sits above the native Streamlit container */
.card-label{
    font-family:'Syne', sans-serif;
    font-size:0.9rem;
    font-weight:700;
    color:#c4b5fd;
    letter-spacing:0.08em;
    text-transform:uppercase;
    margin-bottom:0.5rem;
    
}

/* Style Streamlit's native containers to look like cards */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="column"] > div > [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

/* INPUTS */
.stTextInput input{
    background:var(--surface2) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    border-radius:10px !important;
}
[data-testid="InputInstructions"] {
    display: none !important;
}

.stSelectbox div[data-baseweb="select"]{
    background:var(--surface2) !important;
    border-radius:10px !important;
}

/* BUTTONS */
.stButton button{
    background:linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color:white !important;
    border:none !important;
    border-radius:10px !important;
    font-weight:700 !important;
    padding:0.6rem 1rem !important;
}

.stButton button:hover{
    transform:translateY(-1px);
}

/* CHAT BUBBLES */
.user-msg{
    display:block;
    margin-left:auto;
    background:rgba(124,58,237,0.18);
    border:1px solid rgba(124,58,237,0.4);
    padding:0.9rem 1rem;
    border-radius:14px;
    max-width:40%;
    margin-bottom:0.5rem;
    text-align:center;
}

.bot-msg{
    display:block;
    background:rgba(6,182,212,0.14);
    border:1px solid rgba(6,182,212,0.35);
    padding:0.9rem 1rem;
    border-radius:14px;
    max-width:80%;
    margin-bottom:0.5rem;
}

/* SCROLLBAR */
::-webkit-scrollbar{ width:6px; }
::-webkit-scrollbar-thumb{ background:#33354a; border-radius:10px; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="hero-title" style="font-size:2rem">🎬 WowVideo AI</div>'
        '<div class="hero-sub">Meeting Intelligence</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=...")
    language = st.selectbox("Language", ["english", "hinglish"])
    run_btn = st.button("⚡ Analyse", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">AI Video Assistant</div>'
    '<div class="hero-sub">Transcribe · Summarise · Extract · Chat</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please provide a URL or local file path.")
    else:
        try:
            with st.status("Running pipeline...", expanded=True) as status:
                st.write("🔊 Processing audio...")
                chunks = process_input(source)

                st.write("📝 Transcribing audio...")
                transcript = transcribe_all(chunks, language)

                st.write("🏷️ Generating title...")
                title = generate_title(transcript)

                st.write("📋 Summarising transcript...")
                summary = summarize(transcript)

                st.write("🧠 Extracting insights...")
                insights = extract_meeting_insights(transcript)

                st.write("🔎 Building RAG engine...")
                rag_chain = build_rag_chain(transcript)

                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "summary": summary,
                "transcript": transcript,
                "action_items": insights.get("action_items", []),
                "key_decisions": insights.get("key_decisions", []),
                "open_questions": insights.get("open_questions", []),
                "rag_chain": rag_chain,
            }
            st.session_state.chat_history = []
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result

    # ── TITLE ──
    st.markdown('<div class="card-label">📌 Meeting Title</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f'<div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:700;color:white;line-height:1.4;height:3.5rem;">'
            f'{result["title"]}</div>',
            unsafe_allow_html=True
        )

    # ── SUMMARY + TRANSCRIPT ──
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="card-label">📋 Summary</div>', unsafe_allow_html=True)
        with st.container(border=True, height=350):
            st.markdown(result["summary"])

    with col2:
        st.markdown('<div class="card-label">📝 Transcript</div>', unsafe_allow_html=True)
        with st.container(border=True, height=350):
            st.write(result["transcript"])

    # ── INSIGHTS ──
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="card-label">✅ Action Items</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if result["action_items"]:
                for item in result["action_items"]:
                    st.markdown(f"- {item}")
            else:
                st.markdown("No action items found.")

    with c2:
        st.markdown('<div class="card-label">🔑 Key Decisions</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if result["key_decisions"]:
                for item in result["key_decisions"]:
                    st.markdown(f"- {item}")
            else:
                st.markdown("No key decisions found.")

    with c3:
        st.markdown('<div class="card-label">❓ Open Questions</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if result["open_questions"]:
                for item in result["open_questions"]:
                    st.markdown(f"- {item}")
            else:
                st.markdown("No open questions found.")

    st.markdown("---")

    # ── CHAT ──
    st.markdown('<div class="card-label">💬 Chat With Your Meeting</div>', unsafe_allow_html=True)
    with st.container(border=True):

        # Chat history
        if st.session_state.chat_history:
            chat_html = ""
            for msg in st.session_state.chat_history:
                css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
                chat_html += f'<div class="{css_class}">{msg["content"]}</div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#8f90aa;padding:0.5rem 0">Ask anything about the meeting transcript.</div>',
                unsafe_allow_html=True
            )

        # Input row
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask about the meeting",
                placeholder="What were the key takeaways?",
                label_visibility="collapsed",
                key="chat_input"
            )
        with col2:
            send_btn = st.button("Send", use_container_width=True)

    # Send
    if send_btn and user_input.strip():
        with st.spinner("Thinking..."):
            answer = ask_question(result["rag_chain"], user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

# ─────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:6rem 2rem">
        <div style="font-size:5rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;margin-bottom:1rem">Ready to Analyse</div>
        <div style="color:#8f90aa;max-width:600px;line-height:1.8">
            Paste a YouTube URL or local media file in the sidebar and generate transcripts,
            summaries, action items, decisions, open questions, and a searchable AI chat.
        </div>
    </div>
    """, unsafe_allow_html=True)