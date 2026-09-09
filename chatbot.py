import os
import json
import re
import uuid
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, MessagesState

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="DAARMO", page_icon="🤖", layout="wide")

SYSTEM_PROMPT = """
Your name is DAARMO.
You are created by Dawood Hussain.
When writing any mathematical formula or equation, always format it as LaTeX
using $ for inline math (e.g. $E = mc^2$) and $$ for standalone/display math
(e.g. $$P = \\frac{F}{A}$$). Never wrap math in \\[ \\], \\( \\), or bare
parentheses -- always use the $ / $$ delimiters, including inside tables.
"""

COMMON_EMOJIS = ["😀", "😂", "😍", "👍", "🙏", "🤔", "🔥", "🎉", "❤️", "😢", "😎", "🚀"]

# ---------------------------------------------------------------------------
# Persistence (JSON file on disk)
# ---------------------------------------------------------------------------
# st.session_state only lives for the current browser session / process, so
# it disappears on a page reload or app restart. Saving to a JSON file next
# to the script gives chats a home that survives both.
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.json")


def load_chats_from_disk():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("chats", {}), data.get("current_chat_id")
        except (json.JSONDecodeError, OSError):
            pass
    return {}, None


def save_chats_to_disk():
    data = {
        "chats": st.session_state.chats,
        "current_chat_id": st.session_state.current_chat_id,
    }
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Custom CSS (dark hero-style layout)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero-title { font-size: 2.4rem; font-weight: 700; text-align: center; margin-top: 4rem; }
    .hero-sub { font-size: 1.2rem; text-align: center; opacity: 0.85; }
    .hero-tag { font-size: 1rem; text-align: center; opacity: 0.55; margin-bottom: 3rem; }
    .stChatMessage { border-radius: 12px; }
    button[title="Delete this chat"] {
        transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    }
    button[title="Delete this chat"]:hover,
    div[class*="st-key-delete_btn_"] button:hover {
        background-color: #DC2626 !important;
        border-color: #DC2626 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------
# `chats` holds every conversation: {chat_id: {"title": str, "messages": [...]}}
# `current_chat_id` points at the conversation currently shown in the main panel.
if "chats" not in st.session_state or "current_chat_id" not in st.session_state:
    loaded_chats, loaded_current_id = load_chats_from_disk()
    st.session_state.chats = loaded_chats

    if st.session_state.chats and loaded_current_id in st.session_state.chats:
        st.session_state.current_chat_id = loaded_current_id
    elif st.session_state.chats:
        # Fall back to the most recently created chat if the saved
        # current_chat_id is missing or stale.
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New chat", "messages": []}
        st.session_state.current_chat_id = new_id
        save_chats_to_disk()

if "input_buffer" not in st.session_state:
    st.session_state.input_buffer = ""


def start_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New chat", "messages": []}
    st.session_state.current_chat_id = new_id
    save_chats_to_disk()


def switch_chat(chat_id: str):
    st.session_state.current_chat_id = chat_id
    save_chats_to_disk()


def delete_chat(chat_id: str):
    st.session_state.chats.pop(chat_id, None)

    if not st.session_state.chats:
        # Keep at least one chat around instead of leaving the app empty.
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New chat", "messages": []}
        st.session_state.current_chat_id = new_id
    elif st.session_state.current_chat_id == chat_id:
        # Deleted chat was active -- fall back to the most recent remaining one.
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]

    save_chats_to_disk()


def render_math(content: str) -> str:
    """Streamlit's markdown renderer only recognizes $...$ / $$...$$ for
    LaTeX. Some models (including this one, occasionally) emit \\[ \\] or
    \\( \\) instead, which then shows up as raw, unrendered text. Convert
    those delimiters so formulas always render properly."""
    content = re.sub(r"\\\[(.*?)\\\]", lambda m: f"$$ {m.group(1).strip()} $$", content, flags=re.DOTALL)
    content = re.sub(r"\\\((.*?)\\\)", lambda m: f"$ {m.group(1).strip()} $", content, flags=re.DOTALL)
    return content


@st.cache_resource
def get_chat_graph():
    """Build the LangGraph app once per server process and reuse it across
    every rerun. Streamlit reruns this whole script on every interaction, so
    without @st.cache_resource a brand-new InMemorySaver() (and therefore
    empty memory) would be created on every single click or keystroke."""
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )

    def call_model(state: MessagesState):
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": response}

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    return workflow.compile(checkpointer=InMemorySaver())


def add_emoji(emoji: str):
    st.session_state.input_buffer += emoji


def handle_send():
    """Runs as an on_click callback, i.e. BEFORE the script reruns and the
    input_buffer widget is redrawn -- this is what makes it safe to clear
    st.session_state.input_buffer here."""
    user_input = st.session_state.input_buffer.strip()
    if not user_input:
        return

    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]

    if not messages:
        current_chat["title"] = user_input[:30]

    messages.append({"role": "user", "content": user_input})

    graph = get_chat_graph()
    config = {"configurable": {"thread_id": st.session_state.current_chat_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    response_content = result["messages"][-1].content

    messages.append({"role": "assistant", "content": response_content})

    # Safe here: this callback runs before the widget is re-instantiated.
    st.session_state.input_buffer = ""

    save_chats_to_disk()


# ---------------------------------------------------------------------------
# Sidebar: New Chat + chat history list
# ---------------------------------------------------------------------------
with st.sidebar:
    st.button("➕ New Chat", use_container_width=True, on_click=start_new_chat)
    st.divider()
    st.caption("Chat history")

    # Most recent chat first
    for chat_id in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[chat_id]
        label = chat["title"]
        is_active = chat_id == st.session_state.current_chat_id

        row_col, delete_col = st.columns([6, 2])
        with row_col:
            st.button(
                ("💬 " if not is_active else "🟢 ") + label,
                key=f"chat_btn_{chat_id}",
                use_container_width=True,
                on_click=switch_chat,
                args=(chat_id,),
            )
        with delete_col:
            st.button(
                "🗑️",
                key=f"delete_btn_{chat_id}",
                on_click=delete_chat,
                args=(chat_id,),
                help="Delete this chat",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
current_chat = st.session_state.chats[st.session_state.current_chat_id]
messages = current_chat["messages"]

if not messages:
    st.markdown('<div class="hero-title">Hey, this is DAARMO</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">How can I help you today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-tag">I\'m a smart genius assistant</div>', unsafe_allow_html=True)
else:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(render_math(msg["content"]))

# Floating button to jump to the bottom of the page when the chat gets long.
# st.markdown's HTML sanitizer strips inline event handlers like onclick even
# with unsafe_allow_html=True, so a plain <button onclick=...> never actually
# fires. components.html runs a real <script> (in an iframe, but same-origin,
# so it can reach window.parent.document) that builds the button directly in
# the actual page and wires up a real click listener -- this is what makes it
# work. It also removes any previous copy of itself first so reruns don't
# stack up duplicate buttons.
components.html(
    """
    <script>
    (function() {
        var doc = window.parent.document;
        var existing = doc.getElementById('scrollBottomBtn');
        if (existing) { existing.remove(); }

        var btn = doc.createElement('button');
        btn.id = 'scrollBottomBtn';
        btn.title = 'Scroll to bottom';
        btn.innerHTML = '&#8595;';
        Object.assign(btn.style, {
            position: 'fixed',
            bottom: '5.5rem',
            right: '2rem',
            zIndex: '1000',
            width: '3rem',
            height: '3rem',
            borderRadius: '50%',
            border: '1px solid rgba(255,255,255,0.2)',
            backgroundColor: '#262730',
            color: 'white',
            fontSize: '1.2rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
            transition: 'background-color 0.15s ease'
        });

        btn.addEventListener('mouseenter', function() {
            btn.style.backgroundColor = '#3a3b45';
        });
        btn.addEventListener('mouseleave', function() {
            btn.style.backgroundColor = '#262730';
        });

        btn.addEventListener('click', function() {
            var selectors = [
                'section[data-testid="stMain"]',
                '[data-testid="stAppViewContainer"]',
                '.main'
            ];
            for (var i = 0; i < selectors.length; i++) {
                var el = doc.querySelector(selectors[i]);
                if (el && el.scrollHeight > el.clientHeight + 5) {
                    el.scrollTo({top: el.scrollHeight, behavior: 'smooth'});
                    return;
                }
            }
            window.parent.scrollTo({top: doc.body.scrollHeight, behavior: 'smooth'});
        });

        doc.body.appendChild(btn);
    })();
    </script>
    """,
    height=0,
)

# ---------------------------------------------------------------------------
# Input row: emoji picker + text area + send button
# ---------------------------------------------------------------------------
emoji_col, input_col, send_col = st.columns([1, 8, 1])

with emoji_col:
    with st.popover("🙂"):
        st.caption("Frequently used")
        emoji_rows = [COMMON_EMOJIS[i : i + 4] for i in range(0, len(COMMON_EMOJIS), 4)]
        for row in emoji_rows:
            cols = st.columns(len(row))
            for col, emoji in zip(cols, row):
                col.button(emoji, key=f"emoji_{emoji}", on_click=add_emoji, args=(emoji,))

with input_col:
    user_input = st.text_input(
        "Message DAARMO...",
        key="input_buffer",
        label_visibility="collapsed",
        placeholder="Message Smart Genius...",
    )

with send_col:
    st.button("➤", use_container_width=True, on_click=handle_send)