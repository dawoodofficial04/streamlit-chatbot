# DAARMO Chatbot 🤖

A multi-chat AI assistant built with **Streamlit**, **LangChain (Groq)**, and **LangGraph**. DAARMO keeps a separate, persistent conversation per chat, remembers context within each chat, and renders LaTeX math formulas properly.

## Features

- **Multiple chats** — start new conversations from the sidebar, switch between them, and delete ones you no longer need.
- **Persistent history** — conversations are saved to a local `chat_history.json` file, so they survive a page reload or app restart (see [Deployment notes](#deployment-notes) for an important caveat on Streamlit Community Cloud).
- **Conversational memory** — powered by LangGraph's `StateGraph` + `InMemorySaver`, so the model remembers earlier turns within the same chat without you having to resend the whole history yourself.
- **LaTeX-aware formulas** — math the model writes (`$...$` / `$$...$$`, or stray `\[ \]` / `\( \)` delimiters) is normalized and rendered properly instead of showing up as raw text.
- **Emoji picker** — a quick popover of frequently used emojis you can insert into your message.
- **Scroll-to-bottom button** — a floating button that jumps you to the latest message once a chat gets long.
- **Dark, hero-style UI** with a custom-styled delete button (red on hover) for chat history entries.

## Tech stack

| Piece | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [LangChain](https://python.langchain.com) | Message types / LLM interface |
| [`langchain-groq`](https://python.langchain.com/docs/integrations/chat/groq/) | Chat model client (Groq's `openai/gpt-oss-120b`) |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful graph + `InMemorySaver` checkpointing for chat memory |
| `python-dotenv` | Loads the Groq API key from a `.env` file |

## Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/keys)

## Setup

1. **Clone the repo and enter the project folder.**

2. **Create a virtual environment (recommended) and install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your Groq API key.** Create a `.env` file in the project root:

   ```
   GROQ_API_KEY=your_key_here
   ```

4. **Run the app:**

   ```bash
   streamlit run chatbot.py
   ```

   The app opens at `http://localhost:8501`.

## Project structure

```
.
├── chatbot.py        # Main Streamlit app
├── chat_history.json   # Auto-created on first run — saved chat history (git-ignored)
├── requirements.txt
├── .env                # Your Groq API key (git-ignored, not committed)
└── README.md
```

## Deployment notes

⚠️ **If you deploy this on Streamlit Community Cloud** (or any host with an ephemeral filesystem):

- `chat_history.json` is written to local disk next to the script. Community Cloud containers get rebuilt on redeploys, after periods of inactivity, and during routine maintenance — wiping that file each time. There's also no file browser or SSH access to retrieve it manually.
- Likewise, `InMemorySaver` keeps conversational memory only in server RAM for the life of the running process — a full app restart clears it too, even though the JSON transcript (while it lasts) still displays the old messages.
- For chat history and memory that survive restarts on a platform like this, swap the local JSON file and `InMemorySaver` for a persistent store (e.g. a hosted database, or a LangGraph checkpointer backed by SQLite/Postgres).

On a self-hosted server or your own machine, none of this is an issue — the file and process simply persist as long as you don't delete them or stop the server.

## .gitignore

Make sure the following are **not** committed:

```
chat_history.json
.env
__pycache__/
*.pyc
.venv/
```

`chat_history.json` holds real conversation data and shouldn't be tracked in version control; `.env` holds your API key.

## License

Add your preferred license here (e.g. MIT).
