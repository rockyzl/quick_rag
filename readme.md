# Quick RAG Chatbot

An end‑to‑end Retrieval‑Augmented Generation (RAG) demo built with LangChain, OpenAI, Chroma, and Gradio. It ingests your PDF knowledge base into a local vector store, then serves a conversational UI that can cite sources on demand and remembers who you are across turns.

## What’s inside
- **Ingestion pipeline (`ingest_database.py`)**: loads PDFs from `data/`, chunks them with `RecursiveCharacterTextSplitter`, embeds using `text-embedding-3-large`, and writes to a persistent Chroma DB in `chroma_db/`.
- **Chat backend (`chatbot.py`)**:  
  - Streams answers from `gpt-4o-mini` with retrieved context.  
  - On-demand references: only shows source snippets when the user asks for “reference/source/evidence”.  
  - Conversation memory: summarizes recent turns and keeps a lightweight user profile (name, optional location) via an auxiliary LLM extractor.  
  - Natural tone prompt that avoids robotic “based on…” phrasing.
- **UI**: Gradio ChatInterface for quick local testing.

## Quick start
1) **Prepare environment**
```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

2) **Set secrets**  
Copy `.env.example` → `.env` and fill `OPENAI_API_KEY`. (Optional) set `CHROMA_TELEMETRY=false`.

3) **Add documents**  
Place one or more PDFs in the `data/` folder.

4) **Build the vector store**
```bash
python ingest_database.py
```

5) **Run the chat UI**
```bash
python chatbot.py
```
Then open the local URL shown in the console.

## Usage tips
- Ask normal questions; context is pulled from your PDFs.  
- To see citations, include words like “reference”, “source”, or “evidence” in your question.  
- To update how the bot addresses you, just say “Call me <name>”; location is mentioned at most once unless you bring it up again.  
- If you type a possibly misspelled place/name, the bot will gently ask for confirmation once and then move on.
- Sharing the Gradio link: the visible chat window history is per-browser-session, but the current code’s internal memory/user-profile are global in the process. If multiple people use the same share link at once, their profiles could mingle. To isolate users, move the memory/profile into a per-session `gr.State()` (not yet implemented here).

## Project structure
```
chatbot.py          # Chat runtime, retrieval, streaming, memory, profile handling
ingest_database.py  # PDF loader, splitter, embedding, Chroma writer
requirements.txt    # Locked package versions
.env.example        # API key placeholder
data/               # Put your PDFs here
chroma_db/          # Persisted vector store (created after ingestion)
```

## Notes / constraints
- Uses OpenAI models; ensure your key has access to `gpt-4o-mini` and `text-embedding-3-large`.
- Chroma persists locally; delete `chroma_db/` to rebuild from scratch.
- This repo currently targets Python 3.11+ (per original tutorial).

## Credits
Based on Thomas Janssen’s “Chatbot with RAG and LangChain” tutorial, adapted with conversational memory, on-demand references, and a refined prompt style.
