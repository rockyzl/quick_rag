from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import json

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the model
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# separate non-streaming LLM for profile extraction
profile_llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')

APP_CSS = """
:root {
  --bg: #111216;
  --panel: #111216;
  --bubble: #1c1f2a;
  --text: #e5e7eb;
  --muted: #a5adbe;
  --accent: #2f80ed;
}
body, .gradio-container {
  background: var(--bg);
  color: var(--text);
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}
.gradio-container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 12px 12px 18px;
}
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 {
  color: var(--text);
}
.chatbot {
  border-radius: 12px !important;
  box-shadow: 0 12px 34px rgba(0, 0, 0, 0.4);
  border: 1px solid #1b1d26;
  background: #0f1116;
  padding: 4px;
}
.message {
  max-width: 86% !important;
  min-width: 240px !important;
  border-radius: 12px !important;
  padding: 8px 12px !important;
  margin: 6px 0 !important;
  font-size: 15px !important;
  line-height: 1.4 !important;
  word-break: normal !important;
  white-space: normal !important;
  display: inline-block !important;
  border: none !important;
  background: #1b1d26;
  color: var(--text) !important;
  box-shadow: 0 6px 12px rgba(0,0,0,0.28);
}
.message.user {
  background: #1f2937;
  border: 1px solid rgba(47,128,237,0.2);
}
.svelte-drgfj3 input, .svelte-drgfj3 textarea {
  border-radius: 12px !important;
  border: 1px solid #20222c !important;
  background: #161821 !important;
  color: var(--text) !important;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.25);
}
.gr-button-primary {
  background: #2f80ed;
  border-radius: 10px;
  border: none;
  box-shadow: 0 8px 20px rgba(47, 128, 237, 0.35);
  font-weight: 600;
  color: #f8fafc;
}
.gr-button-secondary, .gr-button {
  border-radius: 9px;
}
.block.gradio-accordion, .panel {
  border-radius: 12px;
}
.caption {
  color: var(--muted);
  margin-bottom: 10px;
}
"""

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

user_profile: dict[str, str] = {}
# Track whether we've already introduced the full location text once.
greeted_with_location = False

def extract_profile_from_message(message: str) -> dict:
    if not message or len(message.strip()) < 5:
        return {}

    prompt = f"""
You are a context-aware assistant whose job is to extract a concise user profile
from a single sentence. The sentence may contain typos.
Return a JSON object containing exactly these keys: "name", "location", "confidence", "note".
Use empty strings if a key cannot be determined. Honesty matters—if spelling seems wrong, mention it in "note"
and set "confidence" to "low". Respond only with a single JSON object, no explanation.

Sentence: "{message}"
"""
    try:
        response = profile_llm.predict(prompt)
        response_text = response.strip()
        if not response_text.startswith("{"):
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1:
                response_text = response_text[start:end+1]
        parsed = json.loads(response_text)
        return {
            "name": parsed.get("name", "").strip(),
            "location": parsed.get("location", "").strip(),
            "confidence": parsed.get("confidence", "").strip(),
            "note": parsed.get("note", "").strip(),
        }
    except Exception:
        return {}

def update_user_profile(message: str):
    if not message:
        return

    profile_delta = extract_profile_from_message(message)
    if profile_delta.get("name"):
        user_profile["name"] = profile_delta["name"]
    if profile_delta.get("location"):
        user_profile["location"] = profile_delta["location"]
    if profile_delta.get("note"):
        user_profile["note"] = profile_delta["note"]
    if profile_delta:
        user_profile["intro"] = message.strip()
    if profile_delta.get("confidence") in {"high", "medium"} and profile_delta.get("name"):
        user_profile["greeting"] = profile_delta["name"]

def format_profile() -> str:
    if not user_profile:
        return "Unknown"
    return "; ".join(f"{k}={v}" for k, v in user_profile.items())

def summarize_history(messages: list[BaseMessage], limit: int = 10) -> str:
    if not messages:
        return "No prior conversation yet."
    relevant = messages[-limit:]
    lines = []
    for msg in relevant:
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            role = "Human"
        elif msg_type == "ai":
            role = "AI"
        else:
            role = msg.__class__.__name__
        content = msg.content.replace("\n", " ").strip()
        lines.append(f"{role}: {content}")
    return " | ".join(lines)


def should_include_references(message: str) -> bool:
    if not message:
        return False
    lowered = message.lower()
    keywords = [
        "reference",
        "source",
        "evidence",
        "cite",
        "citation",
        "where did",
        "which paper",
        "which document",
        "proof",
    ]
    return any(keyword in lowered for keyword in keywords)

# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    update_user_profile(message)

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    reference_lines = []
    seen_sources = set()

    for doc in docs:
        knowledge += doc.page_content + "\n\n"

        source = doc.metadata.get("source", "unknown source")
        page_info = doc.metadata.get("page") or doc.metadata.get("page_number")
        if page_info:
            source = f"{source} (page {page_info})"

        if source not in seen_sources:
            snippet = " ".join(doc.page_content.strip().split())
            if len(snippet) > 120:
                snippet = snippet[:120].rstrip() + "..."
            reference_lines.append(f"- {source}: {snippet}")
            seen_sources.add(source)

    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""
        references_text = ""
        memory_vars = memory.load_memory_variables({})
        conversation_history_summary = summarize_history(memory_vars.get("chat_history", []))
        profile_summary = format_profile()
        latest_intro = user_profile.get("intro", "")

        rag_prompt = f"""
        You are a knowledgeable, conversational assistant designed to support an ongoing dialogue with the same user over time.

        Your goals are:
        - Be accurate and grounded when using the provided knowledge.
        - Sound natural, relaxed, and human in conversation.
        - Maintain continuity: remember who the user is without repeatedly re-introducing or re-confirming their identity.

        Conversation style guidelines:
        - Address the user by their name naturally, the way a person would in an ongoing conversation.
        - Do NOT repeat location or identity details unless the user explicitly brings them up again or there is genuine ambiguity.
        - Avoid phrases that feel like system narration (e.g., “based on the knowledge”, “according to the data”) unless clarity truly requires it.
        - If knowledge is used, weave it in smoothly, as part of the explanation—not as a citation.

        Tone:
        - Friendly, calm, and thoughtful.
        - Curious when appropriate, but not interrogative.
        - Never overly formal or robotic.

        Context handling:
        - Treat the conversation as continuous, not a series of isolated questions.
        - Assume familiarity unless the user signals otherwise.
        - If earlier information contained possible typos or uncertainty, only ask for clarification once, gently, and move on.

        Now respond to the user’s latest message.

        User message:
        {message}

        Conversation history (summary):
        {conversation_history_summary}

        Known user profile:
        {profile_summary}

        Latest user intro (for reference only, do not repeat verbatim):
        "{latest_intro}"

        Knowledge context:
        {knowledge}
        """

        print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

        final_message = partial_message
        if reference_lines and should_include_references(message):
            references_text = "\n\nReferences:\n" + "\n".join(reference_lines)
            final_message = partial_message + references_text
            yield final_message

        memory.save_context({"input": message}, {"output": final_message})
        global greeted_with_location
        if user_profile.get("location"):
            greeted_with_location = True

# initiate the Gradio app (lightly themed)
chatbot = gr.ChatInterface(
    fn=stream_response,
    chatbot=gr.Chatbot(height=520),
    textbox=gr.Textbox(
        placeholder="Ask anything about your PDFs (say 'reference' to see sources)...",
        container=False,
        autoscroll=True,
        scale=7,
    ),
    title="Quick RAG Chatbot",
    description="Conversational RAG over your PDFs. Keeps light context and shows references only when you ask.",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=APP_CSS,
)

# launch the Gradio app
chatbot.launch(share=True, server_name="0.0.0.0", server_port=7860)
