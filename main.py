import json
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
import hmac

import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types as gtypes
from openai import OpenAI

# ────────── STATIC CONFIG ──────────
CFG_PATH = "config2.json"
MOTION_OPTIONS = {
    "value_claim": "Motion to Value Secured Claim",
    "avoid_lien": "Motion to Avoid Judicial Lien",
}

# ────────── STREAMLIT HELPERS ──────────
def uk(prefix: str = "k") -> str:
    """Return a unique Streamlit widget key."""
    return f"{prefix}_{uuid.uuid4().hex}"

def format_citation_text(text: str, max_length: int = 200) -> str:
    """Format citation text for display, truncating if needed."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

# ────────── PERSISTED CFG ──────────
def load_cfg() -> Dict:
    if not Path(CFG_PATH).exists():
        return {"vector_stores": {}}
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("vector_stores", {})
    return cfg

def save_cfg(c: Dict):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)

# ────────── CONFIG & API KEYS ──────────
cfg = load_cfg()
openai_api_key = st.secrets["api_keys"]["openai_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]

# ────────── OPENAI CLIENT (Responses API) ──────────
client = OpenAI(api_key=openai_api_key)

# ────────── GOOGLE GEMINI CLIENT ──────────
g_client = genai.Client(api_key=gemini_api_key)
GEM_MODEL = "gemini-2.0-flash-exp"

# ────────── Password ──────────
def check_password() -> bool:
    """
    Returns True if the user enters the correct password stored in Streamlit secrets.
    """
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(
            str(st.session_state.get("password", "")),
            str(st.secrets.get("APP_PASSWORD", "")),
        ):
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state.get("password_correct", False):
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

# ────────── Gemini Functions ──────────
def get_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "application/octet-stream"

def gem_upload(path: str) -> gtypes.File:
    """Upload file to Gemini and return File object."""
    return g_client.files.upload(path=path)

def gem_extract(path: str, user_prompt: str) -> str:
    """
    Uploads path to Gemini and asks it to extract facts needed for drafting bankruptcy motions.
    """
    _ = user_prompt  # kept for future use
    gfile = gem_upload(path)

    # (Prompt shortened per your instruction)
    prompt = "You are a specialized paralegal assistant focused on extracting structured factual data from uploaded Bankruptcy Petition and Schedule documents (PDFs)."

    contents = [
        gtypes.Content(
            role="user",
            parts=[
                gtypes.Part.from_text(text=prompt),
                gtypes.Part.from_uri(file_uri=gfile.uri, mime_type=gfile.mime_type),
            ],
        ),
    ]
    resp = g_client.models.generate_content(
        model=GEM_MODEL,
        contents=contents,
    )
    return (resp.text or "").strip()

# ────────── SYSTEM INSTRUCTIONS ──────────
# (Prompt shortened per your instruction)
SYSTEM_INSTRUCTIONS = "You are a Bankruptcy Motion Drafting Assistant specializing in the Southern District of Florida (S.D. Fla.)."

def download_container_file(container_id: str, file_id: str, filename: str) -> Tuple[str, bytes]:
    """
    Download a file from a code interpreter container.

    Args:
        container_id: Container ID (e.g., "cntr_...")
        file_id: File ID (e.g., "cfile_...")
        filename: Original filename

    Returns:
        Tuple of (filename, file_bytes)
    """
    try:
        url = f"https://api.openai.com/v1/containers/{container_id}/files/{file_id}/content"
        headers = {"Authorization": f"Bearer {openai_api_key}"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return (filename, response.content)
    except Exception as e:
        st.warning(f"Could not download file {filename}: {str(e)}")
        return (filename, b"")

def extract_annotations_from_response(response_obj) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract file annotations and citations from a response object.

    Returns tuple of (file_annotations, citations)
    where citations contain file_citation type with text/quote.
    """
    file_annotations: List[Dict] = []
    citations: List[Dict] = []

    outputs = getattr(response_obj, "output", None) or []
    for output_item in outputs:
        contents = getattr(output_item, "content", None) or []
        for content in contents:
            anns = getattr(content, "annotations", None) or []
            for ann in anns:
                ann_type = getattr(ann, "type", None)
                if not ann_type:
                    continue

                if ann_type == "container_file_citation":
                    file_annotations.append(
                        {
                            "container_id": getattr(ann, "container_id", None),
                            "file_id": getattr(ann, "file_id", None),
                            "filename": getattr(ann, "filename", None),
                        }
                    )

                elif ann_type == "file_citation":
                    citation_data = {
                        "file_id": getattr(ann, "file_id", None),
                        "quote": getattr(ann, "quote", None),
                        "text": getattr(ann, "text", None),
                    }

                    # Try to get filename from file_id (best-effort)
                    try:
                        if citation_data["file_id"]:
                            file_obj = client.files.retrieve(citation_data["file_id"])
                            citation_data["filename"] = getattr(file_obj, "filename", None) or "Unknown file"
                        else:
                            citation_data["filename"] = "Unknown file"
                    except Exception:
                        citation_data["filename"] = "Unknown file"

                    citations.append(citation_data)

    # Drop malformed container citations (must have ids)
    file_annotations = [
        fa for fa in file_annotations
        if fa.get("container_id") and fa.get("file_id") and fa.get("filename")
    ]

    return file_annotations, citations

def stream_response_with_file_search(
    conversation_history: List[Dict],
    vector_store_ids: List[str],
    motion_context: str,
) -> Tuple[str, List[Tuple[str, bytes]], List[Dict]]:
    """
    Stream a response using Responses API with file_search and code_interpreter tools.

    Args:
        conversation_history: List of message items (user/assistant)
        vector_store_ids: List of vector store IDs to search
        motion_context: Context about motion type, jurisdiction, chapter

    Returns:
        Tuple of (response_text, list of (filename, bytes) for downloads, list of citations)
    """
    input_items = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "system", "content": motion_context},
    ]
    input_items.extend(conversation_history)

    try:
        stream_response = client.responses.create(
            model="gpt-4o",
            input=input_items,
            tools=[
                {"type": "file_search", "vector_store_ids": vector_store_ids},
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
            stream=True,
        )

        holder = st.empty()
        full_text = ""
        response_id = None

        for event in stream_response:
            ev_type = getattr(event, "type", None)
            if ev_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    full_text += delta
                    holder.markdown(full_text)
            elif ev_type == "response.created":
                resp = getattr(event, "response", None)
                response_id = getattr(resp, "id", None) or response_id

        files_to_download: List[Tuple[str, bytes]] = []
        citations: List[Dict] = []

        if response_id:
            try:
                complete_response = client.responses.retrieve(response_id)
                file_annotations, citations = extract_annotations_from_response(complete_response)

                for ann in file_annotations:
                    filename, file_bytes = download_container_file(
                        ann["container_id"],
                        ann["file_id"],
                        ann["filename"],
                    )
                    if file_bytes:
                        files_to_download.append((filename, file_bytes))
            except Exception as e:
                st.warning(f"Could not retrieve files/citations from response: {str(e)}")

        return full_text, files_to_download, citations

    except Exception as e:
        st.error(f"Error creating response: {str(e)}")
        return "", [], []

# ────────── STREAMLIT UI STYLES ──────────
st.set_page_config("Legal Motion Assistant", layout="wide")
st.markdown(
    """
    <style>
    /* --- Base & Fonts --- */
    html, body, [class*="st-"] {
        font-family: 'Georgia', serif;
        color: #333;
    }
    body {background-color: #f0f2f6;}
    h1, h2, h3 {color: #0d1b4c; font-weight: bold;}

    /* --- Main Container --- */
    .block-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem 3rem 3rem 3rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        max-width: 1200px;
        margin: 1rem auto;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #e1e5f0;
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] button p {
        color: #0d1b4c;
    }
    [data-testid="stFileUploader"] button {
        padding: 6px 12px;
        font-size: 14px;
        border: 1px solid #198754;
        background-color: #198754;
        color: white;
        border-radius: 6px;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #157347;
    }

    /* --- Chat Interface --- */
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important;
        line-height: 1.6 !important;
        padding: 12px 15px !important;
        border-radius: 8px !important;
        border: 1px solid #ccc;
        background-color: #f8f9fa;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #0d1b4c;
        box-shadow: 0 0 0 2px rgba(13, 27, 76, 0.2);
    }

    /* Chat Message Styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Assistant messages */
    [data-testid="stChatMessageContent"] {
        background-color: transparent;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #e1e5f0 !important;
        border-left: 4px solid #0d1b4c;
    }

    /* User messages */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #d1e7dd !important;
        border-right: 4px solid #198754;
    }

    /* Avatar styling */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: transparent !important;
    }

    /* --- Buttons & Inputs --- */
    .stButton button {
        background-color: #198754;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }
    .stButton button:hover:not(:disabled) {background-color: #157347;}
    .stButton button:disabled {background-color: #cccccc; color: #888888;}
    .stDownloadButton button {
        background-color: #5c6ac4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.3rem 0.8rem;
        font-size: 14px;
        margin-top: 5px;
        margin-right: 5px;
    }
    .stDownloadButton button:hover:not(:disabled) {background-color: #4553a0;}
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Page", ("Chat", "Admin"))

# ────────── RESET BUTTON ──────────
if st.sidebar.button("🔄 Reset workspace"):
    Path(CFG_PATH).unlink(missing_ok=True)
    cfg = {"vector_stores": {}}
    st.session_state.schedule_uploaded = False
    st.sidebar.success("Workspace cleared – open *Admin* to start fresh.")
    st.rerun()

# ────────── CLEAR CHAT BUTTON ──────────
if page == "Chat" and st.sidebar.button("🗑️ Clear chat"):
    st.session_state.history = []
    st.session_state.schedule_uploaded = False
    st.sidebar.success("Chat history cleared.")
    st.rerun()

# ────────── SESSION INIT ──────────
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "schedule_uploaded" not in st.session_state:
    st.session_state.schedule_uploaded = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = uk("chat_files")

# ============================================================================
# ADMIN
# ============================================================================
if page == "Admin":
    st.title("⚙️ Admin panel")

    # ➊ Create two vector stores
    if len(cfg.get("vector_stores", {})) < 2 and st.button("Create 2 vector stores"):
        cfg.setdefault("vector_stores", {})
        for slug in MOTION_OPTIONS:
            cfg["vector_stores"][slug] = client.vector_stores.create(name=f"{slug}_store").id
        save_cfg(cfg)
        st.success("Vector stores created.")

    # ➋ Show vector stores
    if cfg.get("vector_stores"):
        st.subheader("Vector stores")
        for slug, vsid in cfg["vector_stores"].items():
            st.markdown(f"* **{MOTION_OPTIONS.get(slug, slug)}** → {vsid}")

    # ➌ Upload PDFs into a vector store
    if cfg.get("vector_stores"):
        st.subheader("Upload PDF knowledge")
        with st.form("upload_form", clear_on_submit=True):
            mlabel = st.selectbox("Destination motion type", list(MOTION_OPTIONS.values()))
            slug = next(k for k, v in MOTION_OPTIONS.items() if v == mlabel)
            juris = st.text_input("Jurisdiction")
            files_ = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
            submitted = st.form_submit_button("Upload")

            if submitted:
                if not (juris and files_):
                    st.error("Provide jurisdiction & select PDF(s).")
                else:
                    if slug not in cfg["vector_stores"]:
                        cfg["vector_stores"][slug] = client.vector_stores.create(name=f"{slug}_store").id
                        save_cfg(cfg)

                    with st.spinner("Uploading & indexing …"):
                        for f in files_:
                            file_obj = client.files.create(file=f, purpose="assistants")
                            client.vector_stores.files.create(
                                vector_store_id=cfg["vector_stores"][slug],
                                file_id=file_obj.id,
                            )
                    st.success("Files uploaded & indexed.")

    # ➍ Display indexed PDFs
    if cfg.get("vector_stores"):
        rows = []
        for slug, vsid in cfg["vector_stores"].items():
            try:
                vs_files = client.vector_stores.files.list(vector_store_id=vsid, limit=100)
                items = getattr(vs_files, "data", None) or vs_files

                for vf in items:
                    # vf is a VectorStoreFile; its file id is usually vf.file_id
                    file_id = getattr(vf, "file_id", None) or getattr(vf, "id", None)
                    fname = "(unknown)"
                    if file_id:
                        try:
                            file_obj = client.files.retrieve(file_id)
                            fname = getattr(file_obj, "filename", None) or fname
                        except Exception:
                            pass

                    rows.append(
                        {
                            "Motion": MOTION_OPTIONS.get(slug, slug),
                            "Filename": fname,
                            "Jurisdiction": "N/A",
                        }
                    )
            except Exception as e:
                st.warning(f"Could not list files for {slug}: {e}")

        if rows:
            st.dataframe(pd.DataFrame(rows))

# ============================================================================
# CHAT
# ============================================================================
if page == "Chat":
    st.title("⚖️ Legal Motion Assistant")

    # ───── Mandatory selections ─────
    motion_label = st.sidebar.selectbox(
        "Motion type (required)",
        ["— Select —"] + list(MOTION_OPTIONS.values()),
        key="motion_type",
    )
    slug = (
        next((k for k, v in MOTION_OPTIONS.items() if v == motion_label), None)
        if motion_label != "— Select —"
        else None
    )

    juris = st.sidebar.text_input("Jurisdiction (optional)")
    chapter = st.sidebar.selectbox("Bankruptcy Chapter (optional)", ["", "7", "11", "13"])

    if slug is None:
        st.sidebar.error("Please select a motion type to enable chat.")
        st.stop()

    if slug not in cfg.get("vector_stores", {}):
        st.error("Vector store not found for this motion type. Please create it in Admin.")
        st.stop()

    # ───── DISPLAY HISTORY ─────
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])
            for fn, blob in h.get("files", []):
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_hist"))

            # Show citations if available
            citations = h.get("citations", [])
            if citations and h["role"] == "assistant":
                with st.expander(f"📚 View {len(citations)} reference chunk(s)", expanded=False):
                    for idx, citation in enumerate(citations, 1):
                        st.markdown(f"**Reference {idx}** (from {citation.get('filename', 'Unknown')})")
                        quote_text = citation.get("quote") or citation.get("text", "")
                        if quote_text:
                            st.markdown(f"> {format_citation_text(quote_text)}")
                        st.divider()

    st.markdown("<div style='padding-bottom:70px'></div>", unsafe_allow_html=True)

    # ───── Upload + Chat input widgets ─────
    col_inp, col_up = st.columns([5, 2])
    with col_up:
        uploaded = st.file_uploader(
            "💎",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=st.session_state.uploader_key,
            label_visibility="collapsed",
        )

    with col_inp:
        user_prompt = st.chat_input("Ask or continue …")

    # ───── Handle new turn ─────
    if user_prompt:
        extract_blocks: List[str] = []
        blobs_for_history: List[Tuple[str, bytes]] = []

        if uploaded:
            prog = st.progress(0.0)
            for i, uf in enumerate(uploaded, 1):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name

                with st.spinner(f"Gemini reading {uf.name} …"):
                    gem_text = gem_extract(tmp_path, user_prompt)

                    # accept both older/newer "no info" strings
                    if gem_text and gem_text not in {"NO_RELEVANT_INFO", "NO_RELEVANT_INFO_FOUND_IN_UPLOAD"}:
                        extract_blocks.append(f"EXTRACTED_FROM_UPLOAD File name ({uf.name}):\n{gem_text}")

                    blobs_for_history.append((uf.name, uf.getvalue()))

                prog.progress(i / len(uploaded))

                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            prog.empty()

            if any(getattr(f, "name", "").lower().endswith(".pdf") for f in uploaded):
                st.session_state.schedule_uploaded = True

        # Store user turn
        st.session_state.history.append(
            {"role": "user", "content": user_prompt, "files": blobs_for_history, "citations": []}
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)
            for fn, blob in blobs_for_history:
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_user"))

        # Build motion context
        context_parts = [
            f"Motion type: {motion_label}",
            f"Jurisdiction: {juris or '(unspecified)'}",
            f"Chapter: {chapter or '(unspecified)'}",
        ]
        if extract_blocks:
            context_parts.append("\n".join(extract_blocks))

        motion_context = "\n".join(context_parts)

        # Prepare conversation for API
        conversation_history = [{"role": h["role"], "content": h["content"]} for h in st.session_state.history]

        # Get response from API
        with st.chat_message("assistant"):
            answer, new_files, citations = stream_response_with_file_search(
                conversation_history,
                [cfg["vector_stores"][slug]],
                motion_context,
            )

            for fn, data in new_files:
                st.download_button(f"Download {fn}", data, fn, key=uk("dl_asst"))

            # Display citations if available
            if citations:
                with st.expander(f"📚 View {len(citations)} reference chunk(s)", expanded=False):
                    for idx, citation in enumerate(citations, 1):
                        st.markdown(f"**Reference {idx}** (from {citation.get('filename', 'Unknown')})")
                        quote_text = citation.get("quote") or citation.get("text", "")
                        if quote_text:
                            st.markdown(f"> {format_citation_text(quote_text)}")
                        st.divider()

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "files": new_files, "citations": citations}
        )

        # Reset uploader key → clears file-picker
        st.session_state.uploader_key = uk("chat_files")
        st.rerun()
