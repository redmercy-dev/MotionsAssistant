import json, mimetypes, os, tempfile, uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import hmac
import requests
import pandas as pd
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
def check_password():
    """
    Returns True if the user enters the correct password stored in Streamlit secrets.
    """
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(str(st.session_state["password"]), str(st.secrets["APP_PASSWORD"])):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
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
    gfile = gem_upload(path)

    # (Shortened per your instruction)
    prompt = """Your Role: You are a specialized paralegal assistant focused on extracting structured factual data from uploaded Bankruptcy Petition and Schedule documents (PDFs)."""

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
# (Shortened per your instruction)
SYSTEM_INSTRUCTIONS = """Bankruptcy Motion Drafting (Southern District of Florida Focus)."""

# ────────── OpenAI Container File Download Helpers ──────────
def download_container_file(container_id: str, file_id: str, filename: str) -> Tuple[str, bytes]:
    """
    Download a file from a code interpreter container.
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

# ────────── Robust helpers for SDK objects/dicts ──────────
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _content_to_text(content_obj: Any) -> str:
    """
    Convert a 'content' field from file_search results into displayable text.
    Handles shapes like:
      - {"type":"text","text":"..."}
      - [{"type":"text","text":"..."}, ...]
      - {"text":"..."} or "..."
    """
    if content_obj is None:
        return ""
    if isinstance(content_obj, str):
        return content_obj.strip()

    if isinstance(content_obj, dict):
        # common: {"type":"text","text":"..."}
        t = _get(content_obj, "text")
        if isinstance(t, str) and t.strip():
            return t.strip()
        # sometimes nested
        parts = _get(content_obj, "content") or _get(content_obj, "parts")
        return _content_to_text(parts)

    if isinstance(content_obj, list):
        texts = []
        for item in content_obj:
            if isinstance(item, str):
                if item.strip():
                    texts.append(item.strip())
                continue
            if isinstance(item, dict):
                t = item.get("text") or item.get("content")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
                else:
                    # e.g. {"type":"text","text": "..."}
                    maybe = item.get("text")
                    if isinstance(maybe, str) and maybe.strip():
                        texts.append(maybe.strip())
        return "\n\n".join([t for t in texts if t]).strip()

    return str(content_obj).strip()

# ────────── NEW: Extract container file outputs + retrieved chunks ──────────
def extract_container_files_and_chunks(response_obj) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
      - container_files: from 'container_file_citation' annotations (code_interpreter outputs)
      - chunks: from file_search_call.results (actual retrieved passages)
    """
    container_files: List[Dict] = []
    chunks: List[Dict] = []

    output_items = _as_list(_get(response_obj, "output"))

    # 1) Container files (from message annotations)
    for output_item in output_items:
        content_list = _as_list(_get(output_item, "content"))
        for content in content_list:
            annotations = _as_list(_get(content, "annotations"))
            for ann in annotations:
                if _get(ann, "type") == "container_file_citation":
                    container_files.append(
                        {
                            "container_id": _get(ann, "container_id"),
                            "file_id": _get(ann, "file_id"),
                            "filename": _get(ann, "filename"),
                        }
                    )

    # 2) Retrieved chunks (from file_search_call.results) — requires include=["file_search_call.results"]
    for output_item in output_items:
        if _get(output_item, "type") != "file_search_call":
            continue

        # The docs call it "results" when included. Some SDKs may name it "search_results".
        results = _get(output_item, "results")
        if results is None:
            results = _get(output_item, "search_results")

        for r in _as_list(results):
            file_id = _get(r, "file_id") or _get(r, "file")
            filename = _get(r, "filename")

            # Best-effort filename lookup if missing
            if not filename and file_id:
                try:
                    fobj = client.files.retrieve(file_id)
                    filename = getattr(fobj, "filename", None) or "Unknown file"
                except Exception:
                    filename = "Unknown file"

            # Extract the chunk text
            text = (
                _get(r, "text")
                or _get(r, "chunk")
                or _content_to_text(_get(r, "content"))
                or _content_to_text(_get(r, "document"))
            )

            chunk = {
                "file_id": file_id,
                "filename": filename or "Unknown file",
                "text": text or "",
                "score": _get(r, "score"),
                "rank": _get(r, "rank"),
            }
            # Only keep meaningful entries
            if chunk["text"].strip():
                chunks.append(chunk)

    return container_files, chunks

def _retrieve_response_with_include(response_id: str):
    """
    Some SDK versions support include=... on retrieve; others don't.
    Try with include first, then fall back.
    """
    try:
        return client.responses.retrieve(response_id, include=["file_search_call.results"])
    except TypeError:
        return client.responses.retrieve(response_id)
    except Exception:
        # fallback anyway
        return client.responses.retrieve(response_id)

def stream_response_with_file_search(
    conversation_history: List[Dict],
    vector_store_ids: List[str],
    motion_context: str
) -> Tuple[str, List[Tuple[str, bytes]], List[Dict]]:
    """
    Stream a response using Responses API with file_search and code_interpreter tools.

    Returns:
      (response_text, list_of_downloads, retrieved_chunks)
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
            # ✅ This is the key to get the actual retrieved chunks back
            include=["file_search_call.results"],
            stream=True,
        )

        holder = st.empty()
        full_text = ""
        response_id: Optional[str] = None

        for event in stream_response:
            etype = getattr(event, "type", None)
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    full_text += delta
                    holder.markdown(full_text)
            elif etype == "response.created":
                resp = getattr(event, "response", None)
                rid = getattr(resp, "id", None) if resp else None
                if rid:
                    response_id = rid

        files_to_download: List[Tuple[str, bytes]] = []
        retrieved_chunks: List[Dict] = []

        if response_id:
            try:
                complete_response = _retrieve_response_with_include(response_id)
                container_files, retrieved_chunks = extract_container_files_and_chunks(complete_response)

                for ann in container_files:
                    filename, file_bytes = download_container_file(
                        ann["container_id"],
                        ann["file_id"],
                        ann["filename"],
                    )
                    if file_bytes:
                        files_to_download.append((filename, file_bytes))

            except Exception as e:
                st.warning(f"Could not retrieve files/chunks from response: {str(e)}")

        return full_text, files_to_download, retrieved_chunks

    except Exception as e:
        st.error(f"Error creating response: {str(e)}")
        return "", [], []

# ────────── STREAMLIT UI STYLES ──────────
st.set_page_config("Legal Motion Assistant", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="st-"] { font-family: 'Georgia', serif; color: #333; }
    body { background-color: #f0f2f6; }
    h1, h2, h3 { color: #0d1b4c; font-weight: bold; }

    .block-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem 3rem 3rem 3rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        max-width: 1200px;
        margin: 1rem auto;
    }

    [data-testid="stSidebar"] { background-color: #e1e5f0; padding-top: 1.5rem; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] button p { color: #0d1b4c; }

    [data-testid="stFileUploader"] button {
        padding: 6px 12px;
        font-size: 14px;
        border: 1px solid #198754;
        background-color: #198754;
        color: white;
        border-radius: 6px;
    }
    [data-testid="stFileUploader"] button:hover { background-color: #157347; }

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

    .stChatMessage {
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #e1e5f0 !important;
        border-left: 4px solid #0d1b4c;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #d1e7dd !important;
        border-right: 4px solid #198754;
    }

    .stButton button {
        background-color: #198754;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }
    .stButton button:hover:not(:disabled) { background-color: #157347; }
    .stButton button:disabled { background-color: #cccccc; color: #888888; }

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
    .stDownloadButton button:hover:not(:disabled) { background-color: #4553a0; }
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

# =====================================================================
# ADMIN
# =====================================================================
if page == "Admin":
    st.title("⚙️ Admin panel")

    # ➊ Create two vector stores
    if len(cfg["vector_stores"]) < 2 and st.button("Create 2 vector stores"):
        for slug in MOTION_OPTIONS:
            cfg["vector_stores"][slug] = client.vector_stores.create(
                name=f"{slug}_store"
            ).id
        save_cfg(cfg)
        st.success("Vector stores created.")

    # ➋ Show vector stores
    if cfg["vector_stores"]:
        st.subheader("Vector stores")
        for slug, vsid in cfg["vector_stores"].items():
            st.markdown(f"* **{MOTION_OPTIONS.get(slug, slug)}** → {vsid}")

    # ➌ Upload PDFs into a vector store
    if cfg["vector_stores"]:
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
                        cfg["vector_stores"][slug] = client.vector_stores.create(
                            name=f"{slug}_store"
                        ).id
                        save_cfg(cfg)

                    with st.spinner("Uploading & indexing …"):
                        for f in files_:
                            file_obj = client.files.create(file=f, purpose="assistants")
                            client.vector_stores.files.create(
                                vector_store_id=cfg["vector_stores"][slug],
                                file_id=file_obj.id
                            )
                    st.success("Files uploaded & indexed.")

    # ➍ Display indexed PDFs
    if cfg["vector_stores"]:
        rows = []
        for slug, vsid in cfg["vector_stores"].items():
            try:
                vs_files = client.vector_stores.files.list(vsid, limit=100)
                for vf in vs_files:
                    try:
                        file_obj = client.files.retrieve(vf.id)
                        fname = file_obj.filename
                    except Exception:
                        fname = "(unknown)"

                    rows.append({
                        "Motion": MOTION_OPTIONS.get(slug, slug),
                        "Filename": fname,
                        "Jurisdiction": "N/A",
                    })
            except Exception as e:
                st.warning(f"Could not list files for {slug}: {e}")

        if rows:
            st.dataframe(pd.DataFrame(rows))

# =====================================================================
# CHAT
# =====================================================================
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

    if slug not in cfg["vector_stores"]:
        st.error("Vector store not found for this motion type. Please create it in Admin.")
        st.stop()

    # ───── DISPLAY HISTORY ─────
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])

            for fn, blob in h.get("files", []):
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_hist"))

            # ✅ Show retrieved chunks (from file_search_call.results)
            chunks = h.get("citations", [])
            if chunks and h["role"] == "assistant":
                with st.expander(f"📚 View {len(chunks)} retrieved reference chunk(s)", expanded=False):
                    for idx, c in enumerate(chunks, 1):
                        fname = c.get("filename", "Unknown file")
                        score = c.get("score")
                        header = f"**Chunk {idx}** (from {fname})"
                        if score is not None:
                            header += f" — score: `{score}`"
                        st.markdown(header)

                        txt = c.get("text", "")
                        if txt:
                            st.markdown(f"> {format_citation_text(txt, max_length=800)}")
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
            label_visibility="collapsed"
        )

    with col_inp:
        user_prompt = st.chat_input("Ask or continue …")

    # ───── Handle new turn ─────
    if user_prompt:
        extract_blocks, blobs_for_history = [], []

        if uploaded:
            prog = st.progress(0.0)
            for i, uf in enumerate(uploaded, 1):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name

                with st.spinner(f"Gemini reading {uf.name} …"):
                    gem_text = gem_extract(tmp_path, user_prompt)
                    if gem_text and gem_text not in ("NO_RELEVANT_INFO", "NO_RELEVANT_INFO_FOUND_IN_UPLOAD"):
                        extract_blocks.append(
                            f"EXTRACTED_FROM_UPLOAD File name ({uf.name}):\n{gem_text}"
                        )
                    blobs_for_history.append((uf.name, uf.getvalue()))

                prog.progress(i / len(uploaded))
            prog.empty()

            if any(f.name.lower().endswith(".pdf") for f in uploaded):
                st.session_state.schedule_uploaded = True

        # Store user turn
        st.session_state.history.append(
            {"role": "user", "content": user_prompt, "files": blobs_for_history}
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
            answer, new_files, chunks = stream_response_with_file_search(
                conversation_history,
                [cfg["vector_stores"][slug]],
                motion_context
            )

            for fn, data in new_files:
                st.download_button(f"Download {fn}", data, fn, key=uk("dl_asst"))

            # ✅ Display retrieved chunks immediately
            if chunks:
                with st.expander(f"📚 View {len(chunks)} retrieved reference chunk(s)", expanded=False):
                    for idx, c in enumerate(chunks, 1):
                        fname = c.get("filename", "Unknown file")
                        score = c.get("score")
                        header = f"**Chunk {idx}** (from {fname})"
                        if score is not None:
                            header += f" — score: `{score}`"
                        st.markdown(header)

                        txt = c.get("text", "")
                        if txt:
                            st.markdown(f"> {format_citation_text(txt, max_length=800)}")
                        st.divider()

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "files": new_files, "citations": chunks}
        )

        # Reset uploader key → clears file-picker
        st.session_state.uploader_key = uk("chat_files")
        st.rerun()
