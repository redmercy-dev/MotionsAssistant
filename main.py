import base64
import json
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
import hmac

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
        if hmac.compare_digest(
            str(st.session_state["password"]), str(st.secrets["APP_PASSWORD"])
        ):
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

    prompt = """
**Your Role:** You are a specialized paralegal assistant focused on extracting structured **factual data** from uploaded **Bankruptcy Petition and Schedule documents (PDFs)**. Your goal is to gather the necessary information to prepare for drafting one of two specific motions in the **Southern District of Florida**: Motion to Value Secured Claim (§506) or Motion to Avoid Judicial Lien (§522(f)).

**Instructions:**
1. Analyze the uploaded PDF containing the Bankruptcy Petition and Schedules (A/B, C, D, E/F, Summary, etc.).
2. Extract **only the factual data points** listed below that are explicitly present in the document. Do not infer information not present.
3. Pay close attention to the specific schedules where information is typically found.
4. Output the results in the specified format.

**Do not draft any motion text.**

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
**📊 DATA TO EXTRACT (If Present in Schedules):**

**A. Common Case Information (Check Petition, Headers, Summary):**
* **District:** Verify if "Southern District of Florida" is present.
* **Debtor(s) Full Name(s):**
* **Case Number:** (If listed on the schedules themselves)
* **Bankruptcy Chapter:** (§7, §11, §13)
* **Debtor(s) Address:**

**B. Property & Value Information (Check Schedules A/B):**
* **Real Property:** List each property with:
  * Full Street Address
  * Description (e.g., Single-Family Home)
  * **Debtor's Stated Current Value ($)**
  * **Legal Description** (Extract *only if* explicitly provided on Schedule A)
* **Personal Property:** List relevant items (vehicles, specific valuable goods) with:
  * Description (including Year/Make/Model for vehicles)
  * **VIN** (for vehicles, if listed)
  * **Odometer Reading** (for vehicles, *only if* explicitly listed)
  * **Debtor's Stated Current Value ($)**

**C. Exemption Information (Check Schedule C):**
* For property relevant to potential lien avoidance:
  * Property Description (cross-reference with Sch A/B)
  * **Specific Exemption Statute Cited** (e.g., Fla. Const. Art. X, §4; Fla. Stat. § 222.25)
  * **Value of Claimed Exemption ($)** (or "100%" / "Unlimited")

**D. Secured Creditor & Lien Information (Check Schedule D):**
* For each secured creditor relevant to potential motions:
  * **Creditor's Full Name and Address:**
  * **Account Number:** (If listed)
  * **Description of Collateral:** (Cross-reference Sch A/B)
  * **Amount of Claim ($):** (Total claim amount listed)
  * **Unsecured Portion ($):** (If listed)
  * **Lien Details:** Extract any notes indicating lien type (e.g., "Mortgage", "PMSI", "Judgment Lien", "Second Mortgage").
  * **Identify Senior Liens:** Note if multiple liens exist on the same property listed on Sch A/B.

**E. Judgment Creditor Information (Check Schedules D or E/F):**
* For creditors potentially holding judicial liens:
  * **Creditor's Full Name and Address:**
  * **Amount of Claim ($):** (Listed as unsecured or potentially secured by judgment lien on Sch D)
  * Identify if creditor is listed as having a "Judgment".

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
**⚙️ OUTPUT FORMAT:**

1. **Start with:** EXTRACTED_FROM_UPLOAD:
2. **List Found Data:** Present each extracted data point clearly labeled. Use bullet points or a clear list format.
3. **List Missing Information:** Create a section titled INFORMATION STILL REQUIRED FOR S.D. FLA. MOTION DRAFTING:
   * Under this heading, list **only** the specific data points that were required but not found.
4. **If no relevant data at all is found** output exactly: NO_RELEVANT_INFO_FOUND_IN_UPLOAD

**DO NOT** attempt to draft the motion or ask follow-up questions yourself.
"""

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
    return resp.text.strip()


# ────────── SYSTEM INSTRUCTIONS ──────────
SYSTEM_INSTRUCTIONS = """(your long SYSTEM_INSTRUCTIONS string unchanged)"""
# NOTE: keep your existing SYSTEM_INSTRUCTIONS content here exactly as you had it.
# I’m not re-pasting it again to avoid accidental edits; paste your full block back in.


# ────────── OPENAI FILE RETRIEVAL (FIX FOR EMPTY LINKS) ──────────
def _guess_extension(filename: str, mime_type: str | None) -> str:
    if filename and "." in filename:
        return ""
    if mime_type:
        ext = mimetypes.guess_extension(mime_type)
        return ext or ""
    return ""


def openai_get_file_bytes(file_id: str) -> bytes:
    """
    Fetches bytes for a generated file (e.g., from Code Interpreter) via Files API.

    In Responses API, sandbox:/mnt/data links are not publicly reachable.
    You must download the file content using the Files API endpoint:
    GET /v1/files/{file_id}/content
    """
    # Newer openai-python commonly supports client.files.content(file_id)
    try:
        content = client.files.content(file_id)
        # Depending on SDK version, `content` may be bytes, a stream, or a response-like object.
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)
        if hasattr(content, "read"):
            return content.read()
        # Some versions return a requests-like Response with .content
        if hasattr(content, "content"):
            return content.content
        # Last resort: stringify then encode (not ideal for binaries, but avoids crashes)
        return str(content).encode("utf-8")
    except Exception:
        # Fallback for older SDKs: client.files.retrieve_content(file_id)
        # (May return str for binary in some versions; still better than nothing.)
        content2 = client.files.retrieve_content(file_id)
        if isinstance(content2, (bytes, bytearray)):
            return bytes(content2)
        return str(content2).encode("utf-8")


def extract_generated_files_from_response(final_resp) -> List[Tuple[str, bytes]]:
    """
    Extract files produced by Code Interpreter from a completed Responses API response.

    Requires using include=["code_interpreter_call.outputs"] in the create call.
    """
    out: List[Tuple[str, bytes]] = []

    if not final_resp or not getattr(final_resp, "output", None):
        return out

    for item in final_resp.output:
        # Code Interpreter tool call items are typically type == "code_interpreter_call"
        if getattr(item, "type", None) != "code_interpreter_call":
            continue

        outputs = getattr(item, "outputs", None)
        if not outputs:
            continue

        for o in outputs:
            # Look for file outputs
            o_type = getattr(o, "type", None)
            if o_type not in ("file", "output_file"):
                continue

            file_id = getattr(o, "file_id", None) or getattr(o, "id", None)
            if not file_id:
                continue

            filename = getattr(o, "filename", None) or f"download_{file_id}"
            mime_type = getattr(o, "mime_type", None)

            # Add extension if missing and we can guess
            filename = filename + _guess_extension(filename, mime_type)

            try:
                data = openai_get_file_bytes(file_id)
                out.append((filename, data))
            except Exception as e:
                # Don’t hard fail the whole run; just show the error in UI
                st.warning(f"Could not download generated file {file_id}: {e}")

    return out


# ────────── RESPONSES API STREAM (WITH FILE_SEARCH + CODE_INTERPRETER) ──────────
def stream_response_with_file_search(
    conversation_history: List[Dict],
    vector_store_ids: List[str],
    motion_context: str,
) -> Tuple[str, List[Tuple[str, bytes]]]:
    """
    Stream a response using Responses API with file_search + code_interpreter.
    Returns (response_text, downloadable_files).
    """
    input_items = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "system", "content": motion_context},
    ]
    input_items.extend(conversation_history)

    holder = st.empty()
    full_text = ""
    final_response_obj = None

    try:
        stream = client.responses.create(
            model="gpt-4o",
            input=input_items,
            tools=[
                # file_search tool with your vector store(s)
                {"type": "file_search", "vector_store_ids": vector_store_ids},
                # Enable Code Interpreter (python tool) so it can generate .docx/.pdf/etc
                {
                    "type": "code_interpreter",
                    # Optional container config; auto is simplest.
                    "container": {"type": "auto"},
                },
            ],
            # IMPORTANT: include code interpreter outputs so we can see file_id(s)
            include=["code_interpreter_call.outputs"],
            stream=True,
        )

        for event in stream:
            etype = getattr(event, "type", None)

            # Stream text deltas
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    full_text += delta
                    holder.markdown(full_text)

            # Capture completed response so we can fetch generated files
            if etype == "response.completed":
                final_response_obj = getattr(event, "response", None)

        # After streaming ends, extract tool-generated files (if any)
        generated_files = extract_generated_files_from_response(final_response_obj)

        return full_text.strip(), generated_files

    except Exception as e:
        st.error(f"Error creating response: {str(e)}")
        return "", []


# ────────── STREAMLIT UI STYLES ──────────
st.set_page_config("Legal Motion Assistant", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="st-"] { font-family: 'Georgia', serif; color: #333; }
    body {background-color: #f0f2f6;}
    h1, h2, h3 {color: #0d1b4c; font-weight: bold;}
    .block-container {
        background-color: #ffffff; border-radius: 10px; padding: 2rem 3rem 3rem 3rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); max-width: 1200px; margin: 1rem auto;
    }
    [data-testid="stSidebar"] { background-color: #e1e5f0; padding-top: 1.5rem; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] button p { color: #0d1b4c; }
    [data-testid="stFileUploader"] button {
        padding: 6px 12px; font-size: 14px; border: 1px solid #198754;
        background-color: #198754; color: white; border-radius: 6px;
    }
    [data-testid="stFileUploader"] button:hover { background-color: #157347; }
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important; line-height: 1.6 !important; padding: 12px 15px !important;
        border-radius: 8px !important; border: 1px solid #ccc; background-color: #f8f9fa;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #0d1b4c; box-shadow: 0 0 0 2px rgba(13, 27, 76, 0.2);
    }
    .stChatMessage { border-radius: 10px; padding: 1rem 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stChatMessageContent"] { background-color: transparent; }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #e1e5f0 !important; border-left: 4px solid #0d1b4c;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #d1e7dd !important; border-right: 4px solid #198754;
    }
    [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {
        background-color: transparent !important;
    }
    .stButton button {
        background-color: #198754; color: white; border: none; border-radius: 6px;
        padding: 0.6rem 1.2rem; font-weight: bold;
    }
    .stButton button:hover:not(:disabled) { background-color: #157347; }
    .stButton button:disabled { background-color: #cccccc; color: #888888; }
    .stDownloadButton button {
        background-color: #5c6ac4; color: white; border: none; border-radius: 5px;
        padding: 0.3rem 0.8rem; font-size: 14px; margin-top: 5px; margin-right: 5px;
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
            mlabel = st.selectbox(
                "Destination motion type", list(MOTION_OPTIONS.values())
            )
            slug = next(k for k, v in MOTION_OPTIONS.items() if v == mlabel)
            juris = st.text_input("Jurisdiction")
            files_ = st.file_uploader(
                "PDF files", type="pdf", accept_multiple_files=True
            )
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
                                file_id=file_obj.id,
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


# =====================================================================
# CHAT
# =====================================================================
if page == "Chat":
    st.title("⚖️ Legal Motion Assistant")

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
    chapter = st.sidebar.selectbox(
        "Bankruptcy Chapter (optional)", ["", "7", "11", "13"]
    )

    if slug is None:
        st.sidebar.error("Please select a motion type to enable chat.")
        st.stop()

    if slug not in cfg["vector_stores"]:
        st.error(
            "Vector store not found for this motion type. Please create it in Admin."
        )
        st.stop()

    # ───── DISPLAY HISTORY ─────
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])
            for fn, blob in h.get("files", []):
                st.download_button(
                    f"Download {fn}", blob, fn, key=uk("dl_hist")
                )

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
        extract_blocks, blobs_for_history = [], []

        if uploaded:
            prog = st.progress(0.0)
            for i, uf in enumerate(uploaded, 1):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_{uf.name}"
                ) as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name

                with st.spinner(f"Gemini reading {uf.name} …"):
                    gem_text = gem_extract(tmp_path, user_prompt)
                    if gem_text and gem_text != "NO_RELEVANT_INFO_FOUND_IN_UPLOAD":
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
                st.download_button(
                    f"Download {fn}", blob, fn, key=uk("dl_user")
                )

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
        conversation_history = [
            {"role": h["role"], "content": h["content"]}
            for h in st.session_state.history
        ]

        # Get response from API
        with st.chat_message("assistant"):
            answer, new_files = stream_response_with_file_search(
                conversation_history,
                [cfg["vector_stores"][slug]],
                motion_context,
            )

            # Important: Use download buttons backed by bytes (no sandbox links)
            if new_files:
                st.markdown("### Downloads")
                for fn, data in new_files:
                    st.download_button(
                        f"Download {fn}",
                        data,
                        fn,
                        key=uk("dl_asst"),
                    )

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "files": new_files}
        )

        # Reset uploader key → clears file-picker
        st.session_state.uploader_key = uk("chat_files")
        st.rerun()
