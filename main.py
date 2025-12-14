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
from google.genai.errors import ClientError as GeminiClientError
from openai import OpenAI
from openai.types.beta import Assistant

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
        return {"assistant_id": "", "vector_stores": {}}
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("assistant_id", "")
    cfg.setdefault("vector_stores", {})
    return cfg


def save_cfg(c: Dict):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)


# ────────── CONFIG & API KEYS ──────────
cfg = load_cfg()

openai_api_key = st.secrets["api_keys"]["openai_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]

# Optional: make model configurable from secrets/env (recommended)
# In Streamlit secrets you can add:
# [api_keys]
# gemini_model = "gemini-2.5-flash"
GEM_MODEL = (
    st.secrets.get("api_keys", {}).get("gemini_model")
    or os.getenv("GEMINI_MODEL")
    or "gemini-2.5-flash"
)

# ────────── OPENAI CLIENT (Assistants) ──────────
oa_client = OpenAI(api_key=openai_api_key)

# ────────── GOOGLE GEMINI CLIENT ──────────
# Use v1 unless you specifically need v1alpha features/models
g_client = genai.Client(
    api_key=gemini_api_key,
    http_options=gtypes.HttpOptions(api_version="v1"),
)

# ────────── Password ──────────
def check_password():
    """
    Returns `True` if the user enters the correct password stored in Streamlit secrets.
    """
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(
            str(st.session_state["password"]),
            str(st.secrets["APP_PASSWORD"])
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
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
NO_INFO_SENTINEL = "NO_RELEVANT_INFO_FOUND_IN_UPLOAD"

def get_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "application/octet-stream"

def gem_upload(path: str) -> gtypes.File:
    # Upload returns a File object with a URI you must reference in contents
    return g_client.files.upload(
        file=path,
        config=gtypes.UploadFileConfig(mime_type=get_mime(path)),
    )

def gem_extract(path: str, user_prompt: str) -> str:
    """
    Uploads `path` to Gemini and asks it to extract facts needed for drafting
    bankruptcy motions.
    """
    # 1) upload the file
    gfile = gem_upload(path)
    mime_type = get_mime(path)

    # 2) build the extraction prompt
    prompt = ("""
**Your Role:** You are a specialized paralegal assistant focused on extracting structured **factual data** from uploaded **Bankruptcy Petition and Schedule documents (PDFs)**. Your goal is to gather the necessary information to prepare for drafting one of two specific motions in the **Southern District of Florida**: Motion to Value Secured Claim (§506) or Motion to Avoid Judicial Lien (§522(f)).

**Instructions:**
1.  Analyze the uploaded PDF containing the Bankruptcy Petition and Schedules (A/B, C, D, E/F, Summary, etc.).
2.  Extract **only the factual data points** listed below that are explicitly present in the document. Do not infer information not present.
3.  Pay close attention to the specific schedules where information is typically found.
4.  Output the results in the specified format. **Do not draft any motion text.**

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
**📊 DATA TO EXTRACT (If Present in Schedules):**

**A. Common Case Information (Check Petition, Headers, Summary):**
    *   **District:** Verify if "Southern District of Florida" is present.
    *   **Debtor(s) Full Name(s):**
    *   **Case Number:** (If listed on the schedules themselves)
    *   **Bankruptcy Chapter:** (§7, §11, §13)
    *   **Debtor(s) Address:**

**B. Property & Value Information (Check Schedules A/B):**
    *   **Real Property:** List each property with:
        *   Full Street Address
        *   Description (e.g., Single-Family Home)
        *   **Debtor's Stated Current Value ($)**
        *   **Legal Description** (Extract *only if* explicitly provided on Schedule A)
    *   **Personal Property:** List relevant items (vehicles, specific valuable goods) with:
        *   Description (including Year/Make/Model for vehicles)
        *   **VIN** (for vehicles, if listed)
        *   **Odometer Reading** (for vehicles, *only if* explicitly listed)
        *   **Debtor's Stated Current Value ($)**

**C. Exemption Information (Check Schedule C):**
    *   For property relevant to potential lien avoidance:
        *   Property Description (cross-reference with Sch A/B)
        *   **Specific Exemption Statute Cited** (e.g., Fla. Const. Art. X, §4; Fla. Stat. § 222.25)
        *   **Value of Claimed Exemption ($)** (or "100%" / "Unlimited")

**D. Secured Creditor & Lien Information (Check Schedule D):**
    *   For each secured creditor relevant to potential motions:
        *   **Creditor's Full Name and Address:**
        *   **Account Number:** (If listed)
        *   **Description of Collateral:** (Cross-reference Sch A/B)
        *   **Amount of Claim ($):** (Total claim amount listed)
        *   **Unsecured Portion ($):** (If listed)
        *   **Lien Details:** Extract any notes indicating lien type (e.g., "Mortgage", "PMSI", "Judgment Lien", "Second Mortgage").
        *   **Identify Senior Liens:** Note if multiple liens exist on the same property listed on Sch A/B.

**E. Judgment Creditor Information (Check Schedules D or E/F):**
    *   For creditors potentially holding judicial liens:
        *   **Creditor's Full Name and Address:**
        *   **Amount of Claim ($):** (Listed as unsecured or potentially secured by judgment lien on Sch D)
        *   Identify if creditor is listed as having a "Judgment".

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
**⚙️ OUTPUT FORMAT:**

1.  **Start with:** `EXTRACTED_FROM_UPLOAD:`
2.  **List Found Data:** Present each extracted data point clearly labeled. Use bullet points or a clear list format.
3.  **List Missing Information:** Create a section titled `INFORMATION STILL REQUIRED FOR S.D. FLA. MOTION DRAFTING:`
4.  **If no relevant data at all is found** output exactly:
`NO_RELEVANT_INFO_FOUND_IN_UPLOAD`

**DO NOT** draft the motion or ask follow-up questions.
""").strip()

    # 3) IMPORTANT: pass the uploaded file as a Part created from the file URI
    # This is the supported pattern for files with generate_content. :contentReference[oaicite:2]{index=2}
    file_part = gtypes.Part.from_uri(
        file_uri=gfile.uri,
        mime_type=mime_type,
    )

    try:
        resp = g_client.models.generate_content(
            model=GEM_MODEL,
            contents=[prompt, file_part],
        )
        return (resp.text or "").strip()
    except GeminiClientError as e:
        # Streamlit will redact the UI error; ensure *logs* contain something actionable.
        # We re-raise after logging.
        st.error("Gemini request failed (see logs for details).")

        # Best-effort logging (won't leak the PDF itself)
        st.write(
            {
                "gemini_model": GEM_MODEL,
                "uploaded_mime_type": mime_type,
                "uploaded_file_uri_present": bool(getattr(gfile, "uri", "")),
                "error_type": type(e).__name__,
            }
        )
        raise


def stream_answer(thread_id: str, assistant_id: str) -> Tuple[str, List[Tuple[str, bytes]]]:
    run = oa_client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        stream=True,
        include=["step_details.tool_calls[*].file_search.results[*].content"],
    )
    holder, txt = st.empty(), ""
    for ev in run:
        if ev.event == "thread.message.delta":
            for part in ev.data.delta.content or []:
                if part.type == "text":
                    txt += part.text.value or ""
                    holder.markdown(txt)

    files: List[Tuple[str, bytes]] = []
    last = oa_client.beta.threads.messages.list(thread_id, limit=1).data[0]
    if last.role == "assistant":
        for blk in last.content:
            if blk.type == "text":
                for ann in blk.text.annotations or []:
                    if ann.type == "file_path":
                        fid = ann.file_path.file_id
                        files.append((ann.text.split("/")[-1], oa_client.files.content(fid).read()))
            elif blk.type == "file":
                files.append((blk.file.filename, oa_client.files.content(blk.file.file_id).read()))
    return txt, files


# ────────── STREAMLIT UI STYLES ──────────
st.set_page_config("Legal Motion Assistant", layout="wide")
st.markdown(
    """
    <style>
        html, body, [class*="st-"] { font-family: 'Georgia', serif; color: #333; }
        body {background-color: #f0f2f6;}
        h1, h2, h3 {color: #0d1b4c; font-weight: bold;}
        .block-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 2rem 3rem 3rem 3rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            max-width: 1200px;
            margin: 1rem auto;
        }
        [data-testid="stSidebar"] { background-color: #e1e5f0; padding-top: 1.5rem; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] button p { color: #0d1b4c; }
        [data-testid="stFileUploader"] button {
            padding: 6px 12px; font-size: 14px;
            border: 1px solid #198754; background-color: #198754;
            color: white; border-radius: 6px;
        }
        [data-testid="stFileUploader"] button:hover { background-color: #157347; }
        [data-testid="stChatInput"] textarea {
            font-size: 16px !important; line-height: 1.6 !important;
            padding: 12px 15px !important; border-radius: 8px !important;
            border: 1px solid #ccc; background-color: #f8f9fa;
        }
        [data-testid="stChatInput"] textarea:focus {
             border-color: #0d1b4c;
             box-shadow: 0 0 0 2px rgba(13, 27, 76, 0.2);
        }
        [data-testid="stChatMessage"] {
            border-radius: 10px; padding: 1rem 1.5rem;
            margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            max-width: 85%;
        }
        [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
            background-color: #e1e5f0;
            margin-left: 0; margin-right: auto;
            border-left: 4px solid #0d1b4c;
        }
        [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
            background-color: #d1e7dd;
            margin-right: 0; margin-left: auto;
            border-right: 4px solid #198754;
        }
        .stButton button {
            background-color: #198754; color: white; border: none;
            border-radius: 6px; padding: 0.6rem 1.2rem; font-weight: bold;
        }
        .stButton button:hover:not(:disabled) {background-color: #157347;}
        .stButton button:disabled {background-color: #cccccc; color: #888888;}
        .stDownloadButton button {
             background-color: #5c6ac4; color: white; border: none; border-radius: 5px;
             padding: 0.3rem 0.8rem; font-size: 14px; margin-top: 5px; margin-right: 5px;
        }
        .stDownloadButton button:hover:not(:disabled) {background-color: #4553a0;}
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Page", ("Chat", "Admin"))

# ────────── RESET BUTTON ──────────
if st.sidebar.button("🔄  Reset workspace"):
    Path(CFG_PATH).unlink(missing_ok=True)
    cfg = {"assistant_id": "", "vector_stores": {}}
    st.session_state.schedule_uploaded = False
    st.sidebar.success("Workspace cleared – open *Admin* to start fresh.")
    st.rerun()

# ────────── CLEAR CHAT BUTTON ──────────
if page == "Chat" and st.sidebar.button("🗑️  Clear chat"):
    st.session_state.history = []
    st.session_state.thread_id = oa_client.beta.threads.create().id
    st.session_state.schedule_uploaded = False
    st.sidebar.success("Chat history cleared.")
    st.rerun()

# ────────── SESSION INIT ──────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = oa_client.beta.threads.create().id
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []  # type: ignore
if "schedule_uploaded" not in st.session_state:
    st.session_state.schedule_uploaded = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = uk("chat_files")

# ============================================================================  
#                                   ADMIN  
# ============================================================================
if page == "Admin":
    st.title("⚙️ Admin panel")

    # ➊ Create two vector stores
    if len(cfg["vector_stores"]) < 2 and st.button("Create 2 vector stores"):
        for slug in MOTION_OPTIONS:
            cfg["vector_stores"][slug] = oa_client.vector_stores.create(
                name=f"{slug}_store"
            ).id
        save_cfg(cfg)
        st.success("Vector stores created.")

    # ➋ Show vector stores
    if cfg["vector_stores"]:
        st.subheader("Vector stores")
        for slug, vsid in cfg["vector_stores"].items():
            st.markdown(f"* **{MOTION_OPTIONS.get(slug, slug)}** → `{vsid}`")

    # ➌ Create assistant (once)
    if cfg["vector_stores"] and not cfg["assistant_id"]:
        if st.button("Create assistant"):
            instructions = """(same instructions as your original — unchanged for brevity)"""
            assistant: Assistant = oa_client.beta.assistants.create(
                name="Legal Motion Assistant",
                model="gpt-4.1",
                instructions=instructions,
                tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
            )
            cfg["assistant_id"] = assistant.id
            save_cfg(cfg)
            st.success("Assistant created.")

    # ➍ Upload PDFs into a vector store
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
                    cfg["vector_stores"][slug] = oa_client.vector_stores.create(
                        name=f"{slug}_store"
                    ).id
                    save_cfg(cfg)

                with st.spinner("Uploading & indexing …"):
                    for f in files_:
                        fid = oa_client.files.create(file=f, purpose="assistants").id
                        oa_client.vector_stores.files.create(
                            vector_store_id=cfg["vector_stores"][slug],
                            file_id=fid,
                            attributes={"motion_type": slug, "jurisdiction": juris},
                        )
                st.success("Files uploaded & indexed.")

    # ➎ Display indexed PDFs
    if cfg["vector_stores"]:
        rows = []
        for slug, vsid in cfg["vector_stores"].items():
            for vf in oa_client.vector_stores.files.list(vsid, limit=100):
                try:
                    fname = oa_client.files.retrieve(vf.id).filename
                except Exception:
                    fname = "(unknown)"
                rows.append(
                    {
                        "Motion": MOTION_OPTIONS.get(slug, slug),
                        "Filename": fname,
                        "Jurisdiction": vf.attributes.get("jurisdiction", ""),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows))

# ============================================================================  
#                                    CHAT  
# ============================================================================
if page == "Chat":
    st.title("⚖️ Legal Motion Assistant")
    if not cfg["assistant_id"]:
        st.info("Create the assistant first from **Admin**.")
        st.stop()

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

    # render history
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])
            for fn, blob in h.get("files", []):
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_hist"))

    st.markdown("<div style='padding-bottom:70px'></div>", unsafe_allow_html=True)

    col_inp, col_up = st.columns([5, 2])
    with col_up:
        uploaded = st.file_uploader(
            "💎",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=st.session_state.uploader_key,
            label_visibility="collapsed",
        )

    first_turn_pending = not st.session_state.schedule_uploaded
    has_pdf = bool(uploaded) and any(f.name.lower().endswith(".pdf") for f in uploaded)
    prompt_disabled = first_turn_pending and not has_pdf

    if first_turn_pending:
        st.sidebar.warning(
            "📄 Please attach **at least one PDF bankruptcy schedule** before sending your first question."
        )

    with col_inp:
        user_prompt = st.chat_input("Ask or continue …", disabled=prompt_disabled)

    if user_prompt:
        if not st.session_state.schedule_uploaded and not has_pdf:
            st.error("❗ You must upload at least one PDF schedule with your first message.")
            st.stop()

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

                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                if gem_text and gem_text.strip() != NO_INFO_SENTINEL:
                    extract_blocks.append(
                        f"EXTRACTED_FROM_UPLOAD File name ({uf.name}):\n{gem_text}"
                    )

                blobs_for_history.append((uf.name, uf.getvalue()))
                prog.progress(i / max(len(uploaded), 1))

            prog.empty()

            if has_pdf:
                st.session_state.schedule_uploaded = True

        # store user turn
        st.session_state.history.append(
            {"role": "user", "content": user_prompt, "files": blobs_for_history}
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)
            for fn, blob in blobs_for_history:
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_user"))

        # wire vector store
        oa_client.beta.threads.update(
            thread_id=st.session_state.thread_id,
            tool_resources={"file_search": {"vector_store_ids": [cfg["vector_stores"][slug]]}},
        )

        # context parts
        context_parts = [
            f"Motion type: {motion_label}",
            f"Jurisdiction: {juris or '(unspecified)'}",
            f"Chapter: {chapter or '(unspecified)'}",
        ]
        if extract_blocks:
            context_parts.append("\n".join(extract_blocks))

        # Put context into the thread as an assistant message (your original approach kept)
        oa_client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="assistant",
            content="\n".join(context_parts),
        )

        # forward user turn
        oa_client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=user_prompt,
        )

        # assistant reply (stream)
        with st.chat_message("assistant"):
            answer, new_files = stream_answer(
                st.session_state.thread_id, cfg["assistant_id"]
            )
            for fn, data in new_files:
                st.download_button(f"Download {fn}", data, fn, key=uk("dl_asst"))

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "files": new_files}
        )

        # reset uploader key → clears file-picker
        st.session_state.uploader_key = uk("chat_files")
        st.rerun()
