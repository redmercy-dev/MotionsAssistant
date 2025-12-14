import json
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any
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

# Choose your OpenAI model for Responses
OPENAI_MODEL = "gpt-4.1"  # recommended in migration examples; change if you want
# Gemini model for extraction
GEM_MODEL = "gemini-2.0-flash-exp"

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
oa_client = OpenAI(api_key=openai_api_key)

# ────────── GOOGLE GEMINI CLIENT ──────────
g_client = genai.Client(api_key=gemini_api_key)

# ────────── Password ──────────
def check_password():
    """
    Returns `True` if the user enters the correct password stored in Streamlit secrets.
    """
    def password_entered():
        if hmac.compare_digest(
            str(st.session_state.get("password", "")),
            str(st.secrets["APP_PASSWORD"]),
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
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
    Uploads `path` to Gemini and asks it to extract facts needed for drafting
    bankruptcy motions.
    """
    gfile = gem_upload(path)

    prompt = """
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
2.  **List Found Data:** Present each extracted data point clearly labeled (bullet points or clear list).
3.  **List Missing Information:** Section titled `INFORMATION STILL REQUIRED FOR S.D. FLA. MOTION DRAFTING:` listing required-but-not-found items.
4.  **If no relevant data at all is found**, output exactly:
    `NO_RELEVANT_INFO_FOUND_IN_UPLOAD`

**DO NOT** attempt to draft the motion or ask follow-up questions yourself. Your sole task is extraction and reporting based on the content of the uploaded schedules PDF.
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
    return (resp.text or "").strip()

# ────────── OPENAI: Responses streaming helper ──────────
def _safe_get(d: Any, *path, default=None):
    cur = d
    for p in path:
        try:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                cur = getattr(cur, p)
        except Exception:
            return default
        if cur is None:
            return default
    return cur

def _extract_files_from_response_obj(resp_obj: Any) -> List[Tuple[str, bytes]]:
    """
    Best-effort extraction of files returned by Responses API output items.
    If your model uses code_interpreter, it may generate files with file_ids.
    """
    files: List[Tuple[str, bytes]] = []

    output_items = _safe_get(resp_obj, "output", default=[]) or []
    for item in output_items:
        item_type = _safe_get(item, "type", default="")
        if item_type != "message":
            continue

        content_list = _safe_get(item, "content", default=[]) or []
        for part in content_list:
            ptype = _safe_get(part, "type", default="")

            # Common patterns seen in Responses output payloads
            # - output_file with file_id + filename
            # - file with file_id + filename
            file_id = _safe_get(part, "file_id") or _safe_get(part, "file", "file_id")
            filename = _safe_get(part, "filename") or _safe_get(part, "file", "filename")

            if file_id:
                try:
                    blob = oa_client.files.content(file_id).read()
                    files.append((filename or f"{file_id}.bin", blob))
                except Exception:
                    # If file download fails, skip silently (don’t break chat)
                    pass

    return files

def stream_answer_responses(
    user_prompt: str,
    motion_label: str,
    juris: str,
    chapter: str,
    extract_blocks: List[str],
    vector_store_id: str,
) -> Tuple[str, List[Tuple[str, bytes]], str]:
    """
    Sends one turn via Responses API with streaming.
    Returns (assistant_text, files, new_previous_response_id)
    """
    # Build context "system-like" message (kept similar to your Assistants approach).
    context_parts = [
        f"Motion type: {motion_label}",
        f"Jurisdiction: {juris or '(unspecified)'}",
        f"Chapter: {chapter or '(unspecified)'}",
    ]
    if extract_blocks:
        context_parts.append("\n".join(extract_blocks))
    context_blob = "\n".join(context_parts)

    # Your previous Assistants instructions become Responses "instructions"
    # (The migration guide recommends prompts via dashboard for production,
    # but keeping it inline matches your current code behavior.)
    instructions = LEGAL_MOTION_INSTRUCTIONS()

    tools = [
        {"type": "file_search", "vector_store_ids": [vector_store_id]},
        {"type": "code_interpreter"},
    ]

    prev_id = st.session_state.get("previous_response_id")

    holder, txt = st.empty(), ""
    final_resp_obj = None

    # Stream events
    stream = oa_client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=[
            {"role": "assistant", "content": context_blob},
            {"role": "user", "content": user_prompt},
        ],
        tools=tools,
        store=True,
        previous_response_id=prev_id,
        stream=True,
    )

    for ev in stream:
        ev_type = _safe_get(ev, "type", default="")

        # Primary text streaming event
        if ev_type == "response.output_text.delta":
            delta = _safe_get(ev, "delta", default="") or ""
            if delta:
                txt += delta
                holder.markdown(txt)

        # Capture the completed response object (has output items, id, etc.)
        elif ev_type == "response.completed":
            final_resp_obj = _safe_get(ev, "response")

    # Fallback: if we somehow didn't get completed event, do a non-stream call
    # (rare, but prevents losing the turn)
    if final_resp_obj is None:
        final_resp_obj = oa_client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=[
                {"role": "assistant", "content": context_blob},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            store=True,
            previous_response_id=prev_id,
        )
        txt = getattr(final_resp_obj, "output_text", "") or txt
        holder.markdown(txt)

    new_prev_id = _safe_get(final_resp_obj, "id", default=prev_id) or prev_id
    files = _extract_files_from_response_obj(final_resp_obj)
    return txt, files, new_prev_id

# ────────── Instructions as a function (keeps file cleaner) ──────────
def LEGAL_MOTION_INSTRUCTIONS() -> str:
    return """
**Bankruptcy Motion Drafting (Southern District of Florida Focus)**

> <!-- **MANDATORY DISCLAIMER:** Place this exact text at the very top of *every* generated draft: -->
> *"This is an AI-generated draft. Review by a licensed attorney is required."*

**Your Role:** You are a **Bankruptcy Motion Drafting Assistant** specializing in the **Southern District of Florida (S.D. Fla.)**. Your primary function is to draft specific S.D. Fla. bankruptcy motions (Motion to Value Secured Claim §506 OR Motion to Avoid Judicial Lien §522(f)) and corresponding Proposed Orders, strictly adhering to the details provided by the user, data extracted from uploaded documents (especially Bankruptcy Schedules), and the S.D. Fla. procedural/formatting rules outlined in the **Uploaded Knowledge File**.

**Handling Uploaded Data (Schedules Prioritized):**
*   You will likely receive system messages like `EXTRACTED_FROM_UPLOAD:`. Treat these bullet points, **especially data from uploaded Bankruptcy Schedules (Schedules A/B, C, D, E/F)**, as the primary source of truth and factual evidence provided by the user.
*   **Immediately** after processing extracted data (from schedules or other docs):
    *   Report to the user: "Based on the uploaded [Document Type, e.g., Schedules], I have extracted the following details: [List extracted items relevant to the motion]."
    *   Then state: "To complete the motion for the Southern District of Florida, I still need the following information: [List *only* the specific required details (from Steps 1 & 2 below) that were *not* found in the extracted data]."
*   Do **NOT** proceed to ask unrelated questions or begin drafting until these specific missing details are provided or confirmed as N/A by the user.

**Critical Instruction: Confirm Before Drafting**
*   ❗️ **NEVER BEGIN DRAFTING UNTIL EVERY REQUIRED FIELD FOR S.D. FLA. IS CONFIRMED.**
*   After the user confirms the **Motion Type**, cross-reference with any uploaded data. Then, send **ONE SINGLE MESSAGE** that lists **ONLY** the S.D. Fla. required inputs (Common Details + Specific Motion Details below) that are **still missing** after analyzing uploads. Frame these as clear, bullet-point questions.
*   Wait for the user to provide explicit answers to **every single missing item** (or state "N/A" where applicable) **before** you initiate the drafting process. This includes the Attorney Selection step.

[...KEEP YOUR FULL INSTRUCTIONS HERE...]
""".strip()

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

        [data-testid="stSidebar"] {
            background-color: #e1e5f0;
            padding-top: 1.5rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] button p {
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

        [data-testid="stChatMessage"] {
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    cfg = {"vector_stores": {}}
    st.session_state.schedule_uploaded = False
    st.session_state.history = []
    st.session_state.previous_response_id = None
    st.session_state.uploader_key = uk("chat_files")
    st.sidebar.success("Workspace cleared – open *Admin* to start fresh.")
    st.rerun()

# ────────── CLEAR CHAT BUTTON ──────────
if page == "Chat" and st.sidebar.button("🗑️  Clear chat"):
    st.session_state.history = []
    st.session_state.previous_response_id = None
    st.session_state.schedule_uploaded = False
    st.sidebar.success("Chat history cleared.")
    st.rerun()

# ────────── SESSION INIT ──────────
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []  # type: ignore
if "schedule_uploaded" not in st.session_state:
    st.session_state.schedule_uploaded = False
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = uk("chat_files")

# ============================================================================  
#                                   ADMIN  
# ============================================================================
if page == "Admin":
    st.title("⚙️ Admin panel")

    # ➊ Create two vector stores
    if len(cfg.get("vector_stores", {})) < 2 and st.button("Create 2 vector stores"):
        cfg.setdefault("vector_stores", {})
        for slug in MOTION_OPTIONS:
            vs = oa_client.vector_stores.create(name=f"{slug}_store")
            cfg["vector_stores"][slug] = vs.id
        save_cfg(cfg)
        st.success("Vector stores created.")

    # ➋ Show vector stores
    if cfg.get("vector_stores"):
        st.subheader("Vector stores")
        for slug, vsid in cfg["vector_stores"].items():
            st.markdown(f"* **{MOTION_OPTIONS.get(slug, slug)}** → `{vsid}`")

    # ➍ Upload PDFs into a vector store
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
                    vs = oa_client.vector_stores.create(name=f"{slug}_store")
                    cfg["vector_stores"][slug] = vs.id
                    save_cfg(cfg)

                vsid = cfg["vector_stores"][slug]

                with st.spinner("Uploading & indexing …"):
                    for f in files_:
                        fid = oa_client.files.create(file=f, purpose="assistants").id
                        oa_client.vector_stores.files.create(
                            vector_store_id=vsid,
                            file_id=fid,
                            attributes={"motion_type": slug, "jurisdiction": juris},
                        )
                st.success("Files uploaded & indexed.")

    # ➎ Display indexed PDFs
    if cfg.get("vector_stores"):
        rows = []
        for slug, vsid in cfg["vector_stores"].items():
            listing = oa_client.vector_stores.files.list(vector_store_id=vsid, limit=100)
            for vf in listing.data:
                # VectorStoreFile object has file_id
                file_id = getattr(vf, "file_id", None) or getattr(vf, "id", None)
                fname = "(unknown)"
                try:
                    if file_id:
                        fname = oa_client.files.retrieve(file_id).filename
                except Exception:
                    fname = "(unknown)"
                attrs = getattr(vf, "attributes", {}) or {}
                rows.append(
                    {
                        "Motion": MOTION_OPTIONS.get(slug, slug),
                        "Filename": fname,
                        "Jurisdiction": attrs.get("jurisdiction", ""),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows))

# ============================================================================  
#                                    CHAT  
# ============================================================================
if page == "Chat":
    st.title("⚖️ Legal Motion Assistant")

    if not cfg.get("vector_stores"):
        st.info("Create vector stores and upload knowledge first from **Admin**.")
        st.stop()

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
        st.sidebar.error("Vector store missing for this motion type. Create them in Admin.")
        st.stop()

    vector_store_id = cfg["vector_stores"][slug]

    # ───── Render history ─────
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])
            for fn, blob in h.get("files", []):
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_hist"))

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

                if gem_text and gem_text != "NO_RELEVANT_INFO_FOUND_IN_UPLOAD":
                    extract_blocks.append(f"EXTRACTED_FROM_UPLOAD File name ({uf.name}):\n{gem_text}")

                blobs_for_history.append((uf.name, uf.getvalue()))
                prog.progress(i / len(uploaded))

            prog.empty()
            if any(f.name.lower().endswith(".pdf") for f in uploaded):
                st.session_state.schedule_uploaded = True

        # store user turn in local UI history
        st.session_state.history.append(
            {"role": "user", "content": user_prompt, "files": blobs_for_history}
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)
            for fn, blob in blobs_for_history:
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_user"))

        # assistant reply (Responses streaming)
        with st.chat_message("assistant"):
            answer, new_files, new_prev_id = stream_answer_responses(
                user_prompt=user_prompt,
                motion_label=motion_label,
                juris=juris,
                chapter=chapter,
                extract_blocks=extract_blocks,
                vector_store_id=vector_store_id,
            )

            # persist previous_response_id for multi-turn continuity
            st.session_state.previous_response_id = new_prev_id

            for fn, data in new_files:
                st.download_button(f"Download {fn}", data, fn, key=uk("dl_asst"))

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "files": new_files}
        )

        # reset uploader key → clears file-picker
        st.session_state.uploader_key = uk("chat_files")
        st.rerun()
