import json, mimetypes, os, tempfile, uuid
from pathlib import Path
from typing import Dict, List, Tuple
import hmac

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types as gtypes
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

# ────────── OPENAI  CLIENT (Assistant) ──────────
oa_client = client = OpenAI(api_key=openai_api_key)

# ────────── GOOGLE GEMINI CLIENT ──────────
g_client = genai.Client(
    api_key=gemini_api_key,
    http_options=gtypes.HttpOptions(api_version="v1alpha"),
)
GEM_MODEL = "gemini-2.5-pro-exp-03-25"


# ────────── Password ──────────

def check_password():
    """
    Returns `True` if the user enters the correct password stored in Streamlit secrets.
    """
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(str(st.session_state["password"]), str(st.secrets["APP_PASSWORD"])):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.







def get_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "application/octet-stream"

def gem_upload(path: str) -> gtypes.File:
    return g_client.files.upload(
        file=path,
        config=gtypes.UploadFileConfig(mime_type=get_mime(path)),
    )

def gem_extract(path: str, user_prompt: str) -> str:
    """
    Uploads `path` to Gemini and asks it to extract facts needed for drafting
    bankruptcy motions. Majority of the information required is provided in
    the Motion-Specific Prompts block below.
    """
    # 1) upload the file
    gfile = gem_upload(path)

    # 2) build the extraction prompt
    prompt = (
        "You are a paralegal assistant specialized in extracting structured data from bankruptcy petition and schedule documents. "
        "When the user uploads any Official Form (e.g., Voluntary Petition, Schedule A/B, Schedule D, etc.) related to one of two motion types—"
        "Motion to Value Secured Claim or Motion to Avoid Judicial Lien—identify and explain every variable that can be reliably obtained. "
        "Recognize that different filings may include different subsets of fields; treat extracted values as authoritative and clearly note any missing required details.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🔖 COMMON VARIABLES (available in most petitions/schedules):\n"
        "  • Court jurisdiction\n"
        "  • Bankruptcy chapter (§7, §11, §13)\n"
        "  • Case number : Can be found at the top of The pages \n "
        "  • Filing date\n"
        "  • Debtor full name(s)\n"
        "  • Debtor mailing/residence address\n"
        "  • Attorney of record (name, firm, address, phone, bar number, email)\n"
        "  • Judge’s name (if listed)\n"
        "  • Hearing date & time (if scheduled)\n\n"
        "🏷️ MOTION-SPECIFIC VARIABLES:\n"
        "  ▶ Motion to Value Secured Claim:\n"
        "    • Creditor name(s)\n"
        "    • Collateral type (e.g., vehicle, real property)\n"
        "    • Collateral description / identifying details (VIN, address, legal description)\n"
        "    • Creditor’s claimed secured amount ($)\n"
        "    • Debtor’s asserted value ($)\n"
        "    • Basis for value (appraisal, market analysis, etc.)\n"
        "    • Lien position (first, second, etc.)\n"
        "    • Current balance owed on the loan ($)\n"
        "    • Proof of Claim number & supporting paragraph\n"
        "    • Appraiser name & appraisal date (Exhibit A)\n"
        "    • Statutory citations (11 U.S.C. § 506(a); Fed. R. Bankr. P. 3012)\n"
        "    • Exhibit references (Appraisal Report, Proof of Claim)\n\n"
        "  ▶ Motion to Avoid Judicial Lien:\n"
        "    • Creditor holding the judicial lien\n"
        "    • Property address & description\n"
        "    • Current market value ($)\n"
        "    • Existing mortgage balance(s) ($)\n"
        "    • Homestead exemption amount claimed ($)\n"
        "    • Face amount of the judicial lien ($)\n"
        "    • Date judgment was recorded (MM/DD/YYYY)\n"
        "    • Proof of Claim or Judgment number with recording details\n"
        "    • Statutory citations (11 U.S.C. § 522(f); Fed. R. Bankr. P. 7004/9014)\n"
        "    • Exhibit references (Judgment Lien, Valuation)\n\n"
        "🔍 HOW TO OUTPUT:\n"
        "  1. List each variable you found and its value.\n"
        "  2. Under “Missing Details,” list any required fields not present.\n"
        "  3. Do **not** draft the motion—extract only.\n"
        "  4. If no relevant information is present, output exactly:\n"
        "     NO_RELEVANT_INFO\n"
        
    )


    # 3) call Gemini to extract
    resp = g_client.models.generate_content(
        model=GEM_MODEL,
        contents=[prompt, gfile],
    )
    return resp.text.strip()

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
        /* --- Base & Fonts --- */
        html, body, [class*="st-"] {
            font-family: 'Georgia', serif; /* Classic legal font */
            color: #333; /* Darker text for readability */
        }
        body {
            background-color: #f0f2f6; /* Light gray background */
        }
        h1, h2, h3 {
            color: #0d1b4c; /* Professional dark blue */
            font-weight: bold;
        }

        /* --- Main Container --- */
        .block-container {
            background-color: #ffffff; /* White main content area */
            border-radius: 10px;
            padding: 2rem 3rem 3rem 3rem; /* More padding */
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            max-width: 1200px; /* Limit width for better focus */
            margin: 1rem auto; /* Center container */
        }

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background-color: #e1e5f0; /* Slightly darker sidebar */
            padding-top: 1.5rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] button p { /* Target button text */
             color: #0d1b4c; /* Dark blue text in sidebar */
        }
        [data-testid="stFileUploader"] button {
            padding: 6px 12px;
            font-size: 14px;
            border: 1px solid #198754;
            background-color: #198754;  /* Green */
            color: white;               /* White text for contrast */
            border-radius: 6px;
        }
        [data-testid="stFileUploader"] button:hover {
            background-color: #157347; /* Darker green on hover */
            color: white;
        }

        /* --- Chat Interface --- */
        [data-testid="stChatInput"] textarea {
            font-size: 16px !important;
            line-height: 1.6 !important;
            padding: 12px 15px !important;
            border-radius: 8px !important;
            border: 1px solid #ccc;
            background-color: #f8f9fa; /* Slightly off-white input */
        }
        [data-testid="stChatInput"] textarea:focus {
             border-color: #0d1b4c;
             box-shadow: 0 0 0 2px rgba(13, 27, 76, 0.2);
        }

        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
            max-width: 85%; /* Limit message width */
        }
        [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
            background-color: #e1e5f0; /* Light blue-gray for assistant */
            margin-left: 0;
            margin-right: auto; /* Align left */
            border-left: 4px solid #0d1b4c; /* Accent border */
        }
        [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
            background-color: #d1e7dd; /* Subtle green for user */
            margin-right: 0;
            margin-left: auto; /* Align right */
            border-right: 4px solid #198754; /* Green accent */
        }
        [data-testid="stChatMessage"] p {
            color: #333;
            margin-bottom: 0.5rem;
        }
        [data-testid="stChatMessage"] strong {
            color: #0d1b4c;
        }

        /* --- Buttons & Inputs --- */
        .stButton button {
            background-color: #198754;  /* Green background */
            color: white;               /* White text */
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            transition: background-color 0.2s ease;
        }
        .stButton button:hover:not(:disabled) {
            background-color: #157347; /* Darker green on hover */
        }
        .stButton button:disabled {
            background-color: #cccccc;
            color: #888888;
        }
        .stDownloadButton button {
             background-color: #5c6ac4; /* Slightly different blue */
             color: white;
             border: none;
             border-radius: 5px;
             padding: 0.3rem 0.8rem; /* Smaller padding */
             font-size: 14px;
             margin-top: 5px;
             margin-right: 5px; /* Spacing */
        }
        .stDownloadButton button:hover:not(:disabled) {
             background-color: #4553a0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Page", ("Chat", "Admin"))

# ────────── RESET BUTTON ──────────
if st.sidebar.button("🔄  Reset workspace"):
    Path(CFG_PATH).unlink(missing_ok=True)
    cfg = {"assistant_id": "", "vector_stores": {}}
    st.sidebar.success("Workspace cleared – open *Admin* to start fresh.")
    st.rerun()

# ────────── CLEAR CHAT BUTTON ──────────
if page == "Chat" and st.sidebar.button("🗑️  Clear chat"):
    # Clear chat history and create a new thread
    st.session_state.history = []
    st.session_state.thread_id = oa_client.beta.threads.create().id
    st.sidebar.success("Chat history cleared.")
    st.rerun()


# ────────── SESSION INIT ──────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = oa_client.beta.threads.create().id
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = [] # type: ignore

# ────────── Uploader key for clearing ──────────
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
            instructions = """
                You are a **Bankruptcy Motion Drafting Assistant**.

                You may receive **system messages** that contain “EXTRACTED_FROM_UPLOAD:” followed
                by bullet-point facts from user-uploaded documents extracted. and Include them. 
                And tell the user was was present in those extracted data and what are the details Still missing This is important. 
                Treat those extracts as authoritative evidence.

                If details are still missing, ask the user.

                    ❗️ **Never begin drafting until every required field is confirmed.**  
                    After the user chooses a motion type, send **one single message** that lists **all** required inputs as bullet-point questions.  
                    Wait for clear answers to every item (or explicit “N/A”) **before** you draft anything.

                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    🔖 **COMMON DETAILS** — ask these first (for either motion)

                    • Court jurisdiction (e.g., “Southern District of Florida”)  
                    • Bankruptcy chapter (§7, §11, or §13)  
                    • **Attorney selection** – after gathering the other inputs, present *exactly three* attorneys drawn from the static list supplied in the project files.  
                    For each attorney, show:  
                        – Name (e.g., “Alicia M. Perez, Esq.”)  
                        – Firm (e.g., “Perez & Associates, PLLC”)  
                        – Short 1-sentence description of expertise / jurisdiction focus  
                    Ask the user to pick **one** of the three.  
                    *You must insert the chosen attorney’s full name, firm, address, phone, bar number (if provided), and email into the signature block of both the Motion and the Proposed Order.*  
                    • Case number, judge’s name, hearing date/time (placeholders acceptable)

                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    🏷️ **Motion-Specific Prompts**  
                    (Ask only the block that matches the selected motion type.)

                    1️⃣ **Motion to Value Secured Claim** — please provide:  
                    • Debtor’s full name  
                    • Creditor’s name  
                    • Collateral type (e.g., “2019 Ford F-150”, “123 Main St.”)  
                    • Collateral description / identifying details  
                    • Creditor’s claimed value ($)  
                    • Debtor’s asserted value ($)  
                    • Basis for value (Appraisal, Kelley Blue Book, Market Analysis, etc.)  
                    • Lien position (First, Second, Third …)  
                    • Current balance owed on the loan ($)  
                    • ⏩ *Optional*: upload or cite supporting valuation documents  

                    🔍 **AI Assistant Must Also Include** for this motion:  
                    • Proof of Claim number and paragraph supporting secured amount  
                    • Vehicle VIN or parcel ID / serial number  
                    • Appraiser name & appraisal date (attach as Exhibit A)  
                    • Statutory citations: 11 U.S.C. § 506(a); Fed. R. Bankr. P. 3012  
                    • Structured headings: Jurisdiction & Venue, Factual Background, Legal Standard, Relief Requested, Certificate of Service  
                    • Placeholders for [Judge’s Name], [Date of Hearing], [Hearing Time]  
                    • Exhibit references: Exhibit A (Appraisal Report), Exhibit B (Proof of Claim)  
                    • Draft must match local style for both Motion **and** Proposed Order  
                    • Signature block with the **selected attorney’s** details  

                    2️⃣ **Motion to Avoid Judicial Lien** — please provide:  
                    • Debtor’s full name  
                    • Creditor holding the judicial lien  
                    • Street address of the affected property  
                    • Current market value of the property ($)  
                    • Existing mortgage balance(s) ($)  
                    • Amount claimed as homestead exemption ($)  
                    • Face amount of the judicial lien ($)  
                    • Date the judgment was recorded (MM/DD/YYYY)  

                    🔍 **AI Assistant Must Also Include** for this motion:  
                    • Proof of Claim or Judgment number with recording details  
                    • Statutory citations: 11 U.S.C. § 522(f); Fed. R. Bankr. P. 7004 / 9014  
                    • Structured headings: Jurisdiction & Venue, Factual Background, Legal Standard, Relief Requested, Certificate of Service  
                    • Placeholders for [Judge’s Name], [Date of Hearing], [Hearing Time]  
                    • Exhibit references: Exhibit A (Judgment Lien), Exhibit B (Property Valuation)  
                    • Draft must match local style for both Motion **and** Proposed Order  
                    • Signature block with the **selected attorney’s** details  

                ────────────────────────────────────────
                When *all* answers (including the attorney choice) are received, draft the Motion **followed by** the Proposed Order.
                To Draft the Motion GO through the custom knowledge files provided using file_search to get all relevant sections and instrucitons on how to craft each motion type. Do not rely on your general knoweldge.
                Use only the Uploaded knowledge.
                After generating the Motion, ask the user if any modifications are needed; if not, ask whether they would like to download two separate .docx files (one for the Motion, one for the Proposed Order).  
                Crafting Each motion needs to be from the custom knowledge in the project’s internal file database (accessed via file search)
                Provide the download links upon confirmation.

                Append this disclaimer at the top of every draft:  

                > *“This is an AI-generated draft. Review by a licensed attorney is required.”*
                            """
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
#                                    Hide Streamlit
# ============================================================================

#hide_streamlit_style = """
 #   <style>
    #MainMenu {visibility: hidden;}
  #  /* Hide footer */
  #  footer {visibility: hidden;}
  #  /* Optionally hide the header (if any) */
  #  header {visibility: hidden;}
  #  </style>
   # """
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ============================================================================  
#                                    CHAT  
# ============================================================================
if page == "Chat":
    st.title("⚖️ Legal Motion Assistant")
    if not cfg["assistant_id"]:
        st.info("Create the assistant first from **Admin**.")
        st.stop()

    # REQUIRED motion type
    motion_label = st.sidebar.selectbox(
        "Motion type (required)", ["— Select —"] + list(MOTION_OPTIONS.values()),
        key="motion_type"
    )
    slug = next((k for k, v in MOTION_OPTIONS.items() if v == motion_label), None) \
        if motion_label != "— Select —" else None

    juris = st.sidebar.text_input("Jurisdiction (optional)")
    chapter = st.sidebar.selectbox("Bankruptcy Chapter (optional)", ["", "7", "11", "13"])

    if slug is None:
        st.sidebar.error("Please select a motion type to enable chat.")
        st.stop()

    # Show chat history
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])
            for fn, blob in h.get("files", []):
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_hist"))

    st.markdown("<div style='padding-bottom:80px'></div>", unsafe_allow_html=True)

    col_inp, col_up = st.columns([5, 2])
    with col_up:
        uploaded = st.file_uploader(
            "💎", type=["pdf", "docx", "txt"], accept_multiple_files=True,
            key=st.session_state.uploader_key, label_visibility="collapsed"
        )
    with col_inp:
        user_prompt = st.chat_input("Ask or continue …")

    if user_prompt:
        # 1. Process uploads via Gemini (if any)
        extract_blocks, blobs_for_history = [], []
        if uploaded:
            prog = st.progress(0.0)
            for i, uf in enumerate(uploaded, 1):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name
                with st.spinner(f"Gemini reading {uf.name} …"):
                    gem_text = gem_extract(tmp_path, user_prompt)
                if gem_text and gem_text != "NO_RELEVANT_INFO":
                    extract_blocks.append(
                        f"EXTRACTED_FROM_UPLOAD File name ({uf.name}):\n{gem_text}"
                    )
                blobs_for_history.append((uf.name, uf.getvalue()))
                prog.progress(i / len(uploaded))
            prog.empty()

        st.session_state.history.append(
            {"role": "user", "content": user_prompt, "files": blobs_for_history}
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)
            for fn, blob in blobs_for_history:
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_user"))

        # 2. Wire correct vector store (already guaranteed by required motion)
        oa_client.beta.threads.update(
            thread_id=st.session_state.thread_id,
            tool_resources={"file_search":
                            {"vector_store_ids": [cfg["vector_stores"][slug]]}},
        )

        # 3. System context message
        context_parts = [
            f"Motion type: {motion_label}",
            f"Jurisdiction: {juris or '(unspecified)'}",
            f"Chapter: {chapter or '(unspecified)'}",
        ]
        if extract_blocks:
            context_parts.append("\n".join(extract_blocks))
        oa_client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="assistant",
            content="\n".join(context_parts),
        )

        # 4. Forward user turn
        oa_client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=user_prompt,
        )

        # 5. Stream assistant reply
        with st.chat_message("assistant"):
            answer, new_files = stream_answer(
                st.session_state.thread_id, cfg["assistant_id"]
            )
            for fn, data in new_files:
                st.download_button(f"Download {fn}", data, fn, key=uk("dl_asst"))

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "files": new_files}
        )

        # 6. RESET THE UPLOADER
        st.session_state.uploader_key = uk("chat_files")
        st.rerun()
