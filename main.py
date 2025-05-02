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

# ────────── Gemini Functions ──────────

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
    prompt = ( """
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
    2.  **List Found Data:** Present each extracted data point clearly labeled (e.g., `Debtor Name: John Doe`, `Real Property Address 1: 123 Main St, Miami, FL`, `Schedule A Value: $300,000`, `Schedule D Creditor 1 Name: ABC Bank`, `Schedule D Claim Amount: $50,000`, `Schedule C Exemption Statute: Fla. Const. Art. X, §4`). Use bullet points or a clear list format.
    3.  **List Missing Information:** Create a section titled `INFORMATION STILL REQUIRED FOR S.D. FLA. MOTION DRAFTING:`
        *   Under this heading, list **only** the specific data points (using the terminology from the *Drafting Assistant's* prompt) that were **required** for the potential motion type (Value or Avoid Lien in S.D. Fla) but were **not found** in the schedules. Examples might include:
            *   "Full Case Number (S.D. Fla. Format)"
            *   "Judge's Full Name"
            *   "Motion Type Selection (Value or Avoid Lien)"
            *   "Creditor Selection (if multiple relevant creditors found)"
            *   "Vehicle Odometer Reading"
            *   "Real Property Legal Description (if not on Sch A)"
            *   "Basis for Property Value (e.g., Appraisal, KBB)"
            *   "Proof of Claim Number and Amount (for Value Motion)"
            *   "Judgment Recording Information (Date, Cert #/Book/Page - for Avoid Lien Motion)"
            *   "Confirmation: Exhibit A (Lien Copy) will be attached (for Avoid Lien Motion)"
            *   "Confirmation: Exhibit A (Appraisal) will be attached (if using Appraisal for Value Motion)"
            *   "Procedural Choice: Negative Notice or Set Hearing (for Avoid Lien Motion)"
            *   "Hearing Date/Time/Location (if setting hearing)"
            *   "Attorney Selection (To be asked later)"
    4.  **If no relevant data at all is found** in the document related to debtors, property, creditors, or exemptions, output exactly:
        `NO_RELEVANT_INFO_FOUND_IN_UPLOAD`

    **DO NOT** attempt to draft the motion or ask follow-up questions yourself. Your sole task is extraction and reporting based on the content of the uploaded schedules PDF."""
    )


    # 3) call Gemini to extract
    resp = g_client.models.generate_content(
        model=GEM_MODEL,
        contents=[prompt, gfile],
    )
    return resp.text.strip()

# ────────── Streaming assistant responses ──────────


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

        /* --- Buttons & Inputs --- */
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
    # 📌 NEW – also reset schedule flag
    st.session_state.schedule_uploaded = False
    st.sidebar.success("Workspace cleared – open *Admin* to start fresh.")
    st.rerun()

# ────────── CLEAR CHAT BUTTON ──────────
if page == "Chat" and st.sidebar.button("🗑️  Clear chat"):
    st.session_state.history = []
    st.session_state.thread_id = oa_client.beta.threads.create().id
    # 📌 NEW – reset schedule requirement for new thread
    st.session_state.schedule_uploaded = False
    st.sidebar.success("Chat history cleared.")
    st.rerun()

# ────────── SESSION INIT ──────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = oa_client.beta.threads.create().id
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []  # type: ignore
# 📌 NEW – track whether the schedule PDF has been provided
if "schedule_uploaded" not in st.session_state:
    st.session_state.schedule_uploaded = False

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
            **AI Assistant Prompt: Bankruptcy Motion Drafting (Southern District of Florida Focus)**

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

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            🔖 **STEP 1: GATHER/CONFIRM COMMON S.D. FLA. DETAILS** — Ask *only* for items missing after checking uploads:

            *   **Court Division:** (e.g., "Miami Division", "Fort Lauderdale Division", "West Palm Beach Division"). *Jurisdiction is Southern District of Florida.*
            *   **Bankruptcy Chapter:** (§7, §11, or §13). *(Likely in schedules)*
            *   **Debtor(s) Full Name(s):** As it should appear in the case caption. *(Likely in schedules)*
            *   **Full Case Number:** Including Judge's Initials (S.D. Fla format: `XX-XXXXX-ABC`). *(User must provide)*
            *   **Judge's Full Name:** (e.g., "Hon. A. Bruce Cogburn, U.S.B.J."). *(User must provide)*
            *   **Procedural Approach (S.D. Fla Specific):**
                *   **IF Motion to Value (§506):**
                    *   Confirm: "Is the collateral **Real Property** or **Personal Property**?" (This determines which S.D. Fla Local Form structure to follow).
                    *   Confirm: "S.D. Fla. typically requires these motions to be set for hearing with 21 days notice. Do you have a hearing date/time/location, or should I use placeholders?"
                *   **IF Motion to Avoid Lien (§522(f)):**
                    *   Confirm: "S.D. Fla. allows this motion via 21-day Negative Notice (Local Rule 9013-1(D)) or by setting a hearing. Which procedure do you want to use: **Negative Notice** or **Set Hearing**?"
                    *   If **Set Hearing**: "Do you have the hearing date/time/location, or should I use placeholders?"
                    *   If **Negative Notice**: Acknowledge standard 21-day language will be included in the motion.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            🏷️ **STEP 2: GATHER/CONFIRM MOTION-SPECIFIC S.D. FLA. DETAILS** — Ask *only* for items missing after checking uploads for the chosen motion type:

            1️⃣ **IF Motion to Value Secured Claim (§506 - S.D. Fla.)** — Please provide missing details for Local Form structure:
                *   **Creditor's Full Name:** *(Likely on Schedule D)*
                *   **Collateral Full Description:** *(Check Schedules A/B/D first)*
                    *   For **Vehicle:** Needs Year, Make, Model, **VIN**, **Odometer Reading**.
                    *   For **Real Property:** Needs Full Street Address AND **Full Legal Description** (Mandatory per S.D. Fla rules).
                    *   For **Other Personal Property:** Needs Detailed description.
                *   **Basis for Lien:** (e.g., Purchase Money Security Agreement, Mortgage). *(Check Schedule D notes)*
                *   **Collateral Value:** Debtor's asserted **Fair Market Value ($)** as of the petition date. *(Check Schedules A/B)*
                *   **Basis for Value:** (e.g., Appraisal dated MM/DD/YYYY, KBB, NADA, Tax Assessment, Debtor Estimate). If Appraisal, confirm "Exhibit A (Appraisal)" will be attached. *(May need user input)*
                *   **Proof of Claim Status:** "Has Creditor filed a Proof of Claim? (Yes/No). If Yes, what is the **POC Number** and **Amount Claimed ($)**?" (Needed for S.D. Fla form alternate paragraphs). *(User likely needs to provide)*
                *   **Senior Liens (If Bifurcating/Stripping):** List senior lienholder(s) and payoff amount(s) ($). *(Check Schedule D)*
                *   **Proposed Treatment (Chapter 13 Only):** How will secured/unsecured portions be treated? (e.g., Secured $[X] at Y% interest). *(May need user input/Plan details)*

            🔍 ***AI Assistant MUST Ensure for S.D. Fla. §506 Motion:***
                *   **Follow S.D. Fla Local Form Structure:** Use numbered paragraphs mirroring LF-102 (Personal Property) or the Real Property equivalent as described in the knowledge file.
                *   **Include MANDATORY "IMPORTANT NOTICE..." block** from knowledge file/Local Form.
                *   **Citations:** 11 U.S.C. § 506(a), Fed. R. Bankr. P. 3012, S.D. Fla. Local Rule 3015-3. Add § 1322(b)(2) if Ch 13 strip-off.
                *   **Content:** Accurately reflect collateral details, value, basis, POC status (using correct alternate paragraph), and treatment.
                *   **Formatting:** Adhere strictly to S.D. Fla rules (12pt font, 1.5 spacing, margins, caption style, **bold/descriptive Title**, ALL CAPS WHEREFORE).
                *   **Exhibits:** Reference Exhibit A (Appraisal) if applicable.
                *   **Verification:** S.D. Fla forms typically do *not* require separate debtor verification.
                *   **Proposed Order:** Remind user a corresponding S.D. Fla Local Form Order must be submitted.

            2️⃣ **IF Motion to Avoid Judicial Lien (§522(f) - S.D. Fla.)** — Please provide missing details per Local Rule 4003-2:
                *   **Creditor Holding Lien:** Full Name. *(Check Schedule D/E/F)*
                *   **Lien Details:** *(Check Schedule D/E/F notes first)*
                    *   Originating **Court**, **Case Number**, **Date** of Judgment, **Amount ($)**.
                    *   **Recording Information:** Date and specific details (e.g., Judgment Lien Certificate # in FL State Registry, OR Book/Page if older abstract).
                    *   Confirm: "**Exhibit A (Copy of Judgment/Lien Certificate)** will be attached." (MANDATORY per S.D. Fla LR 4003-2).
                *   **Affected Property Description:** *(Check Schedules A/B/C)*
                    *   For **Real Property:** Full Street Address AND **Full Legal Description** (MANDATORY).
                    *   For **Personal Property:** Detailed Description matching Schedule C (e.g., Household Goods, Vehicle Y/M/M/VIN).
                *   **Property Value:** Current **Fair Market Value ($)**. *(Check Schedule A/B)*
                *   **Exemption Claimed:** Specific **Statute** (e.g., Fla. Const. Art. X, §4; Fla. Stat. § 222.25) and **Amount ($)** claimed (or "Unlimited"). *(Check Schedule C)*
                *   **Other Liens on Property:** List **each** other unavoidable lien (e.g., Mortgages) with **Holder Name** and **Current Balance ($)**. *(Check Schedule D)*

            🔍 ***AI Assistant MUST Ensure for S.D. Fla. §522(f) Motion:***
                *   **Content per LR 4003-2:** Must include full legal description (real property), lien origin/recording details.
                *   **Exhibit A MANDATORY:** State that a copy of the judgment/lien *must* be attached as Exhibit A.
                *   **Citations:** 11 U.S.C. § 522(f), Fed. R. Bankr. P. 4003(d). Reference Rule 7004/9014 regarding service.
                *   **Impairment:** Include the mathematical impairment calculation (§ 522(f)(2)(A)).
                *   **Formatting:** Adhere strictly to S.D. Fla rules (12pt font, 1.5 spacing, margins, caption style, **bold/descriptive Title**, numbered paragraphs).
                *   **Negative Notice:** If chosen by user, include the standard S.D. Fla 21-day negative notice language (per LR 9013-1(D)) prominently.
                *   **Verification:** Debtor verification is *not* typically required by S.D. Fla local rule for §522f motions.
                *   **Proposed Order:** Remind user a Proposed Order including the **full legal description** must be submitted.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            🎩 **STEP 3: ATTORNEY SELECTION** — Ask this *after* confirming all other details:

            *   "Based on the case details for the Southern District of Florida, here are three attorneys potentially suitable for this matter. Please select one:"
                *   *(Present 3 Attorney Options here - drawn from project files or provide placeholders)*
                    *   **Attorney 1:** Name, Firm, S.D. Fla focus description.
                    *   **Attorney 2:** Name, Firm, S.D. Fla focus description.
                    *   **Attorney 3:** Name, Firm, S.D. Fla focus description.
            *   "Please type the number (1, 2, or 3) of the attorney you select."

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ⚖️ **STEP 4: DRAFTING INSTRUCTIONS (S.D. FLA.)**

            *   Once **ALL** information from Steps 1, 2, and 3 is confirmed, proceed to draft for the **Southern District of Florida**.
            *   **Consult the Uploaded Knowledge File EXTENSIVELY** using `file_search` specifically for **S.D. Fla rules and practices**.
            *   **Structure & Content:**
                *   **Motion to Value:** Strictly follow S.D. Fla. Local Form structure (LF-102 personal / real property equivalent) including numbered paragraphs and the mandatory "IMPORTANT NOTICE" block. Use ALL CAPS WHEREFORE clause.
                *   **Motion to Avoid Lien:** Follow S.D. Fla. LR 4003-2 requirements. Include legal description, lien details, impairment calculation. Include 21-day negative notice language if selected. State Exhibit A (Lien Copy) is attached.
            *   **Apply S.D. Fla. Formatting:**
                *   Font: 12-point minimum.
                *   Spacing: 1.5 lines (except block quotes).
                *   Margins: Minimum 1" top, 0.75" sides/bottom.
                *   Caption: Centered Court Name; `In re:` Debtor left; Case # (`XX-XXXXX-ABC`)/Chapter right.
                *   Title: Centered, **Bold**, Descriptive (Party, Relief, Creditor, Collateral).
                *   Headings: Generally ALL CAPS/Bold if used within text (like in Value Form title/notice), otherwise numbered paragraphs flow.
                *   Signature Block: Include "Submitted by:" prefix, `/s/ Attorney Name`, Full Name, Firm, Address, Telephone, Email, **Florida Bar Number**, "Attorney for Debtor(s)".
                *   Certificate of Service: Use standard S.D. Fla language ("I HEREBY CERTIFY..."), list parties served via CM/ECF and Mail (specify method - **comply with Rule 7004 for Creditor/Lienholder**).
            *   Draft the **Motion** first.
            *   Then, draft the corresponding **S.D. Fla. Proposed Order**, mirroring the motion's relief and conforming to S.D. Fla style (use Local Form Order if applicable; include legal description for §522(f) order).

            ────────────────────────────────────────
            🏁 **STEP 5: REVIEW & DOWNLOAD**

            *   After generating the S.D. Fla. Motion and Proposed Order drafts, ask the user: "Please review the generated drafts for the Southern District of Florida. Do you require any modifications?"
            *   If modifications are requested, implement them accurately.
            *   If no modifications are needed, ask: "Would you like download links for the Motion and the Proposed Order as separate .docx files?"
            *   Provide the download links upon confirmation.
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
#                                    CHAT  
# ============================================================================

if page == "Chat":
    st.title("⚖️ Legal Motion Assistant")
    if not cfg["assistant_id"]:
        st.info("Create the assistant first from **Admin**.")
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

    # ─────Download─────
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])
            for fn, blob in h.get("files", []):
                st.download_button(f"Download {fn}", blob, fn, key=uk("dl_hist"))

    st.markdown("<div style='padding-bottom:70px'></div>", unsafe_allow_html=True)

    # ───── Upload + Chat input widgets ─────
    col_inp, col_up,= st.columns([5, 2])
    with col_up:
        uploaded = st.file_uploader(
            "💎", type=["pdf", "docx", "txt"], accept_multiple_files=True,
            key=st.session_state.uploader_key, label_visibility="collapsed"
        )

    first_turn_pending = not st.session_state.schedule_uploaded
    has_pdf = uploaded and any(f.name.lower().endswith(".pdf") for f in uploaded)
    prompt_disabled = first_turn_pending and not has_pdf

    if first_turn_pending:
        st.sidebar.warning(
            "📄 Please attach **at least one PDF bankruptcy schedule** before sending your first question."
        )

    with col_inp:
        user_prompt = st.chat_input("Ask or continue …", disabled=prompt_disabled)

    # ───── Handle new turn ─────
    if user_prompt:
        # (server-side guard)
        if not st.session_state.schedule_uploaded and not has_pdf:
            st.error("❗ You must upload at least one PDF schedule with your first message.")
            st.stop()

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
            if any(f.name.lower().endswith(".pdf") for f in uploaded):
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

        # system context
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
