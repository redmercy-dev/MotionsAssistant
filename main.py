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
CFG_PATH = "config_tender.json"

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
        return {"vector_store_id": None}
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("vector_store_id", None)
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
GEM_MODEL = "gemini-2.5-flash"

# ────────── Password ──────────
def check_password():
    """
    Returns True if the user enters the correct password stored in Streamlit secrets.
    """
    def password_entered():
        if hmac.compare_digest(str(st.session_state["password"]), str(st.secrets["APP_PASSWORD"])):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Mot de passe", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Mot de passe incorrect")
    return False

if not check_password():
    st.stop()

# ────────── Gemini Functions ──────────
def get_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "application/octet-stream"

def gem_upload(path: str) -> gtypes.File:
    """Upload file to Gemini and return File object."""
    return g_client.files.upload(file=path)

def gem_extract(path: str, filename: str) -> tuple[str, bool]:
    """
    Uploads path to Gemini and asks it to extract relevant information for tender response.
    Returns: (extracted_text, is_rc_document)
    """
    gfile = gem_upload(path)

    prompt = f"""Tu es un assistant spécialisé dans l'analyse de documents d'appels d'offres français.

Analyse ce document ({filename}) et extrais TOUTES les informations structurées pertinentes:

1. **DÉTERMINE D'ABORD LE TYPE DE DOCUMENT:**
   - Est-ce un RC (Règlement de Consultation)? → Réponds "TYPE:RC" en première ligne
   - Est-ce un CCAP/CCTP? → Réponds "TYPE:CCAP_CCTP"
   - Est-ce un document de référence? → Réponds "TYPE:REFERENCE"
   - Sinon → Réponds "TYPE:OTHER"

2. Si c'est un RC (Règlement de Consultation):
   - **Objet du marché** (description complète)
   - **Type de marché** (construction neuve/rénovation/maintenance)
   - **Critères d'évaluation** avec pondération EXACTE (ex: "Méthodologie: 40%")
   - **Documents obligatoires à fournir** (liste exhaustive)
   - **Planning demandé** (OUI/NON - cherche "planning", "calendrier", "délais d'exécution")
   - **CVs demandés** (OUI/NON - liste les postes: conducteur travaux, HSE, etc.)
   - **Structure attendue du Mémoire Technique** (sections requises)
   - **Contraintes spécifiques** (site occupé, phasage, délais, accès, normes)
   - **Durée du marché** et dates clés
   - **Budget/montant** si indiqué

3. Si c'est un CCAP/CCTP:
   - Exigences techniques clés
   - Normes et certifications requises
   - Modalités d'exécution
   - Points de vigilance

4. Si c'est un document de référence (exemple, proposition concurrente):
   - Points forts identifiés
   - Structure utilisée
   - Arguments mis en avant

**FORMAT DE RÉPONSE:**
Commence TOUJOURS par "TYPE:XXX" sur la première ligne.
Puis réponds en JSON structuré OU en texte clair avec des sections bien définies et des bullets.

Si le document n'est pas pertinent, réponds: NO_RELEVANT_INFO"""

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
    text = (resp.text or "").strip()
    is_rc = text.startswith("TYPE:RC")
    return text, is_rc

# ────────── SYSTEM INSTRUCTIONS ──────────
SYSTEM_INSTRUCTIONS = """Role & Goal
Tu es un assistant IA aidant à générer des livrables de réponse à des appels d'offres français ("Mémoire Technique", "Planning" optionnel, "CVs" optionnels, analyse concurrentielle optionnelle) à partir de documents d'appel d'offres téléchargés par l'utilisateur (RC requis; CCAP/CCTP optionnels) et de documents de référence internes (exemples SEF/templates, propositions passées, etc.).

Règle fondamentale: Automatisation pilotée par le RC
- Le RC est la source de vérité pour ce qui doit être généré et comment cela doit être structuré.
- Tu dois automatiquement inférer les sections requises et les livrables conditionnels (planning, CVs) à partir du contenu du RC.
- L'utilisateur ne devrait PAS avoir besoin de définir manuellement la structure.

Politique RAG-first et anti-hallucination
- Consulte TOUJOURS les passages récupérés de la base de connaissances / documents téléchargés avant de faire des affirmations factuelles.
- N'invente JAMAIS de noms, certifications, contraintes de projet, quantités, délais ou exigences client.
- Si un détail requis est manquant, indique clairement ce qui manque et propose un format de placeholder (ex: "[À compléter: …]") et demande à l'utilisateur/admin de fournir l'information manquante.

Sorties supportées
Selon la "tâche" demandée par l'utilisateur, tu dois produire:
A) RC_INTERPRETATION_JSON
B) MEMOIRE_TECHNIQUE_DRAFT
C) PLANNING_SPEC (tâches de haut niveau + durées + dépendances)
D) CV_SPEC (données CV structurées par poste)
E) COMPETITOR_ANALYSIS_REPORT

Style de sortie
- Langue française, ton professionnel d'appel d'offres.
- Titres clairs, points bullet, sections structurées.
- Alignement explicite aux critères d'évaluation (si présents): les critères à fort poids doivent recevoir plus de détails.
- Évite le marketing vague; privilégie méthodologie concrète, organisation, contraintes, livrables, QA/HSE, et phasage.

A) Format RC_INTERPRETATION_JSON (strict)
Retourne un objet JSON avec:
{
  "business_type": "new_construction|renovation|maintenance|unknown",
  "evaluation_criteria": [{"name": "...", "weight": "..."}, ...],
  "mandatory_documents": ["...", ...],
  "planning_required": true|false,
  "cvs_required": true|false,
  "required_roles_for_cvs": ["...", ...],
  "required_sections": [{"title":"...", "purpose":"...", "key_points":["..."]}],
  "special_constraints": ["...", ...],
  "open_questions": ["...", ...]
}

B) Format MEMOIRE_TECHNIQUE_DRAFT
Retourne:
- Titre
- Table des matières (titres de sections)
- Puis chaque section avec:
  - Objectif de la section
  - Contenu (français, professionnel)
  - Bullets "Conformité RC" mappant: quelle exigence RC cela adresse (si connu)
- Termine avec une table "Checklist de conformité" (Exigence → Où traité → Statut: couvert/partiel/manquant)

C) Format PLANNING_SPEC
Retourne un objet JSON avec:
{
  "assumptions": ["..."],
  "calendar": {"work_days": "...", "constraints": ["..."]},
  "tasks": [
    {"id":"T1","name":"...","duration_days":X,"depends_on":["T0"],"notes":"..."},
    ...
  ],
  "milestones": [{"name":"...","day":X}]
}
Garde-le réaliste et cohérent avec les contraintes du RC (phasage, site occupé, limites d'accès, etc.). Si contraintes manquantes, mets des hypothèses explicitement.

D) Format CV_SPEC
Retourne un objet JSON avec:
{
  "roles": [
    {
      "role_name":"...",
      "required_by_rc": true,
      "candidates": [
        {
          "full_name":"(seulement si fourni dans les données; sinon placeholder)",
          "title":"...",
          "years_experience":"...",
          "key_projects":["..."],
          "certifications":["..."],
          "responsibilities_on_this_tender":["..."]
        }
      ]
    }
  ],
  "missing_employee_data": ["..."]
}
N'invente pas de personnes ou de certifications.

E) Format COMPETITOR_ANALYSIS_REPORT
Retourne:
- Résumé des forces/faiblesses concurrentes (basé uniquement sur les docs concurrents fournis)
- Points de comparaison vs approche SEF
- 10 améliorations actionnables pour la proposition SEF, alignées avec les critères d'évaluation
- Bullets "Preuves" référençant ce qui a été vu dans les docs concurrents (pas de citations >25 mots)

Comportement général
- Si on te demande de générer un livrable que le RC n'exige PAS (ex: planning quand le RC ne le demande pas), dis: "Le RC ne semble pas exiger X. Confirmez si vous souhaitez quand même le générer."
- Garde toujours les sorties structurées et prêtes pour le rendu de documents en aval (python-docx/reportlab).

IMPORTANT: Utilise TOUJOURS les passages récupérés via RAG avant de répondre. Base tes réponses sur les documents fournis."""

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
        st.warning(f"Impossible de télécharger le fichier {filename}: {str(e)}")
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
    """
    if content_obj is None:
        return ""
    if isinstance(content_obj, str):
        return content_obj.strip()

    if isinstance(content_obj, dict):
        t = _get(content_obj, "text")
        if isinstance(t, str) and t.strip():
            return t.strip()
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
                    maybe = item.get("text")
                    if isinstance(maybe, str) and maybe.strip():
                        texts.append(maybe.strip())
        return "\n\n".join([t for t in texts if t]).strip()

    return str(content_obj).strip()

# ────────── Extract container file outputs + retrieved chunks ──────────
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

    # 2) Retrieved chunks (from file_search_call.results)
    for output_item in output_items:
        if _get(output_item, "type") != "file_search_call":
            continue

        results = _get(output_item, "results")
        if results is None:
            results = _get(output_item, "search_results")

        for r in _as_list(results):
            file_id = _get(r, "file_id") or _get(r, "file")
            filename = _get(r, "filename")

            if not filename and file_id:
                try:
                    fobj = client.files.retrieve(file_id)
                    filename = getattr(fobj, "filename", None) or "Fichier inconnu"
                except Exception:
                    filename = "Fichier inconnu"

            text = (
                _get(r, "text")
                or _get(r, "chunk")
                or _content_to_text(_get(r, "content"))
                or _content_to_text(_get(r, "document"))
            )

            chunk = {
                "file_id": file_id,
                "filename": filename or "Fichier inconnu",
                "text": text or "",
                "score": _get(r, "score"),
                "rank": _get(r, "rank"),
            }
            if chunk["text"].strip():
                chunks.append(chunk)

    container_files = [
        fa for fa in container_files
        if fa.get("container_id") and fa.get("file_id") and fa.get("filename")
    ]
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
        return client.responses.retrieve(response_id)

def stream_response_with_file_search(
    conversation_history: List[Dict],
    vector_store_id: str,
    context: str
) -> Tuple[str, List[Tuple[str, bytes]], List[Dict]]:
    """
    Stream a response using Responses API with file_search and code_interpreter tools.

    Returns:
      (response_text, list_of_downloads, retrieved_chunks)
    """
    input_items = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "system", "content": context},
    ]
    input_items.extend(conversation_history)

    try:
        stream_response = client.responses.create(
            model="gpt-5-mini",
            input=input_items,
            tools=[
                {"type": "file_search", "vector_store_ids": [vector_store_id]},
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
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
                st.warning(f"Impossible de récupérer les fichiers/chunks de la réponse: {str(e)}")

        return full_text, files_to_download, retrieved_chunks

    except Exception as e:
        st.error(f"Erreur lors de la création de la réponse: {str(e)}")
        return "", [], []

# ────────── STREAMLIT UI STYLES ──────────
st.set_page_config("Générateur de Réponse d'Appel d'Offres", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="st-"] {
        font-family: 'Georgia', serif;
        color: #333;
    }
    body {background-color: #f0f2f6;}
    h1, h2, h3 {color: #0d1b4c; font-weight: bold;}

    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    span.material-symbols-rounded,
    span.material-symbols-outlined,
    i.material-icons,
    [data-testid^="chatAvatarIcon-"] span,
    [data-testid="stExpander"] summary span {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
        font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24 !important;
    }

    [data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
    }
    [data-testid="stExpander"] summary span {
        flex: 0 0 auto !important;
    }

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

    [data-testid="stChatMessageContent"] {
        background-color: transparent;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #e1e5f0 !important;
        border-left: 4px solid #0d1b4c;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #d1e7dd !important;
        border-right: 4px solid #198754;
    }

    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: transparent !important;
    }

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
if st.sidebar.button("🔄 Réinitialiser l'espace"):
    Path(CFG_PATH).unlink(missing_ok=True)
    cfg = {"vector_store_id": None}
    st.session_state.clear()
    st.sidebar.success("Espace effacé – ouvrez *Admin* pour recommencer.")
    st.rerun()

# ────────── CLEAR CHAT BUTTON ──────────
if page == "Chat" and st.sidebar.button("🗑️ Effacer le chat"):
    st.session_state.history = []
    st.sidebar.success("Historique du chat effacé.")
    st.rerun()

# ────────── SESSION INIT ──────────
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = uk("chat_files")

# ============================================================================
# ADMIN
# ============================================================================
if page == "Admin":
    st.title("⚙️ Panneau d'administration")

    # ➊ Create vector store
    if not cfg.get("vector_store_id") and st.button("Créer le vector store"):
        vs = client.vector_stores.create(name="tender_documents_store")
        cfg["vector_store_id"] = vs.id
        save_cfg(cfg)
        st.success(f"Vector store créé: {vs.id}")
        st.rerun()

    # ➋ Show vector store
    if cfg.get("vector_store_id"):
        st.subheader("Vector store")
        st.markdown(f"**ID:** `{cfg['vector_store_id']}`")

    # ➌ Upload documents into vector store
    if cfg.get("vector_store_id"):
        st.subheader("Télécharger des documents de référence")
        with st.form("upload_form", clear_on_submit=True):
            doc_type = st.selectbox(
                "Type de document",
                ["RC", "CCAP", "CCTP", "Exemple SEF", "Template", "Proposition concurrente", "Autre"]
            )
            files_ = st.file_uploader("Fichiers PDF", type="pdf", accept_multiple_files=True)
            submitted = st.form_submit_button("Télécharger")

            if submitted:
                if not files_:
                    st.error("Sélectionnez au moins un fichier PDF.")
                else:
                    with st.spinner("Téléchargement et indexation …"):
                        for f in files_:
                            file_obj = client.files.create(file=f, purpose="assistants")
                            client.vector_stores.files.create(
                                vector_store_id=cfg["vector_store_id"],
                                file_id=file_obj.id
                            )
                    st.success(f"{len(files_)} fichier(s) téléchargé(s) et indexé(s).")
                    st.rerun()

    # ➍ Display indexed documents
    if cfg.get("vector_store_id"):
        st.subheader("Documents indexés")
        try:
            vs_files = client.vector_stores.files.list(vector_store_id=cfg["vector_store_id"], limit=100)
            items = getattr(vs_files, "data", None) or vs_files
            
            rows = []
            for vf in items:
                file_id = getattr(vf, "file_id", None) or getattr(vf, "id", None)
                fname = "(inconnu)"
                if file_id:
                    try:
                        file_obj = client.files.retrieve(file_id)
                        fname = getattr(file_obj, "filename", None) or fname
                    except Exception:
                        pass
                rows.append({
                    "Fichier": fname,
                    "ID": file_id or "N/A",
                })
            
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Aucun document indexé pour le moment.")
        except Exception as e:
            st.warning(f"Impossible de lister les fichiers: {e}")

# ============================================================================
# CHAT
# ============================================================================
if page == "Chat":
    st.title("📋 Générateur de Réponse d'Appel d'Offres")

    if not cfg.get("vector_store_id"):
        st.error("Vector store non configuré. Veuillez aller dans Admin pour le créer.")
        st.stop()

    # ───── Task selector ─────
    task_type = st.sidebar.selectbox(
        "Type de sortie",
        [
            "💬 Discussion libre",
            "🔍 Interpréter le RC",
            "📄 Générer Mémoire Technique",
            "📅 Générer Planning",
            "👤 Générer CVs",
            "🔎 Analyser concurrence"
        ],
        key="task_type",
    )

    version = st.sidebar.selectbox("Version", ["V1", "V2", "V3"], key="version")

    # ───── DISPLAY HISTORY ─────
    for h in st.session_state.history:
        avatar = "🙂" if h["role"] == "user" else "🤖"
        with st.chat_message(h["role"], avatar=avatar):
            st.markdown(h["content"])

            for fn, blob in h.get("files", []):
                st.download_button(f"Télécharger {fn}", blob, fn, key=uk("dl_hist"))

            chunks = h.get("citations", [])
            if chunks and h["role"] == "assistant":
                with st.expander(f"📚 Voir {len(chunks)} passage(s) récupéré(s)", expanded=False):
                    for idx, c in enumerate(chunks, 1):
                        fname = c.get("filename", "Fichier inconnu")
                        score = c.get("score")
                        header = f"**Passage {idx}** (de {fname})"
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
            "📎 Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=st.session_state.uploader_key,
            label_visibility="collapsed"
        )

    with col_inp:
        user_prompt = st.chat_input("Posez votre question ou continuez …")

    # ───── Handle new turn ─────
    if user_prompt:
        extract_blocks, blobs_for_history = [], []
        rc_detected = False

        if uploaded:
            prog = st.progress(0.0)
            for i, uf in enumerate(uploaded, 1):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name

                with st.spinner(f"Gemini analyse {uf.name} …"):
                    gem_text, is_rc = gem_extract(tmp_path, uf.name)
                    if gem_text and gem_text not in ("NO_RELEVANT_INFO", "NO_RELEVANT_INFO_FOUND_IN_UPLOAD"):
                        extract_blocks.append(f"EXTRACTED_FROM_UPLOAD Nom du fichier ({uf.name}):\n{gem_text}")
                        if is_rc:
                            rc_detected = True
                    blobs_for_history.append((uf.name, uf.getvalue()))

                prog.progress(i / len(uploaded))

                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            prog.empty()

        # AUTO-WORKFLOW: If RC detected, override user prompt with structured workflow
        if rc_detected and not user_prompt.strip().lower().startswith(("ne génère pas", "attends", "stop")):
            user_prompt = """🚀 RC DÉTECTÉ - LANCEMENT DU WORKFLOW AUTOMATIQUE

Je vais maintenant exécuter le workflow complet:

**ÉTAPE 1: INTERPRÉTATION DU RC**
Analyse le RC extrait ci-dessus et génère l'interprétation structurée (RC_INTERPRETATION_JSON).
Identifie automatiquement:
- Les critères d'évaluation et leur poids
- Les documents obligatoires
- Si un planning est requis
- Si des CVs sont requis (et pour quels postes)
- La structure attendue du Mémoire Technique
- Les contraintes spécifiques

**ÉTAPE 2: PLAN DE DOCUMENT**
Propose un plan détaillé du Mémoire Technique avec:
- Sections alignées sur les critères d'évaluation
- Sous-sections couvrant toutes les exigences RC
- Mapping: chaque exigence RC → section qui la traite

**ÉTAPE 3: GÉNÉRATION DES SECTIONS**
Pour chaque section du plan:
- Génère le contenu professionnel en français
- Utilise RAG pour récupérer informations pertinentes
- Aligne sur les critères à fort coefficient
- Indique conformité RC

**ÉTAPE 4: CHECKLIST DE CONFORMITÉ**
Crée un tableau récapitulatif:
| Exigence RC | Section traitant | Statut |

**ÉTAPE 5: LIVRABLES CONDITIONNELS**
- Si planning requis → Propose PLANNING_SPEC
- Si CVs requis → Propose CV_SPEC avec postes identifiés

Après avoir tout généré, demande-moi si je veux:
- Générer les fichiers DOCX/PDF
- Faire des ajustements
- Générer une V2 améliorée

**COMMENCE MAINTENANT L'ÉTAPE 1 (Interprétation du RC):**"""

        # Store user turn
        st.session_state.history.append(
            {"role": "user", "content": user_prompt, "files": blobs_for_history}
        )

        with st.chat_message("user", avatar="🙂"):
            st.markdown(user_prompt)
            for fn, blob in blobs_for_history:
                st.download_button(f"Télécharger {fn}", blob, fn, key=uk("dl_user"))

        # Build context with task-specific instructions
        task_instructions = {
            "💬 Discussion libre": "Réponds aux questions de l'utilisateur en utilisant les documents disponibles via RAG. Sois précis et cite tes sources.",
            
            "🔍 Interpréter le RC": """TÂCHE SPÉCIFIQUE: Analyse le RC et retourne un JSON strictement conforme au format RC_INTERPRETATION_JSON.
Tu DOIS extraire:
- business_type (type de marché)
- evaluation_criteria avec poids EXACTS (cherche les pourcentages ou points)
- mandatory_documents (tous les documents demandés)
- planning_required (true si le RC demande un planning/calendrier/délais d'exécution)
- cvs_required (true si le RC demande des CV d'équipe) 
- required_roles_for_cvs si applicable (conducteur travaux, HSE, chef chantier, etc.)
- required_sections (structure attendue du Mémoire Technique)
- special_constraints (site occupé, phasage, normes, accès limités)
- open_questions (informations manquantes)

Utilise UNIQUEMENT les informations du RC récupérées via RAG. N'invente rien.""",

            "📄 Générer Mémoire Technique": f"""TÂCHE SPÉCIFIQUE: Génère un Mémoire Technique complet en français (VERSION {version}).

SI C'EST LA PREMIÈRE GÉNÉRATION APRÈS UPLOAD DU RC:
Exécute automatiquement le WORKFLOW COMPLET (5 étapes) sans attendre confirmation.

Format MEMOIRE_TECHNIQUE_DRAFT:
1. Titre professionnel du projet
2. Table des matières détaillée
3. Pour CHAQUE section:
   - **Titre de la section**
   - **Objectif** (2-3 phrases)
   - **Contenu** (3-5 paragraphes MINIMUM, professionnel, concret)
     * Méthodologie détaillée
     * Organisation et moyens
     * Gestion des contraintes identifiées dans le RC
     * Points de vigilance et solutions
   - **Conformité RC** (bullets listant les exigences RC couvertes)
4. **Checklist de conformité finale** (tableau):
   | Exigence RC | Section(s) traitant | Statut |

IMPORTANT:
- Utilise RAG pour récupérer: contraintes RC, critères d'évaluation, exigences techniques
- Priorise les critères à FORT COEFFICIENT (40%+) → 2-3x plus de contenu
- Sois CONCRET: pas de généralités, mais des méthodologies applicables
- Mentionne outils, processus, normes, équipements spécifiques
- Si {version} = V2 ou V3: améliore/détaille vs version précédente selon feedback utilisateur
- N'invente PAS de certifications, références projets, ou données techniques non documentées
- Chaque section doit faire AU MINIMUM 3 paragraphes substantiels""",

            "📅 Générer Planning": """TÂCHE SPÉCIFIQUE: Génère un planning au format PLANNING_SPEC (JSON).
Structure requise:
{
  "assumptions": ["Base sur 5j/semaine", "Équipe de X personnes", etc.],
  "calendar": {
    "work_days": "Lundi-Vendredi", 
    "constraints": ["Site occupé", "Accès limités 8h-17h", etc.]
  },
  "tasks": [
    {
      "id": "T1",
      "name": "Installation de chantier",
      "duration_days": 5,
      "depends_on": [],
      "notes": "Inclut clôtures, base vie, raccordements"
    },
    ...
  ],
  "milestones": [
    {"name": "Début travaux", "day": 0},
    {"name": "Réception", "day": 120}
  ]
}

IMPORTANT:
- Utilise RAG pour récupérer: durée marché, contraintes phasage, délais RC
- Si contraintes manquantes → liste explicitement dans assumptions
- Découpe en 15-25 tâches réalistes avec dépendances logiques
- Durées cohérentes avec type de marché (construction: mois, maintenance: jours)
- Ce JSON servira à générer Gantt PNG + Excel + PDF automatiquement""",

            "👤 Générer CVs": """TÂCHE SPÉCIFIQUE: Génère des CVs structurés au format CV_SPEC (JSON).
Structure requise:
{
  "roles": [
    {
      "role_name": "Conducteur de travaux",
      "required_by_rc": true,
      "candidates": [
        {
          "full_name": "[À compléter: nom prénom du collaborateur]",
          "title": "Conducteur de travaux TCE",
          "years_experience": "[À compléter: X années]",
          "key_projects": [
            "[À compléter: Projet similaire 1]",
            "[À compléter: Projet similaire 2]"
          ],
          "certifications": [
            "[À compléter: ex. CACES, habilitations]"
          ],
          "responsibilities_on_this_tender": [
            "Coordination des corps d'état",
            "Suivi planning et budget",
            "Interface client quotidienne"
          ]
        }
      ]
    }
  ],
  "missing_employee_data": [
    "Identités et parcours des collaborateurs",
    "Certifications et habilitations à jour",
    "Références projets similaires détaillées"
  ]
}

CRITIQUE: N'INVENTE JAMAIS de noms, certifications ou expériences. 
Si données RH manquantes, utilise "[À compléter: ...]" et liste dans missing_employee_data.
Propose des responsabilités réalistes pour le marché concerné.""",

            "🔎 Analyser concurrence": """TÂCHE SPÉCIFIQUE: Analyse les propositions concurrentes au format COMPETITOR_ANALYSIS_REPORT.

Structure:
1. **Résumé forces/faiblesses** de chaque concurrent (basé UNIQUEMENT sur docs fournis)
2. **Tableau comparatif** (notre approche vs concurrents):
   | Critère RC | Concurrent A | Concurrent B | Notre approche proposée |
3. **10 améliorations actionnables** pour notre offre:
   - Alignées sur critères RC à fort coefficient
   - Différenciantes vs concurrence
   - Concrètes et réalisables
4. **Preuves documentées** (citations <25 mots des docs concurrents)

IMPORTANT:
- Utilise RAG pour récupérer passages des propositions concurrentes
- Compare: méthodologie, arguments, présentation, points forts
- Identifie GAPS (ce qu'ils n'ont pas couvert)
- Propose améliorations DIFFÉRENCIANTES et réalistes
- N'invente PAS de contenu concurrent non documenté
- Focus sur critères à fort coefficient pour maximiser notation"""
        }
        
        context_parts = [
            f"VERSION: {version}",
            task_instructions.get(task_type, f"Type de tâche demandée: {task_type}"),
        ]
        if extract_blocks:
            context_parts.append("\n".join(extract_blocks))

        context = "\n".join(context_parts)

        # Prepare conversation for API
        conversation_history = [{"role": hh["role"], "content": hh["content"]} for hh in st.session_state.history]

        # Get response from API
        with st.chat_message("assistant", avatar="🤖"):
            answer, new_files, chunks = stream_response_with_file_search(
                conversation_history,
                cfg["vector_store_id"],
                context
            )

            for fn, data in new_files:
                st.download_button(f"Télécharger {fn}", data, fn, key=uk("dl_asst"))

            if chunks:
                with st.expander(f"📚 Voir {len(chunks)} passage(s) récupéré(s)", expanded=False):
                    for idx, c in enumerate(chunks, 1):
                        fname = c.get("filename", "Fichier inconnu")
                        score = c.get("score")
                        header = f"**Passage {idx}** (de {fname})"
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
