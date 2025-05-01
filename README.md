# Legal Motion Assistant

A Streamlit-powered web application that streamlines drafting two common bankruptcy motions by combining Google Gemini’s document analysis with OpenAI Assistants’ drafting capabilities.

---

## 🚀 Overview

**Legal Motion Assistant** helps legal professionals generate first drafts of:

- **Motion to Value Secured Claim**  
- **Motion to Avoid Judicial Lien**

It features:

1. **Admin panel** for one-time setup and knowledge management  
2. **Interactive chat** for case-specific drafting  

---

## 🔑 Key Features

- **Dual AI Integration**  
  - **Google Gemini** for on-the-fly extraction of facts from uploaded documents  
  - **OpenAI Assistants API** for structured drafting, vector-store lookup, and interactive Q&A  

- **Motion-Specific Workflows**  
  - Tailored logic, prompts, and vector stores for each motion type  

- **Persistent Knowledge Base**  
  - Upload PDFs (statutes, case law, templates) into motion-specific vector stores  

- **On-the-Fly Document Analysis**  
  - Chat-upload PDFs, DOCX, or TXT files; instantly extract case facts (debtor name, creditor details, values, dates, etc.)  

- **Structured Drafting Process**  
  - AI asks for any missing variables, then drafts motion and proposed order  

- **Streaming & Downloads**  
  - Real-time response streaming  
  - Download generated .docx (or .txt) motions directly from the chat  

- **Workspace Controls**  
  - **Clear Chat** to start a new thread  
  - **Reset Workspace** to wipe configuration and start over  

---

## 🛠 Technology Stack

- **Frontend**: Streamlit  
- **AI Document Analysis**: Google Gemini API (Gemini 2.5 Pro Experimental)  
- **AI Drafting & Knowledge Retrieval**: OpenAI Assistants API (gpt-4.1)  
- **Vector Store**: OpenAI Vector Stores  

---

## ⚙️ Workflow

### 1. Admin Setup (one-time)

1. **Create Vector Stores**  
   - Click **Create 2 vector stores** in **Admin**  
   - Stores for “Value Secured Claim” and “Avoid Judicial Lien”  
   - IDs saved to `config2.json`

2. **Create Assistant**  
   - Click **Create Assistant**  
   - AI agent configured with motion-specific prompts, tools, and vector-store links  
   - Assistant ID saved to `config2.json`

3. **Upload Knowledge Base PDFs**  
   - Choose **Destination motion type** and optional **Jurisdiction**  
   - Upload one or more PDF files  
   - Files indexed into the matching vector store  

---

### 2. Chat Usage (drafting)

1. **Select Motion Type** in the sidebar (required)  
2. Optionally specify **Jurisdiction** and **Bankruptcy Chapter**  
3. **Start Conversation** by typing your request  
4. **Upload Case Documents** via the file-upload icon (PDF, DOCX, TXT)  
   - Google Gemini instantly extracts key facts and passes them to the Assistant  
5. **AI Interaction**  
   - Assistant confirms received facts  
   - Retrieves background from vector store  
   - Asks clarifying questions if needed  
   - Generates draft motion & proposed order  
6. **Review & Download**  
   - Review draft in-chat  
   - Download .docx via provided buttons  
7. **Clear or Reset** workspace as needed  

---

## 🗺️ How the Models Collaborate

- **Google Gemini**:  
  - On-demand fact extractor for uploaded documents  
  - Invisible to the user, yields structured variables  

- **OpenAI Assistant**:  
  - Primary conversationalist and drafter  
  - Uses `file_search` tool to query the vector stores  
  - Follows detailed instructions for each motion  

---

## 🖥️ Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/legal-motion-assistant.git
   cd legal-motion-assistant
