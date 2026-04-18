# -*- coding: utf-8 -*-
"""
=============================================================================
  مساعد قانوني تونسي | Assistant Juridique Tunisien
  Complete self-contained Jupyter Notebook version
  Run each cell in order — no external imports needed
=============================================================================
"""

# =============================================================================
# CELL 1 — INSTALL DEPENDENCIES
# Run this cell once, then restart the kernel before running the rest
# =============================================================================

import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install",
    "langchain",
    "langchain-community",
    "chromadb",
    "pdfplumber",
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
    "pandas",
    "torch",
    "accelerate",
    "sentencepiece",
    "ipywidgets",
], check=True)

print("✅ All dependencies installed. Now restart the kernel and run from Cell 2.")

# =============================================================================
# CELL 2 — IMPORTS & CONFIGURATION
# =============================================================================

import os
import re
import warnings
import unicodedata
import pandas as pd
import pdfplumber
import torch
warnings.filterwarnings("ignore")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import InferenceClient

# --- Paths (relative to this notebook file) ---
LAWS_FOLDER     = "./laws/"
CHROMA_DB_PATH  = "./chroma_db/"
CONTACTS_CSV    = "./law_contacts.csv"

# --- Models ---
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PRIMARY_MODEL   = "google/flan-t5-xl"
FALLBACK_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_KEY      = "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"

# --- RAG settings ---
TOP_K_CHUNKS    = 5
MIN_RELEVANCE   = 0.45

# --- Global state ---
embeddings             = None
vectorstore            = None
flan_model             = None
flan_tokenizer         = None
hf_client              = None
PRIMARY_LLM_AVAILABLE  = False

# Create folders
os.makedirs(LAWS_FOLDER,    exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

print("✅ Configuration ready.")
print(f"   Laws folder  : {os.path.abspath(LAWS_FOLDER)}")
print(f"   ChromaDB     : {os.path.abspath(CHROMA_DB_PATH)}")
print(f"   Contacts CSV : {os.path.abspath(CONTACTS_CSV)}")

# =============================================================================
# CELL 3 — LANGUAGE DETECTION
# =============================================================================

def detect_language(text: str) -> str:
    """
    Returns 'ar', 'fr', or 'mixed' based on character ratios.
    Arabic chars > 60%  → ar
    Latin  chars > 60%  → fr
    Neither             → mixed (bilingual response will be given)
    """
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    latin  = len(re.findall(r'[a-zA-ZÀ-ÿ]',    text))
    total  = arabic + latin

    if total == 0:
        return "fr"

    if arabic / total > 0.60:
        return "ar"
    elif latin / total > 0.60:
        return "fr"
    else:
        return "mixed"

# Sanity checks
assert detect_language("ما هو قانون الشغل التونسي؟")          == "ar"
assert detect_language("Qu'est-ce que le droit du travail?")   == "fr"
assert detect_language("ما هو le droit du travail في تونس؟")   == "mixed"

print("✅ Language detection ready.")

# =============================================================================
# CELL 4 — CONTACTS CSV (auto-created with sample data if missing)
# =============================================================================

def ensure_contacts_csv():
    """Creates a sample contacts CSV if it doesn't already exist."""
    if os.path.exists(CONTACTS_CSV):
        print(f"✅ Contacts CSV found: {CONTACTS_CSV}")
        return

    sample = pd.DataFrame([
        {
            "name":           "محمد بن علي",
            "specialization": "قانون الشغل عقود عمل",
            "phone":          "+216 20 000 001",
            "email":          "m.benali@avocat.tn",
            "city":           "تونس",
            "languages":      "ar,fr"
        },
        {
            "name":           "Sonia Trabelsi",
            "specialization": "droit commercial sociétés contrats",
            "phone":          "+216 20 000 002",
            "email":          "s.trabelsi@avocat.tn",
            "city":           "Sfax",
            "languages":      "fr,ar"
        },
        {
            "name":           "Karim Gharbi",
            "specialization": "droit de la famille divorce héritage",
            "phone":          "+216 20 000 003",
            "email":          "k.gharbi@avocat.tn",
            "city":           "Sousse",
            "languages":      "fr"
        },
        {
            "name":           "فاطمة الزهراء المنصوري",
            "specialization": "قانون الأسرة إرث طلاق حضانة",
            "phone":          "+216 20 000 004",
            "email":          "f.mansouri@avocat.tn",
            "city":           "صفاقس",
            "languages":      "ar"
        },
        {
            "name":           "Amine Chaabane",
            "specialization": "droit pénal criminel infractions",
            "phone":          "+216 20 000 005",
            "email":          "a.chaabane@avocat.tn",
            "city":           "Tunis",
            "languages":      "fr,ar"
        },
        {
            "name":           "نور الهدى بوعزيز",
            "specialization": "قانون الاستثمار والتجارة الدولية",
            "phone":          "+216 20 000 006",
            "email":          "n.bouaziz@avocat.tn",
            "city":           "تونس",
            "languages":      "ar,fr,en"
        },
    ])
    sample.to_csv(CONTACTS_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Sample contacts CSV created at: {CONTACTS_CSV}")
    print("   → Replace with real lawyer data before going live.")


ensure_contacts_csv()

# =============================================================================
# CELL 5 — CONTACTS FALLBACK FUNCTION
# =============================================================================

def find_relevant_contacts(question: str, lang: str) -> str:
    """
    Called when the answer is not found in PDFs.
    Searches the contacts CSV for professionals matching the question topic.
    Returns a formatted string in the user's language.
    """
    try:
        df = pd.read_csv(CONTACTS_CSV, encoding="utf-8-sig")
    except Exception as e:
        return f"⚠️ Could not load contacts file: {e}"

    def strip_diacritics(text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", str(text))
            if unicodedata.category(c) != "Mn"
        ).lower()

    # Extract keywords from question (words longer than 3 characters)
    q_clean  = strip_diacritics(question)
    keywords = [w for w in re.split(r'\s+', q_clean) if len(w) > 3]

    # Score contacts by how many question keywords appear in their specialization
    def score(specialization):
        s = strip_diacritics(specialization)
        return sum(1 for kw in keywords if kw in s)

    df["_score"] = df["specialization"].apply(score)
    matched      = df[df["_score"] > 0].sort_values("_score", ascending=False).head(3)

    # If no keyword match → show all contacts sorted by city
    if matched.empty:
        matched = df.sort_values("city").head(3)

    def format_contact(row):
        return (
            f"👤 {row['name']}  —  {row['specialization']}\n"
            f"   📍 {row['city']}   |   📞 {row['phone']}   |   ✉️  {row['email']}"
        )

    contacts_text = "\n\n".join(matched.apply(format_contact, axis=1).tolist())

    if lang == "ar":
        return (
            "لم نجد إجابة في النصوص القانونية المتاحة.\n"
            "إليك متخصصون قانونيون يمكنهم مساعدتك:\n\n"
            + contacts_text
        )
    elif lang == "fr":
        return (
            "Réponse introuvable dans les textes juridiques disponibles.\n"
            "Voici des professionnels juridiques qui peuvent vous aider :\n\n"
            + contacts_text
        )
    else:  # mixed — bilingual output
        return (
            "لم نجد إجابة في النصوص القانونية المتاحة.\n"
            "إليك متخصصون قانونيون يمكنهم مساعدتك:\n\n"
            + contacts_text
            + "\n\n---\n\n"
            "Réponse introuvable dans les textes juridiques disponibles.\n"
            "Voici des professionnels juridiques qui peuvent vous aider :\n\n"
            + contacts_text
        )


print("✅ Contacts fallback function ready.")

# =============================================================================
# CELL 6 — EMBEDDING MODEL
# =============================================================================

print("⏳ Loading multilingual embedding model...")
print("   (This downloads ~120MB on first run — please wait)")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("✅ Embedding model loaded.")

# =============================================================================
# CELL 7 — PDF INGESTION & CHROMADB
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Extracts text from a PDF page by page using pdfplumber.
    pdfplumber handles Arabic RTL layout much better than PyPDF2.
    Returns a list of LangChain Document objects.
    """
    docs     = []
    filename = os.path.basename(pdf_path)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(
                        page_content=text.strip(),
                        metadata={"source": filename, "page": page_num}
                    ))
    except Exception as e:
        print(f"  ⚠️  Error reading {filename}: {e}")
    return docs


def ingest_pdfs():
    """
    Main ingestion pipeline:
      1. Scans ./laws/ for PDFs
      2. Extracts text with pdfplumber
      3. Splits into overlapping chunks
      4. Embeds and stores in ChromaDB
    Skips if ChromaDB already exists (re-run only if you add new PDFs).
    """
    global vectorstore

    # Load existing DB if available
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print("✅ ChromaDB already exists — loading existing index...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        count = vectorstore._collection.count()
        print(f"   → {count} chunks loaded and ready.")
        return

    # Scan for PDFs
    pdf_files = [f for f in os.listdir(LAWS_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️  No PDFs found in ./laws/")
        print(f"   → Please copy your Tunisian law PDFs into: {os.path.abspath(LAWS_FOLDER)}")
        vectorstore = None
        return

    print(f"📂 Found {len(pdf_files)} PDF(s): {pdf_files}\n")

    # Extract text
    all_docs = []
    for pdf_file in pdf_files:
        path = os.path.join(LAWS_FOLDER, pdf_file)
        print(f"  Reading: {pdf_file} ...", end=" ")
        docs = extract_text_from_pdf(path)
        all_docs.extend(docs)
        print(f"{len(docs)} pages extracted.")

    if not all_docs:
        print("⚠️  No text could be extracted. Check if PDFs are scanned images.")
        print("   → If so, you need OCR (e.g. pytesseract) before ingestion.")
        vectorstore = None
        return

    # Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        # Arabic comma (،) added as a separator alongside standard ones
        separators=["\n\n", "\n", ".", "،", "?", "؟", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"\n✂️  Created {len(chunks)} chunks from {len(all_docs)} pages.")

    # Embed and store
    print("⏳ Embedding chunks and saving to ChromaDB...")
    print("   (This may take several minutes on first run)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist()
    print(f"✅ Ingestion complete — {len(chunks)} chunks stored in ChromaDB.")


ingest_pdfs()

# =============================================================================
# CELL 8 — PRIMARY LLM: google/flan-t5-xl (local)
# =============================================================================

def load_primary_llm():
    """
    Loads google/flan-t5-xl locally.
    ~3GB model — takes 2-5 min on first download, ~1 min after caching.
    Sets PRIMARY_LLM_AVAILABLE = True on success.
    """
    global flan_model, flan_tokenizer, PRIMARY_LLM_AVAILABLE

    try:
        print(f"⏳ Loading {PRIMARY_MODEL}...")
        print("   (Downloads ~3GB on first run — please be patient)")
        flan_tokenizer = T5Tokenizer.from_pretrained(PRIMARY_MODEL)
        flan_model     = T5ForConditionalGeneration.from_pretrained(
            PRIMARY_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        flan_model.eval()
        PRIMARY_LLM_AVAILABLE = True
        print(f"✅ {PRIMARY_MODEL} loaded and ready.")

    except Exception as e:
        print(f"⚠️  Could not load {PRIMARY_MODEL}: {e}")
        print("   → Switching to HuggingFace API fallback (Mistral).")
        PRIMARY_LLM_AVAILABLE = False


def generate_with_primary(prompt: str) -> str:
    """Runs inference on local flan-t5-xl."""
    inputs = flan_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    with torch.no_grad():
        outputs = flan_model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
            do_sample=True,
            top_p=0.92,
            temperature=0.3,        # low = more factual, less creative
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


load_primary_llm()

# =============================================================================
# CELL 9 — FALLBACK LLM: Mistral via HuggingFace Inference API
# =============================================================================

def load_fallback_llm():
    """
    Initializes the HuggingFace Inference API client.
    Used automatically when flan-t5-xl is unavailable or fails.
    """
    global hf_client
    try:
        hf_client = InferenceClient(model=FALLBACK_MODEL, token=HF_API_KEY)
        print("✅ HuggingFace fallback LLM (Mistral) client ready.")
    except Exception as e:
        print(f"⚠️  Could not initialize HF fallback client: {e}")
        hf_client = None


def generate_with_fallback(prompt: str) -> str:
    """Calls Mistral via HuggingFace Inference API."""
    if hf_client is None:
        return "ANSWER_NOT_FOUND"
    try:
        return hf_client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.3,
            repetition_penalty=1.1,
        ).strip()
    except Exception as e:
        print(f"  ⚠️ Fallback LLM error: {e}")
        return "ANSWER_NOT_FOUND"


load_fallback_llm()

# =============================================================================
# CELL 10 — UNIFIED LLM CALLER (primary → fallback chain)
# =============================================================================

def call_llm(prompt: str) -> str:
    """
    Tries flan-t5-xl (primary) first.
    If it's unavailable, crashed, or returned empty → falls back to Mistral API.
    """
    if PRIMARY_LLM_AVAILABLE and flan_model is not None:
        try:
            print("   [LLM] Using google/flan-t5-xl (primary)...")
            result = generate_with_primary(prompt)
            if result and len(result.strip()) > 5:
                return result
            print("   [LLM] Primary returned empty — switching to fallback...")
        except Exception as e:
            print(f"   [LLM] flan-t5-xl failed ({e}) — switching to fallback...")

    print("   [LLM] Using Mistral via HuggingFace API (fallback)...")
    return generate_with_fallback(prompt)


print("✅ LLM caller chain ready.")
print(f"   Primary  : {'✅ flan-t5-xl (local)' if PRIMARY_LLM_AVAILABLE else '❌ unavailable'}")
print(f"   Fallback : {'✅ Mistral API' if hf_client else '❌ unavailable'}")

# =============================================================================
# CELL 11 — PROMPT BUILDER
# =============================================================================

def build_prompt(question: str, context: str, lang: str) -> str:
    """
    Builds language-appropriate prompt for the LLM.
    Instructs the model to answer ONLY from the provided PDF context.
    """
    if lang == "ar":
        return (
            "أنت مساعد قانوني تونسي متخصص. أجب بالعربية التونسية فقط.\n"
            "استخدم فقط النصوص القانونية التالية للإجابة. لا تستخدم أي معرفة خارجية.\n"
            "إذا لم تجد الإجابة في هذه النصوص، اكتب فقط الكلمة: ANSWER_NOT_FOUND\n"
            "اذكر دائماً المصدر هكذا: [اسم_الملف.pdf، صفحة رقم X]\n\n"
            f"النصوص القانونية:\n{context}\n\n"
            f"السؤال: {question}\n\n"
            "الإجابة:"
        )

    elif lang == "fr":
        return (
            "Tu es un assistant juridique tunisien spécialisé. Réponds UNIQUEMENT en français.\n"
            "Utilise SEULEMENT les textes juridiques ci-dessous. Aucune connaissance externe.\n"
            "Si tu ne trouves pas la réponse, écris uniquement: ANSWER_NOT_FOUND\n"
            "Cite toujours la source: [nom_fichier.pdf, page X]\n\n"
            f"Textes juridiques:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Réponse:"
        )

    else:  # mixed — bilingual answer requested
        return (
            "You are a Tunisian legal assistant. The user is writing in both Arabic and French.\n"
            "Reply FIRST in Tunisian Arabic, then repeat the SAME answer in French, separated by '---'.\n"
            "Use ONLY the legal texts below. No outside knowledge whatsoever.\n"
            "If not found, write only: ANSWER_NOT_FOUND\n"
            "Always cite source like this: [filename.pdf, page X]\n\n"
            f"Legal texts:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (Arabic first, then --- then French):"
        )


print("✅ Prompt builder ready.")

# =============================================================================
# CELL 12 — MAIN ask() FUNCTION
# =============================================================================

def ask(question: str) -> dict:
    """
    Full RAG pipeline. Returns a dict:
    {
      "answer" : str,              — the response text
      "lang"   : "ar"|"fr"|"mixed",
      "source" : "llm"|"contacts"|"error"
    }

    Flow:
      1. Detect language
      2. Search ChromaDB for relevant chunks
      3. If best score < MIN_RELEVANCE → contacts fallback
      4. Build prompt → call LLM (flan-t5 → Mistral)
      5. If LLM says ANSWER_NOT_FOUND → contacts fallback
    """
    if vectorstore is None:
        return {
            "answer": (
                "⚠️ لم يتم فهرسة أي ملفات PDF بعد.\n"
                "⚠️ Aucun PDF indexé. Ajoutez vos PDFs dans ./laws/ et relancez la cellule d'ingestion."
            ),
            "lang": "fr",
            "source": "error"
        }

    # Step 1 — Language detection
    lang = detect_language(question)
    print(f"   [Lang] Detected: {lang}")

    # Step 2 — Retrieve relevant chunks with similarity scores
    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            question, k=TOP_K_CHUNKS
        )
    except Exception as e:
        return {"answer": f"⚠️ Retrieval error: {e}", "lang": lang, "source": "error"}

    if not results:
        print("   [RAG] No results found → contacts fallback.")
        return {"answer": find_relevant_contacts(question, lang), "lang": lang, "source": "contacts"}

    # Step 3 — Check relevance score threshold
    best_score = max(score for _, score in results)
    print(f"   [RAG] Best relevance score: {best_score:.3f}  (min threshold: {MIN_RELEVANCE})")

    if best_score < MIN_RELEVANCE:
        print("   [RAG] Score below threshold → contacts fallback.")
        return {"answer": find_relevant_contacts(question, lang), "lang": lang, "source": "contacts"}

    # Step 4 — Build context string from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content}"
        for doc, _ in results
    )

    # Step 5 — Build prompt and call LLM
    prompt   = build_prompt(question, context, lang)
    response = call_llm(prompt).strip()

    # Step 6 — Handle "I don't know" response from LLM
    if "ANSWER_NOT_FOUND" in response.upper():
        print("   [LLM] Model returned ANSWER_NOT_FOUND → contacts fallback.")
        return {"answer": find_relevant_contacts(question, lang), "lang": lang, "source": "contacts"}

    return {"answer": response, "lang": lang, "source": "llm"}


print("✅ ask() function ready.")

# =============================================================================
# CELL 13 — INTERACTIVE CHAT WIDGET
# =============================================================================

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Inject CSS styles
display(HTML("""
<style>
  .chat-outer {
    max-width: 740px;
    margin: 0 auto;
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  .chat-header {
    background: #1a5276;
    color: white;
    padding: 14px 20px;
    border-radius: 12px 12px 0 0;
    font-size: 15px;
    font-weight: 600;
    text-align: center;
  }
  .chat-header span {
    font-size: 12px;
    opacity: 0.75;
    display: block;
    margin-top: 3px;
    font-weight: 400;
  }
  .msg-user {
    background: #dcf8c6;
    padding: 10px 15px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    margin: 8px 0 8px auto;
    font-size: 14px;
    direction: rtl;
    text-align: right;
    word-break: break-word;
  }
  .msg-bot {
    background: #f1f0f0;
    padding: 10px 15px;
    border-radius: 18px 18px 18px 4px;
    max-width: 85%;
    margin: 8px 0;
    font-size: 14px;
    direction: rtl;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg-contacts {
    background: #fff8e1;
    border-left: 4px solid #ffa000;
    padding: 10px 15px;
    border-radius: 4px;
    max-width: 90%;
    margin: 8px 0;
    font-size: 13px;
    white-space: pre-wrap;
  }
  .msg-error {
    background: #ffebee;
    border-left: 4px solid #e53935;
    padding: 10px 15px;
    border-radius: 4px;
    max-width: 90%;
    margin: 8px 0;
    font-size: 13px;
  }
  .lang-tag {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    display: inline-block;
    margin-bottom: 3px;
  }
  .lang-ar    { background: #e8f5e9; color: #2e7d32; }
  .lang-fr    { background: #e3f2fd; color: #1565c0; }
  .lang-mixed { background: #f3e5f5; color: #6a1b9a; }
  .src-tag {
    font-size: 10px;
    color: #aaa;
    margin-top: 4px;
    padding-left: 4px;
  }
</style>
"""))

# --- Widgets ---
chat_output = widgets.Output(
    layout=widgets.Layout(
        height="420px",
        overflow_y="auto",
        border="1px solid #ddd",
        padding="10px",
        background_color="white"
    )
)

text_box = widgets.Text(
    placeholder="اكتب سؤالك / Posez votre question...",
    layout=widgets.Layout(width="75%", height="38px")
)

send_btn = widgets.Button(
    description="إرسال ➤",
    button_style="primary",
    layout=widgets.Layout(width="12%", height="38px")
)

clear_btn = widgets.Button(
    description="مسح 🗑",
    button_style="warning",
    layout=widgets.Layout(width="10%", height="38px")
)

status_lbl = widgets.Label(value="✅ جاهز | Prêt")

input_row = widgets.HBox(
    [text_box, send_btn, clear_btn],
    layout=widgets.Layout(gap="8px", margin_top="6px")
)

lang_display = {
    "ar":    ("🇹🇳 عربي",    "lang-ar"),
    "fr":    ("🇫🇷 Français", "lang-fr"),
    "mixed": ("🌐 Mélangé",  "lang-mixed"),
}

source_display = {
    "llm":      "📚 من النصوص القانونية | Depuis les textes PDF",
    "contacts": "📋 من الدليل المهني | Depuis l'annuaire",
    "error":    "⚠️ خطأ | Erreur",
}


def on_send(_):
    question = text_box.value.strip()
    if not question:
        return

    text_box.value       = ""
    send_btn.disabled    = True
    status_lbl.value     = "⏳ جاري البحث... Recherche en cours..."

    # Show user message
    with chat_output:
        display(HTML(f'<div class="msg-user">{question}</div>'))

    # Call the RAG pipeline
    result  = ask(question)
    answer  = result["answer"]
    lang    = result["lang"]
    source  = result["source"]

    lang_text, lang_cls = lang_display.get(lang, ("🌐", "lang-mixed"))
    src_text            = source_display.get(source, "")

    bubble_cls = (
        "msg-contacts" if source == "contacts" else
        "msg-error"    if source == "error"    else
        "msg-bot"
    )

    with chat_output:
        display(HTML(
            f'<span class="lang-tag {lang_cls}">{lang_text}</span><br>'
            f'<div class="{bubble_cls}">{answer}</div>'
            f'<div class="src-tag">{src_text}</div>'
            f'<hr style="border:none;border-top:1px solid #eee;margin:6px 0">'
        ))

    send_btn.disabled = False
    status_lbl.value  = "✅ جاهز | Prêt"


def on_clear(_):
    with chat_output:
        clear_output()
    status_lbl.value = "✅ جاهز | Prêt"


send_btn.on_click(on_send)
clear_btn.on_click(on_clear)
text_box.on_submit(on_send)  # Enter key also submits

# --- Render everything ---
display(HTML('<div class="chat-header">🏛️ مساعد قانوني تونسي | Assistant Juridique Tunisien<span>اكتب بالعربية أو الفرنسية أو بكليهما | Écrivez en arabe, français ou les deux</span></div>'))
display(chat_output)
display(input_row)
display(status_lbl)

# =============================================================================
# CELL 14 — (OPTIONAL) Single question — programmatic use
# =============================================================================

# Uncomment and run this cell to ask a single question without the widget:

# question = "ما هي حقوق العامل في قانون الشغل التونسي؟"
# result   = ask(question)
# print("Language:", result["lang"])
# print("Source  :", result["source"])
# print("Answer  :\n", result["answer"])