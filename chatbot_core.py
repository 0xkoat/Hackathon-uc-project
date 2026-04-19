# -*- coding: utf-8 -*-
"""
chatbot_core.py  — FIXED v4 (FINAL)
-------------------------------------
Root cause of all failures:
  - hf-inference provider no longer supports big LLMs (Mistral, Zephyr, etc.)
  - Raw HTTP calls to old/new URLs all fail for chat models

Solution:
  - Use huggingface_hub InferenceClient which auto-routes to the correct provider
  - Use novita provider (free, no login required, supports many models)
  - Fallback chain across multiple providers and models
"""

import os
import re
import unicodedata
import warnings
import pandas as pd
import pdfplumber

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

LAWS_FOLDER    = "./laws/"
CHROMA_DB_PATH = "./chroma_db/"
CONTACTS_CSV   = "./law_contacts.csv"
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

HF_API_KEY = "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"

# Each entry is (provider, model)
# novita and featherless-ai are free and don't require model-specific agreements
LLM_OPTIONS = [
    ("novita",          "meta-llama/llama-3.1-8b-instruct"),
    ("novita",          "mistralai/mistral-7b-instruct-v0.3"),
    ("featherless-ai",  "mistralai/Mistral-7B-Instruct-v0.3"),
    ("featherless-ai",  "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("together",        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ("together",        "mistralai/Mistral-7B-Instruct-v0.3"),
]

TOP_K_CHUNKS  = 6
MIN_RELEVANCE = 0.05

os.makedirs(LAWS_FOLDER,    exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# =============================================================================
# GLOBALS
# =============================================================================

embeddings  = None
vectorstore = None

# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

def detect_language(text: str) -> str:
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    latin  = len(re.findall(r'[a-zA-ZÀ-ÿ]',    text))
    total  = arabic + latin
    if total == 0:
        return "fr"
    if arabic / total > 0.60:
        return "ar"
    if latin  / total > 0.60:
        return "fr"
    return "mixed"

# =============================================================================
# PDF INGESTION
# =============================================================================

def _extract_pdf(path: str) -> list:
    docs     = []
    filename = os.path.basename(path)
    try:
        with pdfplumber.open(path) as pdf:
            for num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(
                        page_content=text.strip(),
                        metadata={"source": filename, "page": num}
                    ))
    except Exception as e:
        print(f"[WARN] Could not read {filename}: {e}")
    return docs


def ingest_pdfs():
    global vectorstore

    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print(f"[DB] Loading existing ChromaDB from {CHROMA_DB_PATH} ...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        try:
            count = vectorstore._collection.count()
            print(f"[DB] Loaded {count} chunks.")
        except Exception:
            print("[DB] Existing index loaded.")
        return

    pdfs = [f for f in os.listdir(LAWS_FOLDER) if f.endswith(".pdf")]
    if not pdfs:
        print("[WARN] No PDFs found in ./laws/")
        return

    print(f"[DB] Indexing {len(pdfs)} PDF(s)...")
    all_docs = []
    for f in pdfs:
        docs = _extract_pdf(os.path.join(LAWS_FOLDER, f))
        all_docs.extend(docs)
        print(f"  → {f}: {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        separators=["\n\n", "\n", ".", "،", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[DB] Storing {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist()
    print(f"[DB] Done — {len(chunks)} chunks indexed.")

# =============================================================================
# STARTUP
# =============================================================================

def _load_embeddings():
    global embeddings
    print("[EMB] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("[EMB] Ready.")


def init():
    _load_embeddings()
    ingest_pdfs()
    _ensure_contacts_csv()
    print("\n✅ Chatbot fully initialized and ready.\n")

# =============================================================================
# CONTACTS
# =============================================================================

def _ensure_contacts_csv():
    if not os.path.exists(CONTACTS_CSV):
        pd.DataFrame([
            {"name": "محمد بن علي",            "specialization": "قانون الشغل عقود عمل",           "phone": "+216 20 000 001", "email": "m.benali@avocat.tn",    "city": "تونس",   "languages": "ar,fr"},
            {"name": "Sonia Trabelsi",          "specialization": "droit commercial sociétés",        "phone": "+216 20 000 002", "email": "s.trabelsi@avocat.tn",  "city": "Sfax",   "languages": "fr,ar"},
            {"name": "Karim Gharbi",            "specialization": "droit de la famille divorce",      "phone": "+216 20 000 003", "email": "k.gharbi@avocat.tn",    "city": "Sousse", "languages": "fr"},
            {"name": "فاطمة الزهراء المنصوري",  "specialization": "قانون الأسرة إرث طلاق",           "phone": "+216 20 000 004", "email": "f.mansouri@avocat.tn",  "city": "صفاقس",  "languages": "ar"},
            {"name": "Amine Chaabane",          "specialization": "droit pénal criminel infractions", "phone": "+216 20 000 005", "email": "a.chaabane@avocat.tn",  "city": "Tunis",  "languages": "fr,ar"},
        ]).to_csv(CONTACTS_CSV, index=False, encoding="utf-8-sig")
        print("[CSV] Contacts file created.")


def _find_contacts(question: str, lang: str) -> str:
    try:
        df = pd.read_csv(CONTACTS_CSV, encoding="utf-8-sig")
    except Exception as e:
        return f"⚠️ Contacts unavailable: {e}"

    def strip_diac(t):
        return "".join(
            c for c in unicodedata.normalize("NFD", str(t))
            if unicodedata.category(c) != "Mn"
        ).lower()

    keywords = [w for w in re.split(r'\s+', strip_diac(question)) if len(w) > 3]
    df["_score"] = df["specialization"].apply(
        lambda s: sum(1 for kw in keywords if kw in strip_diac(s))
    )
    matched = df[df["_score"] > 0].sort_values("_score", ascending=False).head(3)
    if matched.empty:
        matched = df.sort_values("city").head(3)

    def fmt(row):
        return (f"👤 {row['name']} — {row['specialization']}\n"
                f"   📍 {row['city']}  |  📞 {row['phone']}  |  ✉️  {row['email']}")

    contacts = "\n\n".join(matched.apply(fmt, axis=1).tolist())

    if lang == "ar":
        return "لم نجد إجابة في النصوص القانونية.\nإليك متخصصون يمكنهم مساعدتك:\n\n" + contacts
    if lang == "fr":
        return "Réponse introuvable dans les textes juridiques.\nVoici des professionnels qui peuvent vous aider :\n\n" + contacts
    return "لم نجد إجابة.\n\n" + contacts

# =============================================================================
# PROMPT MESSAGES
# =============================================================================

def _build_messages(question: str, context: str, lang: str) -> list:
    if lang == "ar":
        system = (
            "أنت مساعد قانوني تونسي متخصص. "
            "أجب بالعربية فقط بناءً على النصوص القانونية التونسية المقدمة. "
            "اذكر المصدر دائماً هكذا: [اسم_الملف.pdf، صفحة X]. "
            "إذا لم تجد إجابة واضحة قل ذلك صراحةً."
        )
        user = (
            f"النصوص القانونية:\n{context}\n\n"
            f"السؤال: {question}\n\n"
            "أجب الآن بالعربية مع ذكر المصدر:"
        )
    elif lang == "fr":
        system = (
            "Tu es un assistant juridique tunisien expert. "
            "Réponds en français uniquement en te basant sur les textes juridiques fournis. "
            "Cite toujours la source comme: [fichier.pdf, page X]. "
            "Si la réponse n'est pas dans les textes, dis-le clairement."
        )
        user = (
            f"Textes juridiques:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Réponds maintenant en français avec la source:"
        )
    else:
        system = (
            "You are a Tunisian legal assistant. "
            "Answer only based on the provided legal texts. "
            "Cite sources as [filename.pdf, page X]."
        )
        user = f"Legal texts:\n{context}\n\nQuestion: {question}\n\nAnswer with source:"

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

# =============================================================================
# LLM CALLER — InferenceClient with provider fallback chain
# =============================================================================

def _call_llm(messages: list) -> str:
    for provider, model in LLM_OPTIONS:
        try:
            print(f"[LLM] Trying {provider} / {model} ...")
            client = InferenceClient(
                provider=provider,
                api_key=HF_API_KEY,
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                temperature=0.1,
                stream=False,
            )
            text = response.choices[0].message.content.strip()
            if text:
                print(f"[LLM] ✅ Got answer from {provider}/{model}")
                return text
        except Exception as e:
            err = str(e)
            print(f"[LLM] {provider}/{model} failed: {err[:120]}, trying next...")
            continue

    print("[LLM] All providers/models failed.")
    return ""

# =============================================================================
# PUBLIC ask()
# =============================================================================

def ask(question: str, pdf_filename: str = None) -> dict:
    if vectorstore is None:
        return {"answer": "⚠️ Aucun PDF indexé. Ajoutez vos PDFs dans ./laws/ et redémarrez.",
                "lang": "fr", "source": "error"}

    lang = detect_language(question)

    # Greeting shortcut
    greeting_kw = ["hello", "hi", "مرحبا", "bonjour", "salut", "اهلا", "صباح", "مساء"]
    if any(kw in question.lower() for kw in greeting_kw) and len(question.split()) <= 4:
        if lang == "ar":
            return {"answer": "مرحباً! أنا المساعد القانوني التونسي. اختر نوع العقد أولاً ثم اسأل سؤالك.", "lang": lang, "source": "greeting"}
        return {"answer": "Bonjour ! Je suis l'assistant juridique tunisien. Choisissez d'abord le type de contrat.", "lang": lang, "source": "greeting"}

    # Vector search
    try:
        all_results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K_CHUNKS * 2)
        print(f"[DEBUG] Total results: {len(all_results)}")

        if pdf_filename:
            results = [(d, s) for d, s in all_results if d.metadata.get("source") == pdf_filename]
            print(f"[DEBUG] Filtered to {pdf_filename}: {len(results)} results")
            if not results:
                results = all_results[:TOP_K_CHUNKS]
                print("[DEBUG] Falling back to global results")
        else:
            results = all_results[:TOP_K_CHUNKS]

    except Exception as e:
        return {"answer": f"⚠️ Erreur de recherche: {e}", "lang": lang, "source": "error"}

    if not results:
        return {"answer": _find_contacts(question, lang), "lang": lang, "source": "contacts"}

    best_score = max(s for _, s in results)
    print(f"[DEBUG] Best score: {best_score:.3f} (min: {MIN_RELEVANCE})")

    if best_score < MIN_RELEVANCE:
        return {"answer": _find_contacts(question, lang), "lang": lang, "source": "contacts"}

    # Build context
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source','?')}, page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d, _ in results[:TOP_K_CHUNKS]
    )

    messages = _build_messages(question, context, lang)
    response = _call_llm(messages)

    if not response:
        return {"answer": _find_contacts(question, lang), "lang": lang, "source": "contacts"}

    return {"answer": response, "lang": lang, "source": "llm"}