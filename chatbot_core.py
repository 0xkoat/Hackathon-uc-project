# -*- coding: utf-8 -*-
"""
chatbot_core.py
---------------
Pure logic module — no UI, no loops.
Imported by both the Jupyter notebook and the Flask web server.
"""

import os
import re
import unicodedata
import warnings
import pandas as pd
import pdfplumber
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import InferenceClient

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

LAWS_FOLDER     = "./laws/"
CHROMA_DB_PATH  = "./chroma_db/"
CONTACTS_CSV    = "./law_contacts.csv"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PRIMARY_MODEL   = "google/flan-t5-xl"
HF_API_KEY      = "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"
FALLBACK_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
TOP_K_CHUNKS    = 5
MIN_RELEVANCE   = 0.45

os.makedirs(LAWS_FOLDER,    exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# =============================================================================
# GLOBALS (populated by init())
# =============================================================================

embeddings             = None
vectorstore            = None
flan_model             = None
flan_tokenizer         = None
hf_client              = None
PRIMARY_LLM_AVAILABLE  = False

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
        print("[DB] Loading existing ChromaDB index...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        print(f"[DB] {vectorstore._collection.count()} chunks ready.")
        return

    pdfs = [f for f in os.listdir(LAWS_FOLDER) if f.endswith(".pdf")]
    if not pdfs:
        print("[WARN] No PDFs found in ./laws/ — upload your law PDFs first.")
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
    print(f"[DB] Creating {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist()
    print(f"[DB] Done. {len(chunks)} chunks stored.")

# =============================================================================
# LLM LOADERS
# =============================================================================

def _load_embeddings():
    global embeddings
    print("[EMB] Loading multilingual embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("[EMB] Ready.")


def _load_flan():
    global flan_model, flan_tokenizer, PRIMARY_LLM_AVAILABLE
    try:
        print(f"[LLM] Loading {PRIMARY_MODEL} (may take a few minutes)...")
        flan_tokenizer = T5Tokenizer.from_pretrained(PRIMARY_MODEL)
        flan_model     = T5ForConditionalGeneration.from_pretrained(
            PRIMARY_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        flan_model.eval()
        PRIMARY_LLM_AVAILABLE = True
        print(f"[LLM] {PRIMARY_MODEL} ready.")
    except Exception as e:
        print(f"[LLM] Could not load flan-t5-xl: {e}")
        print("[LLM] Will use HuggingFace API fallback.")
        PRIMARY_LLM_AVAILABLE = False


def _load_fallback():
    global hf_client
    try:
        hf_client = InferenceClient(model=FALLBACK_MODEL, token=HF_API_KEY)
        print("[LLM] HuggingFace fallback client ready.")
    except Exception as e:
        print(f"[LLM] Fallback client error: {e}")


def init():
    """Call once at startup to load all models and index PDFs."""
    _load_embeddings()
    ingest_pdfs()
    _load_flan()
    _load_fallback()
    _ensure_contacts_csv()
    print("\n✅ Chatbot fully initialized and ready.")

# =============================================================================
# CONTACTS CSV
# =============================================================================

def _ensure_contacts_csv():
    if not os.path.exists(CONTACTS_CSV):
        pd.DataFrame([
            {"name": "محمد بن علي",           "specialization": "قانون الشغل عقود عمل",         "phone": "+216 20 000 001", "email": "m.benali@avocat.tn",    "city": "تونس",    "languages": "ar,fr"},
            {"name": "Sonia Trabelsi",         "specialization": "droit commercial sociétés",      "phone": "+216 20 000 002", "email": "s.trabelsi@avocat.tn",  "city": "Sfax",    "languages": "fr,ar"},
            {"name": "Karim Gharbi",           "specialization": "droit de la famille divorce",    "phone": "+216 20 000 003", "email": "k.gharbi@avocat.tn",    "city": "Sousse",  "languages": "fr"},
            {"name": "فاطمة الزهراء المنصوري", "specialization": "قانون الأسرة إرث طلاق",         "phone": "+216 20 000 004", "email": "f.mansouri@avocat.tn",  "city": "صفاقس",   "languages": "ar"},
            {"name": "Amine Chaabane",         "specialization": "droit pénal criminel infractions","phone": "+216 20 000 005", "email": "a.chaabane@avocat.tn",  "city": "Tunis",   "languages": "fr,ar"},
        ]).to_csv(CONTACTS_CSV, index=False, encoding="utf-8-sig")
        print("[CSV] Sample contacts CSV created.")


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
        return "Réponse introuvable dans les textes.\nVoici des professionnels qui peuvent vous aider :\n\n" + contacts
    return (
        "لم نجد إجابة في النصوص القانونية.\nإليك متخصصون يمكنهم مساعدتك:\n\n" + contacts +
        "\n\n---\n\nRéponse introuvable dans les textes.\nVoici des professionnels qui peuvent vous aider :\n\n" + contacts
    )

# =============================================================================
# PROMPT BUILDER
# =============================================================================

def _build_prompt(question: str, context: str, lang: str) -> str:
    if lang == "ar":
        return (
            "أنت مساعد قانوني تونسي. أجب بالعربية التونسية فقط.\n"
            "استخدم فقط النصوص أدناه. لا معرفة خارجية.\n"
            "إذا لم تجد الإجابة اكتب فقط: ANSWER_NOT_FOUND\n"
            "اذكر المصدر دائماً: [اسم_الملف.pdf، صفحة X]\n\n"
            f"النصوص:\n{context}\n\nالسؤال: {question}\n\nالإجابة:"
        )
    if lang == "fr":
        return (
            "Tu es un assistant juridique tunisien. Réponds UNIQUEMENT en français.\n"
            "Utilise SEULEMENT les textes ci-dessous. Aucune connaissance externe.\n"
            "Si introuvable, écris uniquement: ANSWER_NOT_FOUND\n"
            "Cite toujours la source: [fichier.pdf, page X]\n\n"
            f"Textes:\n{context}\n\nQuestion: {question}\n\nRéponse:"
        )
    return (
        "You are a Tunisian legal assistant. User writes in Arabic and French.\n"
        "Reply FIRST in Tunisian Arabic, then the SAME answer in French after '---'.\n"
        "Use ONLY the texts below. No outside knowledge.\n"
        "If not found, write only: ANSWER_NOT_FOUND\n"
        "Always cite: [filename.pdf, page X]\n\n"
        f"Texts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

# =============================================================================
# LLM CALLERS
# =============================================================================

def _call_flan(prompt: str) -> str:
    inputs  = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = flan_model.generate(
        inputs["input_ids"],
        max_new_tokens=512, do_sample=True,
        top_p=0.92, temperature=0.3, no_repeat_ngram_size=3
    )
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _call_fallback(prompt: str) -> str:
    if hf_client is None:
        return "ANSWER_NOT_FOUND"
    try:
        return hf_client.text_generation(
            prompt, max_new_tokens=512, temperature=0.3, repetition_penalty=1.1
        ).strip()
    except Exception as e:
        print(f"[LLM] Fallback error: {e}")
        return "ANSWER_NOT_FOUND"


def _call_llm(prompt: str) -> str:
    if PRIMARY_LLM_AVAILABLE:
        try:
            result = _call_flan(prompt)
            if result:
                return result
        except Exception as e:
            print(f"[LLM] flan-t5-xl failed ({e}), switching to fallback...")
    return _call_fallback(prompt)

# =============================================================================
# PUBLIC ask() FUNCTION
# =============================================================================

def ask(question: str) -> dict:
    """
    Main entry point. Returns a dict:
      { "answer": str, "lang": str, "source": "llm" | "contacts" | "error" }
    """
    if vectorstore is None:
        return {"answer": "⚠️ No PDFs indexed. Please add PDFs to ./laws/ and restart.",
                "lang": "fr", "source": "error"}

    lang = detect_language(question)

    try:
        results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K_CHUNKS)
    except Exception as e:
        return {"answer": f"⚠️ Retrieval error: {e}", "lang": lang, "source": "error"}

    if not results:
        return {"answer": _find_contacts(question, lang), "lang": lang, "source": "contacts"}

    best_score = max(score for _, score in results)

    if best_score < MIN_RELEVANCE:
        return {"answer": _find_contacts(question, lang), "lang": lang, "source": "contacts"}

    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source','?')}, page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d, _ in results
    )

    prompt   = _build_prompt(question, context, lang)
    response = _call_llm(prompt)

    if "ANSWER_NOT_FOUND" in response.upper():
        return {"answer": _find_contacts(question, lang), "lang": lang, "source": "contacts"}

    return {"answer": response, "lang": lang, "source": "llm"}
