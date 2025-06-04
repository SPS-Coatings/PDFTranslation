# ─────────────────────────────────────────────────────────────────────────────
#  Universal PDF-to-Markdown + Translation (Image Analysis Excluded)
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────── Standard library ──────────────────────────────────
import io, os, sys, json, csv, re, tempfile, contextlib, warnings
from pathlib import Path
from typing  import List, Tuple, Optional, Any

# ────────────────────────── Third-party basics ────────────────────────────────
import streamlit as st

# ────────────────────────── TEMP PATCH for charset_normalizer ↑ pdfminer SIX ──
# Newer `charset-normalizer >= 4.0` removed `is_cjk_uncommon`; some releases of
# pdfminer.six (pulled in by pdfplumber) still import it. We stub it in once
# so the import chain succeeds even with the latest charset-normalizer.
try:
    import charset_normalizer.utils as _cnu
    if not hasattr(_cnu, "is_cjk_uncommon"):
        def _dummy_is_cjk_uncommon(cp: int) -> bool:
            """
            Return False for every code-point.
            Stub for pdfminer.six compatibility with newer charset-normalizer.
            """
            return False
        _cnu.is_cjk_uncommon = _dummy_is_cjk_uncommon
except Exception:
    # If charset_normalizer itself is missing, pdfplumber will raise later and
    # we’ll surface that error; no further action needed here.
    pass

# ────────────────────────── PDF text extraction (pdfplumber) ──────────────────
_PDFPLUMBER_AVAILABLE = False
_PDFPLUMBER_ERR       = ""
try:
    import pdfplumber  # pulls in pdfminer.six -> charset_normalizer
    _PDFPLUMBER_AVAILABLE = True
except Exception as ex:  # catch *all* import problems
    _PDFPLUMBER_ERR = repr(ex)

# ────────────────────────── OpenAI ≥ 1.0 client (o3-mini) ─────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ────────────────────────── DeepL simple REST helper ──────────────────────────
import requests
DEEPL_SUPPORTED_LANGS = {
    "BG":"Bulgarian","CS":"Czech","DA":"Danish","DE":"German","EL":"Greek",
    "EN":"English","ES":"Spanish","ET":"Estonian","FI":"Finnish","FR":"French",
    "HU":"Hungarian","ID":"Indonesian","IT":"Italian","JA":"Japanese",
    "KO":"Korean","LT":"Lithuanian","LV":"Latvian","NB":"Norwegian (Bokmål)",
    "NL":"Dutch","PL":"Polish","PT":"Portuguese","RO":"Romanian","RU":"Russian",
    "SK":"Slovak","SL":"Slovenian","SV":"Swedish","TR":"Turkish",
    "UK":"Ukrainian","ZH":"Chinese"
}
LANG_NAME_TO_CODE = {v:k for k,v in DEEPL_SUPPORTED_LANGS.items()}

# ────────────────────────── Streamlit page config ─────────────────────────────
st.set_page_config(
    page_title="PDF → Markdown → Translate",
    page_icon="📝",
    layout="wide"
)

# ────────────────────────── Session-state (API keys) ──────────────────────────
for key in ("OPENAI_API_KEY","DEEPL_API_KEY"):
    st.session_state.setdefault(key, None)

# ────────────────────────── Sidebar – configuration UI ────────────────────────
with st.sidebar:
    st.title("🔑 Configuration")

    # 1 · OpenAI
    if not st.session_state.OPENAI_API_KEY:
        okey = st.text_input("OpenAI Key (o3-mini)", type="password",
                             help="Create at https://platform.openai.com/")
        if okey:
            st.session_state.OPENAI_API_KEY = okey.strip()
            st.success("OpenAI key saved — reloading.")
            st.rerun()
    else:
        st.success("OpenAI key ✓")
        if st.button("Reset OpenAI Key"):
            st.session_state.OPENAI_API_KEY = None
            st.rerun()

    # 2 · DeepL
    if not st.session_state.DEEPL_API_KEY:
        dkey = st.text_input("DeepL API Key", type="password",
                             help="Get from https://www.deepl.com/account/summary")
        if dkey:
            st.session_state.DEEPL_API_KEY = dkey.strip()
            st.success("DeepL key saved.")
    else:
        st.success("DeepL key ✓")
        if st.button("Reset DeepL Key"):
            st.session_state.DEEPL_API_KEY = None
            st.rerun()

    translation_provider = st.radio(
        "Translation provider", options=["o3-mini","DeepL"], index=0
    )

    st.info("PDF → Markdown → Translate")
    st.warning("AI output may contain errors. Verify critical data.")

# ────────────────────────── OpenAI client -------------------------------------
openai_client = (
    OpenAI(api_key=st.session_state.OPENAI_API_KEY)
    if _OPENAI_AVAILABLE and st.session_state.OPENAI_API_KEY else None
)

# ────────────────────────── Helper function: PDF → Markdown -------------------
def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """
    Extract text from each page of a PDF, returning structured Markdown.
    """
    if not _PDFPLUMBER_AVAILABLE:
        st.error(
            "pdfplumber import failed.\n\n"
            f"```python\n{_PDFPLUMBER_ERR}\n```"
        )
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_md = []
            for i, p in enumerate(pdf.pages, 1):
                text = p.extract_text() or ""
                page_md = f"## Page {i}\n\n{text.strip()}"
                pages_md.append(page_md)
        return "\n\n".join(pages_md)
    except Exception as ex:
        st.error(f"PDF parsing failed:\n\n```\n{ex}\n```")
        return ""

# ────────────────────────── Main UI layout ------------------------------------
st.title("📄 PDF → Markdown → Translate")
st.caption("Structured Markdown extraction with optional translation")

analysis_container     = st.container()
translation_container  = st.container()

# PDF Upload & Optional Prompt
st.subheader("Upload a PDF")
uploaded_pdf = st.file_uploader("📤 PDF (multi-page)", type=["pdf"])

user_prompt = st.text_area(
    "✍️ Optional user prompt:",
    placeholder="Add any note or request here…",
    height=100
)

analyze_button = st.button("🔍 Convert to Markdown")

analysis_result = ""
if analyze_button:
    if not uploaded_pdf:
        st.warning("Please upload a PDF first.")
    else:
        with analysis_container:
            st.subheader("📄 PDF → Markdown")
            pdf_bytes = uploaded_pdf.read()
            analysis_result = pdf_to_markdown(pdf_bytes)
            st.markdown("### Resulting Markdown")
            st.markdown(analysis_result or "*No output*")

        st.session_state["analysis_result"] = analysis_result

# ────────────────────────── Translation section ─────────────────────────────--
st.markdown("---")
with translation_container:
    st.subheader("🌐 Translate Markdown")

    if not st.session_state.get("analysis_result"):
        st.info("Run analysis/conversion first.")
    else:
        langs  = ["English"] + sorted(LANG_NAME_TO_CODE.keys())
        target = st.selectbox("Target language:", langs, index=0)

        if target == "English":
            st.info("Already in English — no translation needed.")
        else:
            if translation_provider == "o3-mini" and openai_client is None:
                st.warning("Add OpenAI key to use o3-mini translation.")
            if translation_provider == "DeepL" and not st.session_state.DEEPL_API_KEY:
                st.warning("Add DeepL key for translation.")

            if st.button("Translate Markdown"):
                src_md = st.session_state["analysis_result"]
                with st.spinner(f"Translating via {translation_provider} …"):
                    if translation_provider == "DeepL":
                        try:
                            resp = requests.post(
                                "https://api-free.deepl.com/v2/translate",
                                data={
                                    "auth_key":    st.session_state.DEEPL_API_KEY,
                                    "text":        src_md,
                                    "target_lang": LANG_NAME_TO_CODE[target]
                                },
                                timeout=30
                            )
                            if resp.status_code == 200:
                                trans = resp.json()["translations"][0]["text"]
                                st.markdown(f"## 🌐 Translated ({target})")
                                st.markdown(trans)
                            else:
                                st.error(f"DeepL error {resp.status_code}: {resp.text}")
                        except Exception as ex:
                            st.error(f"DeepL request failed:\n\n```\n{ex}\n```")
                    else:  # translation_provider == "o3-mini"
                        try:
                            system_msg = (
                                f"Translate the following Markdown into {target}, "
                                "preserving all headings and formatting. "
                                "Respond ONLY with the translation."
                            )
                            comp = openai_client.chat.completions.create(
                                model="o3-mini",
                                messages=[
                                    {"role":"system","content":system_msg},
                                    {"role":"user","content":src_md},
                                ]
                            )
                            trans = comp.choices[0].message.content.strip()
                            st.markdown(f"## 🌐 Translated ({target})")
                            st.markdown(trans)
                        except Exception as ex:
                            st.error(f"o3-mini translation error:\n\n```\n{ex}\n```")

# ────────────────────────── End notes ─────────────────────────────────────────
"""
Run:
    streamlit run universal_pdf_md_translator.py
────────────────────────────────────────────────────────────────────────────
"""
