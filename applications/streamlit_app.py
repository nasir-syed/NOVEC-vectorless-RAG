import os
import json
import time
import logging
import tempfile
from typing import List, Dict

import streamlit as st

try:
    from novec import RAGConfig, PageIndexAPI, RAGEngine, QueryExecutor
    _backend_ok = True
    _backend_err = None
except Exception as _e:
    _backend_ok = False
    _backend_err = str(_e)

st.set_page_config(
    page_title="NOVEC RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS 
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #111111 !important;
    color: #f3f3f3 !important;
}
[data-testid="stHeader"] { background: #111111 !important; }
[data-testid="stSidebar"] { background: #111111 !important; }

h1, h2, h3, h4 { color: #f3f3f3 !important; letter-spacing: -0.02em; }
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888 !important;
    margin-bottom: 0.5rem;
}

.rag-card {
    background: transparent;
    border-bottom: 1px solid #222;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}
.rag-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.stButton > button {
    background: #f3f3f3 !important;
    color: #111111 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:disabled { opacity: 0.35 !important; cursor: not-allowed !important; }

/* Secondary / ghost buttons */
button[kind="secondary"],
.btn-ghost > button {
    background: transparent !important;
    color: #f3f3f3 !important;
    border: 1px solid #333 !important;
}
button[kind="secondary"]:hover,
.btn-ghost > button:hover { border-color: #666 !important; }

/* Danger button */
.btn-danger > button {
    background: #3d1a1a !important;
    color: #f87171 !important;
    border: 1px solid #5a2424 !important;
}
.btn-danger > button:hover { background: #4d2020 !important; }

.stTextArea textarea, .stTextInput input {
    background: #1a1a1a !important;
    color: #f3f3f3 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #555 !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] {
    background: #1a1a1a !important;
    border: 1px dashed #333 !important;
    border-radius: 8px !important;
}

.stCheckbox label { font-size: 0.88rem !important; color: #ccc !important; }
.stCheckbox span[data-testid="stCheckboxValue"] { border-color: #f3f3f3 !important; }

/* ── Document row ────────────────────────────────────────────────── */
.doc-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 0.75rem;
    border-radius: 6px;
    border: 1px solid #222;
    background: #161616;
    margin-bottom: 0.5rem;
    transition: border-color 0.15s;
}
.doc-row:hover { border-color: #333; }
.doc-name {
    flex: 1;
    font-size: 0.85rem;
    color: #e5e5e5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-badge {
    display: block;
    width: 72%;
    height: 2.5rem;
    text-align: center;
    font-size: 0.65rem;
    padding: 0.6rem 0.4rem;
    margin-left: auto;
    border-radius: 10px;
    font-weight: 600;
}
.badge-ready  { background: #14532d; color: #4ade80; border: 1px solid #166534; }
.badge-proc   { background: #713f12; color: #fbbf24; border: 1px solid #92400e; }
.badge-failed { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }

.answer-box {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.7;
    white-space: pre-wrap;
    color: #e5e5e5;
}
.citation { color: #a5b4fc; font-weight: 600; }

.answer-response-panel {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px 8px 0 0;
    padding: 1.25rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: #e5e5e5;
}
.answer-response-panel .inline-cite {
    color: #a5b4fc;
    font-weight: 600;
}
.answer-error-panel {
    background: #1a0e0e;
    border: 1px solid #3d1a1a;
    border-radius: 8px;
    padding: 1.1rem 1.5rem;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #f87171;
}
.citations-panel {
    background: #111;
    border: 1px solid #2a2a2a;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 0.85rem 1.5rem 1rem 1.5rem;
}
.citations-header {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.6rem;
}
.citation-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.citation-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    font-size: 0.75rem;
    color: #a5b4fc;
    font-weight: 500;
    white-space: nowrap;
}
.citation-chip-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.1rem;
    height: 1.1rem;
    background: #2d2d4e;
    border-radius: 50%;
    font-size: 0.6rem;
    font-weight: 700;
    color: #818cf8;
    flex-shrink: 0;
}
.no-citations {
    font-size: 0.78rem;
    color: #444;
    font-style: italic;
}

.log-container {
    background: #0d0d0d;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: 'SFMono-Regular', 'Menlo', 'Monaco', monospace;
    font-size: 0.75rem;
    line-height: 1.6;
    max-height: 320px;
    overflow-y: auto;
    color: #9ca3af;
}
.log-info    { color: #6ee7b7; }
.log-error   { color: #f87171; }
.log-warning { color: #fbbf24; }
.log-success { color: #34d399; }

.upload-status {
    font-size: 0.82rem;
    color: #888;
    padding: 0.4rem 0;
}

.step-pill {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    color: #9ca3af;
    margin-bottom: 0.5rem;
}

[data-testid="stModal"] {
    background: rgba(0,0,0,0.7) !important;
}
[data-testid="stModalContent"] {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
}

hr { border-color: #222 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# Session state defaults
def _init_state():
    defaults = {
        "documents": [],          
        "selected_doc": None,     
        "logs": [],               
        "answer": None,           
        "query_running": False,
        "docs_loaded": False,
        "upload_running": False,
        "confirm_delete_id": None,
        "confirm_delete_name": None,
        "show_upload_modal": False,
        "upload_status": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

class UILogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        entry = f"[{record.levelname}] {msg}"
        if not st.session_state.logs or st.session_state.logs[-1] != entry:
            st.session_state.logs.append(entry)
        if len(st.session_state.logs) > 300:
            st.session_state.logs = st.session_state.logs[-300:]

if _backend_ok:
    ui_logger = logging.getLogger("streamlit_rag")
    ui_logger.setLevel(logging.INFO)
    ui_logger.handlers = [h for h in ui_logger.handlers if not isinstance(h, UILogHandler)]
    _ui_handler = UILogHandler()
    _ui_handler.setFormatter(logging.Formatter("%(asctime)s — %(message)s", "%H:%M:%S"))
    ui_logger.addHandler(_ui_handler)
    ui_logger.propagate = False 

def _add_log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{level}] {ts} — {msg}")


def _get_config() -> RAGConfig:
    if "rag_config" not in st.session_state:
        try:
            st.session_state.rag_config = RAGConfig(logger=ui_logger)
        except ValueError as e:
            st.error(f"Configuration error: {str(e)}")
            st.stop()
    return st.session_state.rag_config


def _get_api() -> PageIndexAPI:
    config = _get_config()
    if "api" not in st.session_state:
        st.session_state.api = PageIndexAPI(config)
    return st.session_state.api


def _get_rag() -> RAGEngine:
    config = _get_config()
    if "rag" not in st.session_state:
        st.session_state.rag = RAGEngine(config)
    return st.session_state.rag


def _get_query_executor() -> QueryExecutor:
    if "query_executor" not in st.session_state:
        st.session_state.query_executor = QueryExecutor(_get_api(), _get_rag())
    return st.session_state.query_executor


def _load_documents():
    docs = _get_api().fetch_documents()
    st.session_state.documents = docs if docs else []
    st.session_state.docs_loaded = True
    existing_ids = {d.get("id") for d in st.session_state.documents}
    if st.session_state.selected_doc not in existing_ids:
        st.session_state.selected_doc = None


def _badge(status: str) -> str:
    status = (status or "").lower()
    if status == "completed":
        return '<span class="doc-badge badge-ready">Ready</span>'
    elif status in ("processing", "pending", "queued"):
        return '<span class="doc-badge badge-proc">Processing</span>'
    else:
        return f'<span class="doc-badge badge-failed">{status or "Unknown"}</span>'


def _render_logs():
    if not st.session_state.logs:
        st.markdown('<div class="log-container"><span style="color:#444">No logs yet.</span></div>', unsafe_allow_html=True)
        return

    raw_logs = st.session_state.logs[-200:]
    deduped = [raw_logs[0]] if raw_logs else []
    for line in raw_logs[1:]:
        if line != deduped[-1]:
            deduped.append(line)

    lines_html = []
    for line in reversed(deduped):
        if "[ERROR]" in line:
            cls = "log-error"
        elif "[WARNING]" in line:
            cls = "log-warning"
        elif "✅" in line or "success" in line.lower():
            cls = "log-success"
        else:
            cls = "log-info"
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines_html.append(f'<div class="{cls}">{safe}</div>')
    st.markdown(
        '<div class="log-container">' + "".join(lines_html) + '</div>',
        unsafe_allow_html=True,
    )


def _render_answer(answer_data: dict):
    import re as _re

    is_error   = answer_data.get("_error", False)
    response   = answer_data.get("response", "")
    citations  = answer_data.get("citations", [])

    if is_error:
        safe_resp = (
            response
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        st.markdown(
            f'<div class="answer-error-panel">{safe_resp}</div>',
            unsafe_allow_html=True,
        )
        return

    safe_resp = (
        response
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    highlighted = _re.sub(
        r"(\[[^\]]+\])",
        r'<span class="inline-cite">\1</span>',
        safe_resp,
    )
    st.markdown(
        f'<div class="answer-response-panel">{highlighted}</div>',
        unsafe_allow_html=True,
    )

    if citations:
        chips_html = ""
        for i, cite in enumerate(citations, 1):
            safe_cite = (
                str(cite)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            chips_html += (
                f'<span class="citation-chip">'
                f'<span class="citation-chip-num">{i}</span>'
                f'{safe_cite}'
                f'</span>'
            )
        sources_block = (
            f'<div class="citations-panel">'
            f'<div class="citations-header">Sources</div>'
            f'<div class="citation-list">{chips_html}</div>'
            f'</div>'
        )
    else:
        sources_block = (
            '<div class="citations-panel">'
            '<div class="citations-header">Sources</div>'
            '<span class="no-citations">No source references available.</span>'
            '</div>'
        )

    st.markdown(sources_block, unsafe_allow_html=True)

    plain_citations = "\n".join(
        f"[{i}] {c}" for i, c in enumerate(citations, 1)
    ) if citations else ""
    plain_text = response + (
        ("\n\nSources:\n" + plain_citations) if plain_citations else ""
    )
    copy_js = f"""
    <script>
    function copyAnswer() {{
        const txt = {json.dumps(plain_text)};
        navigator.clipboard.writeText(txt).then(() => {{
            document.getElementById('copy-btn').innerText = 'Copied';
            setTimeout(() => document.getElementById('copy-btn').innerText = '⎘ Copy answer', 1500);
        }});
    }}
    </script>
    <button id="copy-btn" onclick="copyAnswer()"
        style="margin-top:0.6rem; background:#1a1a1a; border:1px solid #333; color:#aaa;
               border-radius:5px; padding:0.3rem 0.8rem; font-size:0.75rem; cursor:pointer;">
        ⎘ Copy answer
    </button>
    """
    st.markdown(copy_js, unsafe_allow_html=True)

@st.dialog("Confirm deletion")
def _delete_modal(doc_id: str, doc_name: str):
    st.markdown(f"Are you sure you want to delete **{doc_name}**? This action cannot be undone.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Delete", key="modal_confirm_del", type="primary"):
            with st.spinner("Deleting…"):
                ok = _get_api().delete_document(doc_id)
            if ok:
                _add_log(f"Deleted document '{doc_name}' (id={doc_id})")
                _load_documents()
                st.success("Document deleted.")
            else:
                _add_log(f"Failed to delete '{doc_name}'", "ERROR")
                st.error("Deletion failed.")
            st.rerun()
    with col2:
        if st.button("Cancel", key="modal_cancel_del"):
            st.rerun()

@st.dialog("Upload PDF document")
def _upload_modal():
    st.markdown("Select a **PDF** file from your machine to upload and index.")
    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        key="modal_file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload & Index", key="modal_do_upload", type="primary"):
                _add_log(f"Starting upload for '{uploaded_file.name}'")
                status_ph = st.empty()

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False, dir="/tmp"
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                api = _get_api()

                status_ph.info("⬆Uploading to PageIndex…")
                doc_id = api.upload_document(tmp_path)

                if not doc_id:
                    status_ph.error("Upload failed. Check logs.")
                    _add_log("Upload failed", "ERROR")
                    os.unlink(tmp_path)
                    return

                _add_log(f"Uploaded '{uploaded_file.name}' → doc_id={doc_id}")
                status_ph.info("Waiting for indexing to complete…")

                # Poll for indexing completion
                ready = api.wait_for_indexing(doc_id, timeout=300, poll_interval=5)
                os.unlink(tmp_path)

                if ready:
                    _add_log(f"Document '{uploaded_file.name}' indexed and ready.")
                    status_ph.success("Document is ready!")
                    _load_documents()
                    time.sleep(0.8)
                    st.rerun()
                else:
                    status_ph.error("Indexing failed or timed out.")
                    _add_log("Indexing failed or timed out", "ERROR")
        with col2:
            if st.button("Cancel", key="modal_cancel_upload"):
                st.rerun()

def _run_query(query: str, selected_docs_list: List[Dict]):
    st.session_state.query_running = True
    st.session_state.answer = None

    _add_log(f'Query started: "{query}"')
    _add_log(f"Selected documents: {[d.get('name') for d in selected_docs_list]}")

    def progress_callback(step: str, message: str):
        _add_log(message)

    executor = _get_query_executor()
    result = executor.execute_query(query, selected_docs_list, progress_callback)

    if result["success"]:
        answer_json = result.get("answer_json") or {
            "response": result.get("answer", ""),
            "citations": result.get("citations", []),
        }
        n_citations = len(answer_json.get("citations", []))
        _add_log(f"Answer generated successfully — {n_citations} citation(s).")
        st.session_state.answer = answer_json
    else:
        error_msg = result.get("error", "Unknown error")
        _add_log(f"Query failed: {error_msg}", "ERROR")
        st.session_state.answer = {
            "response": error_msg,
            "citations": [],
            "_error": True,
        }

    st.session_state.query_running = False
    _add_log("Query processing complete.")


# MAIN LAYOUT

st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0; border-bottom: 1px solid #222; margin-bottom: 1.5rem;">
  <h1 style="margin:0; font-size:1.6rem; font-weight:700; letter-spacing:-0.03em;">
    NOVEC RAG
  </h1>
  <p style="margin:0.25rem 0 0 0; color:#666; font-size:0.85rem;">
    Vector-less RAG powered by PageIndex + OpenAI
  </p>
</div>
""", unsafe_allow_html=True)

if not _backend_ok:
    st.error(f"**Backend import failed:** {_backend_err}\n\nMake sure `novec.py`, `config.py`, and all dependencies are present.")
    st.stop()

if not st.session_state.docs_loaded:
    with st.spinner("Loading documents…"):
        _load_documents()


# SECTION 1 — DOCUMENT MANAGER

doc_hdr_col, actions_col = st.columns([6, 2])
with doc_hdr_col:
    st.markdown('<div class="rag-card-title">&nbsp;Documents</div>', unsafe_allow_html=True)

with actions_col:
    btn1, btn2 = st.columns([1, 1], gap="small")  

    with btn1:
        if st.button("Refresh", key="refresh_docs", use_container_width=True):
            with st.spinner("Refreshing…"):
                _load_documents()
            st.rerun()

    with btn2:
        if st.button("Upload", key="open_upload_modal", type="primary", use_container_width=True):
            _upload_modal()

docs = st.session_state.documents

    
if not docs:
    st.markdown(
        '<p style="color:#555; font-size:0.85rem; padding:0.5rem 0;">No documents found. Upload a PDF to get started.</p>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<p style="color:#666; font-size:0.75rem; margin-bottom:0.75rem;">'
        f'Total: {len(docs)} document(s) available</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.selected_doc is None:
        st.markdown(
            '<span style="color:#666; font-size:0.8rem;">Select a document below before running a query.</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span style="color:#888; font-size:0.8rem;">1 document selected</span>',
            unsafe_allow_html=True
        )

    for doc in docs:
        doc_id     = doc.get("id", "")
        doc_name   = doc.get("name", "Unnamed")
        doc_status = doc.get("status", "unknown")

        col1, col2, col3 = st.columns([0.70, 0.17, 0.12], gap="small")

        with col1:
            selected = st.radio(
                doc_name,
                options=[doc_id],
                index=0 if st.session_state.selected_doc == doc_id else None,
                key=f"radio_{doc_id}",
                label_visibility="collapsed",
                format_func=lambda x: doc_name,
            )
            if selected == doc_id:
                st.session_state.selected_doc = doc_id

        with col2:
            st.markdown(_badge(doc_status), unsafe_allow_html=True)

        with col3:
            if st.button("🗑", key=f"del_{doc_id}", help="Delete document", use_container_width=True):
                _delete_modal(doc_id, doc_name)

# SECTION 2 — USER QUERY INPUT
st.markdown('<div class="rag-card"></div>', unsafe_allow_html=True)
st.markdown('<div class="rag-card-title">&nbsp;Query</div>', unsafe_allow_html=True)

query_text = st.text_area(
    "Enter your question",
    placeholder="e.g. What are the key findings in the Q3 report?",
    height=90,
    key="query_input",
    label_visibility="collapsed",
)

run_btn = st.button(
    "Run",
    key="run_query_btn",
    disabled=st.session_state.query_running or not query_text.strip() or not st.session_state.selected_doc,
    type="primary",
)

st.markdown('</div>', unsafe_allow_html=True)

# HANDLE QUERY EXECUTION (after state is ready)
if run_btn and query_text.strip():
    selected_doc_objects = [
        d for d in st.session_state.documents
        if d.get("id") == st.session_state.selected_doc
    ]
    if not selected_doc_objects:
        st.warning("Please select a document before running a query.")
    else:
        with st.spinner("Processing query…"):
            _run_query(query_text.strip(), selected_doc_objects)
        st.rerun()

# SECTION 3 — RESULTS DISPLAY
st.markdown('<div class="rag-card"></div>', unsafe_allow_html=True)
st.markdown('<div class="rag-card-title">&nbsp;Answer</div>', unsafe_allow_html=True)

if st.session_state.query_running:
    st.markdown('<div class="upload-status">Generating answer…</div>', unsafe_allow_html=True)
elif st.session_state.answer:
    _render_answer(st.session_state.answer)
else:
    st.markdown(
        '<div style="color:#444; font-size:0.85rem; padding:0.5rem 0;">Results will appear here after you run a query.</div>',
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

# SECTION 4 — LOGGING PANEL
st.markdown('<div class="rag-card"></div>', unsafe_allow_html=True)
log_hdr_col, clear_log_col = st.columns([8, 1])
with log_hdr_col:
    st.markdown('<div class="rag-card-title">&nbsp;Logs</div>', unsafe_allow_html=True)
with clear_log_col:
    if st.button("Clear", key="clear_logs"):
        st.session_state.logs = []
        st.rerun()

_render_logs()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<hr><p style="text-align:center; color:#333; font-size:0.72rem; margin:0.5rem 0 1rem 0;">'
    'NOVEC RAG · Powered by PageIndex &amp; OpenAI</p>',
    unsafe_allow_html=True,
)