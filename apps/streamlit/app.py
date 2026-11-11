"""Streamlit demonstrator for NuExtract3 Studio.

Run with:
    streamlit run apps/streamlit/app.py

The UI never talks to vLLM directly: it calls the FastAPI orchestrator, which
encodes documents, routes pages and validates outputs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.streamlit.api_client import ApiClient, ApiError
from nuextract_app.config import get_settings
from nuextract_app.core.templates import TemplateRegistry

DEFAULT_API_URL = os.getenv("NUEXTRACT_API_BASE_URL", "http://localhost:8080")
DOC_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"]

st.set_page_config(
    page_title="NuExtract3 Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, value in {
    "api_base_url": DEFAULT_API_URL,
    "template_editor": "{}",
    "extract_results": [],
    "md_result": None,
    "template_result": None,
    "health": None,
    "ready": None,
    "templates": None,
}.items():
    st.session_state.setdefault(key, value)

# A template generated in the "Template builder" tab is applied at the start of
# the next run, before the editor widget is instantiated.
if "pending_template" in st.session_state:
    st.session_state["template_editor"] = st.session_state.pop("pending_template")
    st.session_state["_last_template"] = "custom"
    st.session_state["template_selected"] = "custom"


def _local_templates() -> list[dict[str, Any]]:
    registry = TemplateRegistry(get_settings().templates_dir)
    return [
        {"name": info.name, "description": info.description, "template": info.template}
        for info in registry.list_templates()
    ]


@st.cache_data(ttl=120, show_spinner=False, max_entries=4)
def _load_templates(api_url: str) -> tuple[list[dict[str, Any]], str]:
    try:
        client = ApiClient(api_url, timeout_s=10)
        try:
            return client.list_templates(), "api"
        finally:
            client.close()
    except Exception:
        return _local_templates(), "local"


@st.cache_data(show_spinner=False, max_entries=24)
def _pdf_first_page(data: bytes) -> bytes | None:
    try:
        import pymupdf as fitz

        with fitz.open(stream=data, filetype="pdf") as doc:
            page = doc.load_page(0)
            return page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).tobytes("png")
    except Exception:
        return None


def render_preview(filename: str, data: bytes) -> None:
    if filename.lower().endswith(".pdf"):
        png = _pdf_first_page(data)
        if png:
            st.image(png, caption="Page 1", use_container_width=True)
        else:
            st.caption("Preview unavailable.")
    else:
        st.image(data, caption=filename, use_container_width=True)


def usage_of(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "prompt": int(usage.get("prompt_tokens", 0) or 0),
        "completion": int(usage.get("completion_tokens", 0) or 0),
        "total": int(usage.get("total_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 NuExtract3 Studio")
    st.caption("Structured extraction & document-to-Markdown")

    st.subheader("Backend")
    st.text_input("API base URL", key="api_base_url")

    if st.button("Check backend", use_container_width=True):
        client = ApiClient(st.session_state.api_base_url, timeout_s=15)
        try:
            st.session_state.health = client.health()
            ready, payload = client.ready()
            st.session_state.ready = {"ok": ready, **payload}
        except Exception as exc:
            st.session_state.health = None
            st.session_state.ready = {"ok": False, "status": str(exc)}
        finally:
            client.close()

    if st.button("Refresh templates", use_container_width=True):
        st.session_state.templates = None
        st.session_state.pop("template_selected", None)
        st.session_state.pop("_last_template", None)
        _load_templates.clear()
        st.rerun()

    health = st.session_state.health
    ready = st.session_state.ready
    if health:
        st.success(f"{health['model']}")
        device = health.get("device") or "auto"
        st.caption(f"provider: {health['provider']} · device: {device} · v{health['version']}")
        if health.get("torch_available"):
            st.caption("PyTorch available for in-process inference")
    if ready:
        if ready.get("ok"):
            resolved = ready.get("resolved_provider") or ready.get("provider")
            details = [f"backend: {resolved}"]
            if ready.get("device"):
                details.append(f"device: {ready['device']}")
            if ready.get("dtype"):
                details.append(f"dtype: {ready['dtype']}")
            st.success("Model backend ready")
            st.caption(" · ".join(details))
            if ready.get("note"):
                st.caption(str(ready["note"]))
        else:
            st.warning("Model backend not ready")
            if ready.get("status"):
                st.caption(str(ready["status"]))
            if ready.get("note"):
                st.caption(str(ready["note"]))

    st.divider()
    st.subheader("Processing")
    thinking = st.checkbox(
        "Reasoning mode",
        value=False,
        help="Slower but better on complex layouts. Keep off for invoices/receipts.",
    )
    force_vision = st.checkbox(
        "Force vision",
        value=False,
        help="Ignore native PDF text and always send page images.",
    )
    dpi = st.slider("Render DPI", min_value=100, max_value=300, value=200, step=25)
    max_pages = st.slider("Max pages", min_value=1, max_value=100, value=40, step=1)

    st.divider()
    st.caption("Model: [numind/NuExtract3](https://huggingface.co/numind/NuExtract3)")
    st.caption(f"API docs: {st.session_state.api_base_url}/docs")


# ---------------------------------------------------------------------------
# Templates (cached in session to avoid repeated failed calls)
# ---------------------------------------------------------------------------
if st.session_state.templates is None:
    with st.spinner("Loading templates…"):
        st.session_state.templates, st.session_state.template_source = _load_templates(
            st.session_state.api_base_url
        )

templates: list[dict[str, Any]] = st.session_state.templates
template_map = {item["name"]: item for item in templates}
template_names = list(template_map)
options = [*template_names, "custom"]

st.title("Document extraction with NuExtract3")
st.caption(
    "Upload a PDF, a scan or a photo. NuExtract3 handles OCR, layout understanding and "
    "schema-guided extraction in a single vision-language pass."
)

extract_tab, markdown_tab, builder_tab, about_tab = st.tabs(
    ["🔎 Extract", "📝 Document → Markdown", "🧩 Template builder", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Extract tab
# ---------------------------------------------------------------------------
with extract_tab:
    left, right = st.columns([1.0, 1.15], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Documents",
            type=DOC_TYPES,
            accept_multiple_files=True,
            key="extract_files",
        )

        selector_col, reset_col = st.columns([3, 1])
        with selector_col:
            selected = st.selectbox(
                "Template",
                options=options,
                key="template_selected",
                help="Pick a bundled schema or edit the JSON below.",
            )
        with reset_col:
            st.write("")
            if st.button("Reset", use_container_width=True, help="Restore the selected template"):
                st.session_state["template_editor"] = (
                    json.dumps(template_map[selected]["template"], indent=2, ensure_ascii=False)
                    if selected != "custom"
                    else "{}"
                )
                st.session_state["_last_template"] = selected

        if selected != "custom" and st.session_state.get("_last_template") != selected:
            st.session_state["template_editor"] = json.dumps(
                template_map[selected]["template"], indent=2, ensure_ascii=False
            )
            st.session_state["_last_template"] = selected

        st.text_area(
            "NuExtract template (JSON)",
            key="template_editor",
            height=340,
            help="Leaves are types: verbatim-string, string, number, date, enum list, …",
        )
        instructions = st.text_area(
            "Instructions (optional)",
            placeholder="e.g. Output dates as ISO-8601. Use the buyer as customer when no bill-to is present.",
            height=80,
        )

        extract_clicked = st.button("Extract documents", type="primary", use_container_width=True)

        if st.session_state.get("template_source") == "local":
            st.caption("⚠️ API offline — showing bundled templates from disk.")

    with right:
        st.subheader("Results")

        if extract_clicked:
            if not uploaded:
                st.warning("Upload at least one document.")
            else:
                try:
                    template_obj = json.loads(st.session_state["template_editor"])
                    if not isinstance(template_obj, (dict, list)):
                        raise ValueError("Template must be a JSON object or array.")
                except (json.JSONDecodeError, ValueError) as exc:
                    st.error(f"Invalid template JSON: {exc}")
                else:
                    client = ApiClient(st.session_state.api_base_url)
                    results: list[dict[str, Any]] = []
                    progress = st.progress(0.0, text="Processing…")
                    try:
                        for index, file in enumerate(uploaded, start=1):
                            data = file.getvalue()
                            progress.progress(
                                (index - 1) / len(uploaded),
                                text=f"Processing {file.name} ({index}/{len(uploaded)})…",
                            )
                            try:
                                payload = client.extract(
                                    file.name,
                                    data,
                                    template=template_obj,
                                    instructions=instructions or None,
                                    thinking=thinking,
                                    force_vision=force_vision,
                                    dpi=dpi,
                                    max_pages=max_pages,
                                )
                                results.append({"_ok": True, "_data": data, **payload})
                            except ApiError as exc:
                                results.append(
                                    {
                                        "_ok": False,
                                        "_data": data,
                                        "filename": file.name,
                                        "error": str(exc),
                                    }
                                )
                            except Exception as exc:  # network errors
                                results.append(
                                    {
                                        "_ok": False,
                                        "_data": data,
                                        "filename": file.name,
                                        "error": str(exc),
                                    }
                                )
                    finally:
                        client.close()
                    progress.empty()
                    st.session_state.extract_results = results

        results = st.session_state.get("extract_results") or []
        if not results:
            st.info("Results will appear here. Choose a template and click **Extract documents**.")
        for index, result in enumerate(results):
            with st.container(border=True):
                st.markdown(f"#### {result.get('filename', 'document')}")
                if not result.get("_ok"):
                    st.error(result.get("error", "Unknown error"))
                    continue

                usage = usage_of(result)
                metric_cols = st.columns(4)
                metric_cols[0].metric("Pages", result.get("page_count", "—"))
                metric_cols[1].metric("Scanned", "Yes" if result.get("is_scanned") else "No")
                metric_cols[2].metric("Tokens", f"{usage['total']:,}")
                metric_cols[3].metric("Latency", f"{result.get('latency_s', 0):.1f}s")

                payload_json = json.dumps(result.get("result"), ensure_ascii=False, indent=2)
                st.json(result.get("result"), expanded=2)

                download_col, preview_col = st.columns([1, 1])
                with download_col:
                    st.download_button(
                        "Download JSON",
                        data=payload_json,
                        file_name=f"{Path(result.get('filename', 'result')).stem}.json",
                        mime="application/json",
                        key=f"download_json_{index}",
                        use_container_width=True,
                    )
                with preview_col:
                    show_preview = st.checkbox(
                        "Page preview", key=f"preview_{index}", label_visibility="collapsed"
                    )
                    if show_preview:
                        render_preview(result.get("filename", ""), result["_data"])

                if result.get("warnings"):
                    with st.expander(f"⚠️ {len(result['warnings'])} warning(s)"):
                        for warning in result["warnings"]:
                            st.write(f"- {warning}")

                if result.get("pages"):
                    with st.expander("Page routing"):
                        st.dataframe(result["pages"], hide_index=True, use_container_width=True)

                st.caption(f"request_id: `{result.get('request_id', '—')}`")

# ---------------------------------------------------------------------------
# Markdown tab
# ---------------------------------------------------------------------------
with markdown_tab:
    st.subheader("Convert a document to Markdown")
    st.caption("Useful for OCR, RAG preprocessing, and preserving tables as HTML.")

    md_file = st.file_uploader(
        "Document", type=DOC_TYPES, accept_multiple_files=False, key="md_file"
    )
    render_html = st.checkbox("Render HTML tables", value=False)
    md_clicked = st.button("Convert to Markdown", type="primary")

    if md_clicked:
        if md_file is None:
            st.warning("Upload a document.")
        else:
            client = ApiClient(st.session_state.api_base_url)
            try:
                with st.spinner("Converting…"):
                    st.session_state.md_result = client.markdown(
                        md_file.name,
                        md_file.getvalue(),
                        thinking=thinking,
                        dpi=dpi,
                        max_pages=max_pages,
                    )
                    st.session_state.md_result["_data"] = md_file.getvalue()
            except Exception as exc:
                st.session_state.md_result = None
                st.error(str(exc))
            finally:
                client.close()

    md_result = st.session_state.get("md_result")
    if md_result:
        usage = usage_of(md_result)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Pages", md_result.get("page_count", "—"))
        metric_cols[1].metric("Tokens", f"{usage['total']:,}")
        metric_cols[2].metric("Latency", f"{md_result.get('latency_s', 0):.1f}s")

        preview_col, content_col = st.columns([1, 2], gap="large")
        with preview_col:
            render_preview(md_result.get("filename", ""), md_result["_data"])
        with content_col:
            st.download_button(
                "Download Markdown",
                data=md_result["markdown"],
                file_name=f"{Path(md_result.get('filename', 'document')).stem}.md",
                mime="text/markdown",
                key="download_markdown",
            )
            st.markdown(md_result["markdown"], unsafe_allow_html=render_html)
            with st.expander("Raw Markdown"):
                st.code(md_result["markdown"], language="markdown")

# ---------------------------------------------------------------------------
# Template builder tab
# ---------------------------------------------------------------------------
with builder_tab:
    st.subheader("Build an extraction template")
    st.caption(
        "Ask NuExtract3 to design a template from a description, or convert an existing "
        "JSON Schema / Pydantic schema."
    )

    prompt_col, schema_col = st.columns(2, gap="large")

    with prompt_col:
        st.markdown("**From a description**")
        prompt = st.text_area(
            "What do you want to extract?",
            placeholder="Key details from a commercial lease: parties, rent, deposit, term and renewal options.",
            height=120,
            key="template_prompt",
        )
        if st.button("Generate template", use_container_width=True):
            if len(prompt.strip()) < 3:
                st.warning("Describe the extraction first.")
            else:
                client = ApiClient(st.session_state.api_base_url)
                try:
                    with st.spinner("Generating…"):
                        payload = client.generate_template(prompt, thinking=thinking)
                        st.session_state.template_result = payload["template"]
                except Exception as exc:
                    st.error(str(exc))
                finally:
                    client.close()

    with schema_col:
        st.markdown("**From JSON Schema**")
        schema_text = st.text_area(
            "JSON Schema",
            value=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "vendor": {"type": "string"},
                        "invoice_date": {"type": "string", "format": "date"},
                        "total": {"type": "number"},
                        "currency": {"enum": ["USD", "EUR", "TND"]},
                    },
                },
                indent=2,
            ),
            height=120,
            key="schema_input",
        )
        if st.button("Convert schema", use_container_width=True):
            try:
                schema_obj = json.loads(schema_text)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON Schema: {exc}")
            else:
                client = ApiClient(st.session_state.api_base_url)
                try:
                    with st.spinner("Converting…"):
                        payload = client.convert_schema(schema_obj)
                        st.session_state.template_result = payload["template"]
                        for dropped in payload.get("dropped", []):
                            st.warning(dropped)
                except Exception as exc:
                    st.error(str(exc))
                finally:
                    client.close()

    if st.session_state.template_result is not None:
        st.divider()
        st.markdown("**Generated template**")
        st.json(st.session_state.template_result, expanded=3)
        if st.button("Use in Extract tab", type="primary"):
            st.session_state["pending_template"] = json.dumps(
                st.session_state.template_result, indent=2, ensure_ascii=False
            )
            st.rerun()

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with about_tab:
    st.subheader("How it works")
    st.markdown(
        """
1. **Ingestion** — the API accepts a PDF or image. PDFs are rendered page by page
   with PyMuPDF; native text layers are detected.
2. **Routing** — digital pages with a text layer are sent as *text* (cheaper, more
   precise). Scanned pages are sent as *images*. Mixed documents are handled per page.
3. **Inference** — a model backend serves `numind/NuExtract3`. On GPU this is
   vLLM; on CPU it can be Ollama or llama.cpp with the official GGUF build, or
   in-process Transformers. The JSON template and instructions travel as
   `chat_template_kwargs` (or Ollama message roles).
4. **Post-processing** — JSON is repaired (code fences, trailing commas), chunks are
   merged, and missing keys are filled so the output always matches the template.
5. **Serving** — FastAPI exposes `/v1/extract`, `/v1/extract/batch`, `/v1/markdown`,
   `/v1/templates/*`, `/health` and `/ready`. This Streamlit app is a pure client.
        """
    )
    st.markdown("**Links**")
    link_cols = st.columns(3)
    link_cols[0].link_button(
        "Model card", "https://huggingface.co/numind/NuExtract3", use_container_width=True
    )
    link_cols[1].link_button(
        "API docs", f"{st.session_state.api_base_url}/docs", use_container_width=True
    )
    link_cols[2].link_button(
        "NuExtract platform", "https://nuextract.ai/", use_container_width=True
    )

    st.markdown("**Demo limitations**")
    st.markdown(
        """
- Scanned documents are capped by **Max pages** and **Render DPI**.
- Long PDFs are split into chunks of `pages_per_request`; list fields are concatenated.
- Outputs are not guaranteed to be valid against arbitrary JSON Schema constraints.
- This demo is unauthenticated: do not expose it publicly without adding auth.
        """
    )
