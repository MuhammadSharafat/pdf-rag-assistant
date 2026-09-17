import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        # Kick off the event and block until the send completes
        asyncio.run(send_rag_ingest_event(path))
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def fetch_run_detail(run_id: str) -> dict:
    # The events/{id}/runs list endpoint only returns bare status metadata.
    # The per-run endpoint includes the actual "output" field, which on a
    # failed run holds the exception name/message/stack.
    url = f"{_inngest_api_base()}/runs/{run_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data)


def wait_for_run_output(event_id: str, timeout_s: float = 300.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    completed_but_empty_since = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                output = run.get("output")
                if output:
                    return output
                # Status flipped to Completed but output hasn't attached yet
                # (a brief lag in the dev server). Give it a short grace
                # window before giving up and returning empty.
                if completed_but_empty_since is None:
                    completed_but_empty_since = time.time()
                elif time.time() - completed_but_empty_since > 5.0:
                    return {}
            if status in ("Failed", "Cancelled"):
                # Pull the detailed run record for the real error; fall back to
                # the bare run dict if that call itself fails for some reason.
                try:
                    detail_run = fetch_run_detail(run["run_id"])
                    detail = detail_run.get("output") or detail_run.get("error") or detail_run
                except Exception:
                    detail = run.get("output") or run.get("error") or run
                raise RuntimeError(f"Function run {status}: {detail}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])
            num_contexts = output.get("num_contexts", 0)

        st.caption(f"Retrieved {num_contexts} chunk(s) from the vector store")
        if not answer:
            st.warning("Got an empty answer. Raw output from the run, for debugging:")
            st.json(output)

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")