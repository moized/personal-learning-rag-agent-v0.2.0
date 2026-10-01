import requests
import streamlit as st

API = st.sidebar.text_input("Backend URL", "http://localhost:8000")
st.set_page_config(page_title="Personal Learning Knowledge Agent", layout="wide")
st.title("Personal Learning Knowledge Agent")

try:
    health = requests.get(f"{API}/health", timeout=5).json()
    st.caption(
        f"LLM={'on' if health.get('llm') else 'local fallback'} · "
        f"reasoning={health.get('reasoning_model')} · fast={health.get('fast_model')} · "
        f"embedding={health.get('embedding_model')} · retrieval={health.get('retrieval')}"
    )
except Exception:
    st.warning("Backend not reachable yet.")

courses = requests.get(f"{API}/courses", timeout=10).json() if requests.get(f"{API}/courses", timeout=10).ok else []
course_options = {c["name"]: c["id"] for c in courses}
selected_course_name = st.sidebar.selectbox("Course", ["(none)"] + list(course_options))
selected_course = course_options.get(selected_course_name)

st.header("1. Course")
with st.form("course_form"):
    c_name = st.text_input("Course name")
    c_domain = st.text_input("Domain (optional)")
    c_desc = st.text_area("Description (optional)")
    c_outline = st.text_area("Existing outline / syllabus (optional)")
    if st.form_submit_button("Create course") and c_name:
        st.json(requests.post(f"{API}/courses", json={"name": c_name, "domain": c_domain or None, "description": c_desc or None, "outline": c_outline or None}, timeout=30).json())

st.header("2. Add learning material")
st.caption("Hints improve routing but the system can infer the actual subject from the content.")

topic = st.text_input("Topic / subject hint (optional)", placeholder="e.g. RAG, retrieval, SQL, plasma physics")
domain = st.text_input("Domain hint (optional)", placeholder="e.g. AI / Information Retrieval")
description = st.text_area("Description / context (optional)", placeholder="What is this lecture/book about? Add anything useful, but it is not required.")

with st.expander("Paste structured metadata (optional)"):
    metadata = st.text_area("JSON or Python dictionary", placeholder="{'lecturer': '...', 'week': 4, 'topics': ['BM25', 'reranking']}", height=120)

uploaded = st.file_uploader(
    "Upload PDF / TXT / MD / DOCX / SRT / VTT / JSON",
    type=["pdf", "txt", "md", "docx", "srt", "vtt", "json"],
    accept_multiple_files=True,
)
if uploaded and st.button("Queue uploaded files"):
    for f in uploaded:
        files = {"file": (f.name, f.getvalue())}
        data = {
            "course_id": str(selected_course) if selected_course else "",
            "topic_hint": topic or "",
            "domain_hint": domain or "",
            "description": description or "",
            "metadata_json": metadata or "",
        }
        result = requests.post(f"{API}/sources/file", files=files, data=data, timeout=60)
        st.write(f.name, result.json())

st.subheader("Manual transcript / subtitle / structured paste")
manual_format = st.selectbox("Input format", ["plain", "srt", "vtt", "python_dict", "json"], help="python_dict is parsed safely with ast.literal_eval; arbitrary code is never executed.")
with st.form("manual_source"):
    t_name = st.text_input("Source / lecture name", placeholder="Lecture 1 — Introduction")
    t_payload = st.text_area(
        "Paste transcript / content / dictionary",
        height=300,
        placeholder=(
            "Plain transcript..."
            if manual_format == "plain" else
            "{'name': 'Lecture 1', 'topic': 'Retrieval', 'description': '...', 'transcript': '...'}"
            if manual_format == "python_dict" else
            '{"name":"Lecture 1","topic":"Retrieval","transcript":"..."}'
            if manual_format == "json" else
            "WEBVTT\n..."
        ),
    )
    submitted = st.form_submit_button("Queue manual source")
    if submitted and t_name and t_payload:
        r = requests.post(
            f"{API}/sources/manual",
            data={
                "name": t_name,
                "payload": t_payload,
                "input_format": manual_format,
                "course_id": str(selected_course) if selected_course else "",
                "domain_hint": domain or "",
                "topic_hint": topic or "",
                "description": description or "",
            },
            timeout=60,
        )
        st.json(r.json())

st.subheader("YouTube video / playlist")
url = st.text_input("YouTube URL")
y_name = st.text_input("YouTube source name (optional)")
if st.button("Queue YouTube") and url:
    r = requests.post(
        f"{API}/sources/youtube",
        params={"url": url, "name": y_name or None, "course_id": selected_course, "domain_hint": domain or None, "topic_hint": topic or None, "description": description or None, "metadata_json": metadata or None},
        timeout=60,
    )
    st.json(r.json())

st.header("3. Persistent processing queue")
if st.button("Refresh jobs"):
    st.json(requests.get(f"{API}/jobs", timeout=10).json())

st.header("4. Course knowledge / syllabus")
if selected_course and st.button("Course stats"):
    st.json(requests.get(f"{API}/courses/{selected_course}/stats", timeout=10).json())
if st.button("Generate syllabus"):
    st.markdown(requests.get(f"{API}/syllabus", params={"course_id": selected_course}, timeout=240).json()["syllabus"])

st.header("5. Study")
study_topic = st.text_input("Topic to study", placeholder="e.g. hybrid retrieval and reranking")
col1, col2 = st.columns(2)
with col1:
    if st.button("Build copy-paste study pack") and study_topic:
        result = requests.post(f"{API}/study-pack", json={"topic": study_topic, "course_id": selected_course, "top_k": 12}, timeout=240).json()
        st.text_area("Study text — paste into ChatGPT / Gemini", result["study_text"], height=600)
        st.json(result["sources"])
with col2:
    question = st.text_area("Ask the knowledge base", placeholder="Explain why BM25 and dense retrieval complement each other.")
    if st.button("Ask") and question:
        result = requests.post(f"{API}/study", json={"question": question, "course_id": selected_course, "top_k": 8}, timeout=240).json()
        st.markdown(result["answer"])
        st.caption(f"Groups: {result.get('selected_group_ids')} · queries: {result.get('query_plan', {}).get('search_queries', [])}")
        st.json(result["sources"])
