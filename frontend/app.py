import requests
import streamlit as st

API = st.sidebar.text_input("Backend URL", "http://localhost:8000")
st.set_page_config(page_title="Personal Learning Knowledge Agent", layout="wide")
st.title("Personal Learning Knowledge Agent")

try:
    health = requests.get(f"{API}/health", timeout=5).json()
    st.caption(
        f"AI mode={health.get('ai_mode')} · "
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
if selected_course and st.button("Course stats"):
    st.json(requests.get(f"{API}/courses/{selected_course}/stats", timeout=10).json())

st.header("4. Decisions waiting for your external AI")
st.caption(
    "In manual mode the system asks you instead of calling an AI API. Copy each prompt into ChatGPT, Gemini or "
    "Claude, paste the AI's full reply here, and the paused work continues."
)
try:
    pending = requests.get(f"{API}/decisions", timeout=10).json()
except Exception:
    pending = []
if not pending:
    st.success("Nothing is waiting for you.")
for i, d in enumerate(pending):
    with st.expander(f"#{d['id']} · {d.get('label') or d['task']}", expanded=i == 0):
        st.code(d["prompt"], language=None, wrap_lines=True)
        reply = st.text_area("Paste the AI's reply", key=f"reply-{d['id']}", height=200)
        if st.button("Submit answer", key=f"submit-{d['id']}") and reply.strip():
            r = requests.post(f"{API}/decisions/{d['id']}/answer", json={"answer": reply}, timeout=240)
            if r.ok:
                body = r.json()
                st.success(f"Saved. Resumed jobs: {body['requeued_jobs'] or 'none'}.")
                if body.get("result"):
                    st.json(body["result"])
            else:
                st.error(r.json().get("detail", r.text))

st.header("5. Study outline")
if st.button("Generate / refresh outline"):
    r = requests.post(f"{API}/syllabus", json={"course_id": selected_course}, timeout=240)
    if r.status_code == 202:
        st.info("The outline needs one decision: answer it in section 4 and the outline is saved automatically.")
    elif r.ok and r.json().get("status") == "empty":
        st.warning(r.json()["message"])
    elif not r.ok:
        st.error(r.text)
outline = requests.get(f"{API}/syllabus", params={"course_id": selected_course}, timeout=60).json()
topics = outline.get("topics") or []
titles = {t["number"]: t["title"] for t in topics}
if outline.get("status") == "ready":
    st.markdown(outline["syllabus"])
    st.dataframe(
        [{"#": t["number"], "Topic": t["title"], "Status": t["status"],
          "Parts sent": f"{t['parts_sent']}/{t['parts_total']}", "Excerpts": t["excerpts"]} for t in topics],
        hide_index=True,
    )
    with st.form("topic_status"):
        number = st.selectbox("Topic", list(titles), format_func=lambda n: f"{n}. {titles[n]}")
        status = st.selectbox("Set status", ["pending", "skipped", "done"])
        if st.form_submit_button("Update topic"):
            topic_id = next(t["id"] for t in topics if t["number"] == number)
            st.json(requests.post(f"{API}/syllabus/topics/{topic_id}/status", json={"status": status}, timeout=30).json())
else:
    st.caption("No outline yet. Add material, let it finish processing, then generate the outline.")

st.header("6. Study with your external AI")
if outline.get("status") != "ready":
    st.caption("Available once the study outline exists.")
else:
    msg_key = f"tutor_msg_{selected_course}"
    st.markdown("**Step 1.** Paste this into a new chat with your AI. It will ask where you want to start.")
    if st.button("Show kickoff message"):
        st.session_state[f"kickoff_{selected_course}"] = requests.get(
            f"{API}/tutor/kickoff", params={"course_id": selected_course}, timeout=60).json()["text"]
    if st.session_state.get(f"kickoff_{selected_course}"):
        st.code(st.session_state[f"kickoff_{selected_course}"], language=None, wrap_lines=True)

    st.markdown("**Step 2.** Paste the AI's `START: …; SKIP: …` line, or choose here.")
    with st.form("tutor_start"):
        start_reply = st.text_input("START/SKIP line from the AI", placeholder="START: 3; SKIP: 5, 7")
        start_number = st.selectbox("…or start from topic", list(titles), format_func=lambda n: f"{n}. {titles[n]}")
        skip_numbers = st.multiselect("Skip topics", list(titles), format_func=lambda n: f"{n}. {titles[n]}")
        if st.form_submit_button("Start"):
            payload = {"course_id": selected_course}
            payload.update({"reply": start_reply} if start_reply.strip() else {"start": start_number, "skip": skip_numbers})
            r = requests.post(f"{API}/tutor/start", json=payload, timeout=120)
            if r.ok:
                st.session_state[msg_key] = r.json()
            else:
                st.error(r.json().get("detail", r.text))

    st.markdown("**Step 3.** Paste each part into the chat, study it, then get the next one.")
    if st.button("Next part"):
        st.session_state[msg_key] = requests.post(f"{API}/tutor/next", json={"course_id": selected_course}, timeout=120).json()
    message = st.session_state.get(msg_key)
    if message:
        if message.get("done"):
            st.success(message["message"])
        else:
            st.caption(f"Topic {message['number']} — {message['title']} · part {message['part']} of {message['parts']} · "
                       f"{message['remaining_topics']} topic(s) still open")
            st.code(message["text"], language=None, wrap_lines=True)

st.header("7. Ask or build a one-off study pack")
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
        if result.get("answer"):
            st.markdown(result["answer"])
        else:
            st.caption("Manual mode: paste this prompt, evidence included, into your AI.")
            st.code(result["prompt"], language=None, wrap_lines=True)
        st.caption(f"Groups: {result.get('selected_group_ids')} · queries: {result.get('query_plan', {}).get('search_queries', [])}")
        st.json(result["sources"])
