// Personal Learning: the student page. Talks to the FastAPI backend on the same origin.
// No framework or build step: small render functions, event delegation and polling.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// ---------- Safe HTML templates: every interpolated value is escaped unless it is itself html``.
class Html { constructor(s) { this.s = s; } }
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const fmt = (v) => v instanceof Html ? v.s : Array.isArray(v) ? v.map(fmt).join("")
  : v == null || v === false ? "" : String(v).replace(/[&<>"']/g, (c) => ESC[c]);
const html = (strings, ...values) => new Html(strings.reduce((out, s, i) => out + s + (i < values.length ? fmt(values[i]) : ""), ""));
const setHtml = (el, h) => { el.innerHTML = h.s; };

// ---------- API
class ApiError extends Error { constructor(message, status) { super(message); this.status = status; } }

async function api(path, { method = "GET", json, form, params } = {}) {
  const query = new URLSearchParams(Object.entries(params || {}).filter(([, v]) => v !== null && v !== undefined && v !== "")).toString();
  let res;
  try {
    res = await fetch(query ? `${path}?${query}` : path, {
      method,
      headers: json ? { "Content-Type": "application/json" } : undefined,
      body: json ? JSON.stringify(json) : form,
    });
  } catch {
    throw new ApiError("Can't reach the app. Is it still running?", 0);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    throw new ApiError(typeof detail === "string" ? detail : "Something went wrong. Please check what you entered.", res.status);
  }
  return { status: res.status, body };
}

// ---------- State
const TABS = ["material", "plan", "study"];
const state = {
  mode: "manual",
  courses: [],
  courseId: null,
  tab: "material",
  sources: [],
  decisions: [],
  plan: null,
  current: null,
};
const copyTexts = new Map();
const lastSourceState = new Map();
const signatures = { sources: "", plan: "" };

const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, String(value)); } catch { /* private mode */ } },
};

// ---------- Copy text shown to students
const MODE_NOTE = {
  manual: "No AI key is set up, so now and then you'll copy a request into your own AI (ChatGPT, Gemini or Claude) and paste its answer back.",
  gemini: "Connected to Gemini: sorting and planning happen automatically.",
  local: "Offline mode: material is organized without an AI, so topics follow your sources.",
};
const SOURCE_CHIP = { ready: "Ready", working: "Working", queued: "Waiting in line", needs_help: "Needs your help", failed: "Problem" };
const KIND = { file: "FILE", text: "TEXT", video: "VIDEO" };
const TOPIC_STATUS = { pending: "Not started", in_progress: "In progress", done: "Done", skipped: "Skipped" };
const WHY = {
  organize: "Your AI will sort these passages into topics and name the key ideas in each.",
  syllabus: "Your AI will turn your topics into a study plan, ordered from the basics to the advanced parts.",
};
const AI_LINKS = [["ChatGPT", "https://chatgpt.com/"], ["Gemini", "https://gemini.google.com/app"], ["Claude", "https://claude.ai/new"]];
const FILE_TYPES = [".pdf", ".docx", ".txt", ".md", ".srt", ".vtt", ".json"];

// ---------- Small UI helpers
function toast(message, kind = "info") {
  const box = $("#toasts");
  while (box.childElementCount >= 2) box.firstElementChild.remove();
  const el = document.createElement("div");
  el.className = kind === "error" ? "toast error" : "toast";
  el.textContent = message;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 4000);
}

async function busy(button, task) {
  if (button?.getAttribute("aria-busy") === "true") return;
  button?.setAttribute("aria-busy", "true");
  if (button) button.disabled = true;
  try {
    await task();
  } catch (error) {
    toast(error.message || "Something went wrong.", "error");
  } finally {
    if (button?.isConnected) {
      button.removeAttribute("aria-busy");
      button.disabled = false;
    }
  }
}

async function copy(key, button) {
  const text = copyTexts.get(key);
  if (text == null) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = Object.assign(document.createElement("textarea"), { value: text, readOnly: true });
    area.style.cssText = "position:fixed;opacity:0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  button.dataset.label ??= button.textContent;
  button.textContent = "Copied ✓";
  setTimeout(() => { if (button.isConnected) button.textContent = button.dataset.label; }, 2000);
}

const aiLinks = () => html`<div class="ai-links">${AI_LINKS.map(([name, url]) =>
  html`<a class="btn btn-sm" href="${url}" target="_blank" rel="noopener">Open ${name}</a>`)}</div>`;

const emptyCard = (title, text, tab, label) => html`
  <div class="card empty-card">
    <h2>${title}</h2>
    <p class="muted">${text}</p>
    <button class="btn btn-primary" type="button" data-go="${tab}">${label}</button>
  </div>`;

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
// form.name / form.title are the form's own properties, so read inputs through `elements`.
const field = (form, name) => form.elements.namedItem(name).value;
const topicHint = () => $("#topic-hint").value.trim();

// ---------- Courses
async function loadCourses(selectId) {
  state.courses = (await api("/courses")).body;
  const has = state.courses.length > 0;
  $("#welcome").hidden = has;
  $("#workspace").hidden = !has;
  $("#course-switch").hidden = !has;
  $("#mode-footer").hidden = !has;
  if (!has) {
    state.courseId = null;
    $("#welcome-name").focus();
    return;
  }
  const wanted = selectId ?? Number(store.get("courseId"));
  state.courseId = state.courses.some((c) => c.id === wanted) ? wanted : state.courses[0].id;
  setHtml($("#course"), html`${state.courses.map((c) => html`<option value="${c.id}">${c.name}</option>`)}`);
  $("#course").value = String(state.courseId);
  await openCourse();
}

async function openCourse() {
  store.set("courseId", state.courseId);
  Object.assign(state, { sources: [], decisions: [], plan: null, current: null });
  signatures.sources = signatures.plan = "";
  lastSourceState.clear();
  copyTexts.clear();
  $("#material-help").replaceChildren();
  $("#plan-help").replaceChildren();
  $("#ask-result").replaceChildren();
  await refreshMaterial(true);
  await setTab(tabFromHash());
}

async function createCourse(form) {
  const name = field(form, "name").trim();
  if (!name) return;
  const { body } = await api("/courses", { method: "POST", json: { name } });
  form.reset();
  $("#course-dialog").close();
  history.replaceState(null, "", "#material");
  await loadCourses(body.course_id);
}

// ---------- Tabs
const tabFromHash = () => {
  const hash = location.hash.slice(1);
  return TABS.includes(hash) ? hash : (store.get("tab") || "material");
};

async function setTab(tab) {
  state.tab = TABS.includes(tab) ? tab : "material";
  store.set("tab", state.tab);
  if (location.hash.slice(1) !== state.tab) history.replaceState(null, "", `#${state.tab}`);
  for (const t of TABS) {
    const button = $(`#tab-${t}`);
    button.setAttribute("aria-selected", String(t === state.tab));
    button.tabIndex = t === state.tab ? 0 : -1;
    $(`#panel-${t}`).hidden = t !== state.tab;
  }
  if (state.tab === "plan") await refreshPlan();
  if (state.tab === "study") {
    await refreshPlan();
    await refreshStudy();
  }
}

function renderDots() {
  $("#tab-material .dot").hidden = !state.decisions.some((d) => d.action !== "syllabus");
  $("#tab-plan .dot").hidden = !state.decisions.some((d) => d.action === "syllabus");
}

// ---------- Step 1: material
async function refreshMaterial(first = false) {
  if (!state.courseId) return;
  const id = state.courseId;
  const [sources, decisions] = await Promise.all([
    api(`/courses/${id}/sources`),
    api("/decisions", { params: { course_id: id } }),
  ]);
  if (id !== state.courseId) return;
  const becameReady = announce(sources.body, first);
  const planAsked = state.decisions.some((d) => d.action === "syllabus");
  state.sources = sources.body;
  state.decisions = decisions.body;
  renderSources();
  renderHelp();
  renderDots();
  renderMaterialNext();
  const planStillAsked = state.decisions.some((d) => d.action === "syllabus");
  if (state.tab === "plan" && (becameReady || planAsked !== planStillAsked)) await refreshPlan();
}

function announce(sources, first) {
  let becameReady = false;
  for (const s of sources) {
    const before = lastSourceState.get(s.id);
    lastSourceState.set(s.id, s.state);
    if (first || !before || before === s.state) continue;
    if (s.state === "ready") { toast(`"${s.name}" is ready.`); becameReady = true; }
    // On step 1 the help card is already on screen.
    if (s.state === "needs_help" && state.tab !== "material") toast(`"${s.name}" needs your AI's help (step 1).`);
    if (s.state === "failed") toast(`"${s.name}" could not be processed.`, "error");
  }
  return becameReady;
}

function renderSources() {
  const sig = JSON.stringify(state.sources);
  if (sig === signatures.sources) return;
  signatures.sources = sig;
  const list = $("#source-list");
  if (!state.sources.length) {
    setHtml(list, html`<li class="empty">Nothing here yet. Add a file, some text or a video above.</li>`);
    return;
  }
  setHtml(list, html`${state.sources.map((s) => {
    const pct = s.progress == null ? null : Math.round(s.progress * 100);
    const detail = s.state === "ready" ? s.summary : s.state === "queued" ? "" : s.step;
    return html`
      <li class="source" data-state="${s.state}">
        <span class="source-icon" aria-hidden="true">${KIND[s.kind] || "FILE"}</span>
        <div>
          <div class="source-name">${s.name}</div>
          <div class="source-step"><span class="chip chip-${s.state}">${SOURCE_CHIP[s.state] || s.state}</span>${detail ? html`<span>${detail}</span>` : ""}</div>
          ${pct == null ? "" : html`<div class="bar" role="progressbar" aria-label="${s.step}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><span style="width:${pct}%"></span></div>`}
          ${s.state === "needs_help" ? html`<p class="small muted">Answer the request at the top of this page to continue.</p>` : ""}
          ${s.state === "failed" ? html`<p class="source-error">${s.error}</p>
            <div class="actions"><button class="btn btn-sm" type="button" data-action="retry" data-id="${s.id}">Try again</button></div>` : ""}
        </div>
      </li>`;
  })}`);
}

function renderMaterialNext() {
  const ready = state.sources.some((s) => s.state === "ready");
  const busyNow = state.sources.some((s) => ["queued", "working", "needs_help"].includes(s.state));
  const el = $("#material-next");
  if (!ready || busyNow) {
    el.replaceChildren();
    return;
  }
  if (el.childElementCount) return;
  setHtml(el, html`
    <div class="callout success">
      <p><strong>Your material is ready.</strong> Next, turn it into a study plan.</p>
      <button class="btn btn-primary" type="button" data-go="plan">Go to study plan →</button>
    </div>`);
}

function setAddMode(mode) {
  for (const b of $$("[data-add]")) b.setAttribute("aria-selected", String(b.dataset.add === mode));
  for (const p of $$("[data-add-panel]")) p.hidden = p.dataset.addPanel !== mode;
}

async function uploadFiles(files) {
  if (!files.length || !state.courseId) return;
  $("#dropzone").setAttribute("aria-busy", "true");
  try {
    for (const file of files) {
      if (!FILE_TYPES.some((ext) => file.name.toLowerCase().endsWith(ext))) {
        toast(`"${file.name}" isn't a supported file. Use PDF, Word, text, Markdown or subtitle files.`, "error");
        continue;
      }
      const form = new FormData();
      form.append("file", file);
      form.append("course_id", state.courseId);
      if (topicHint()) form.append("topic_hint", topicHint());
      try {
        const { body } = await api("/sources/file", { method: "POST", form });
        toast(body.deduplicated ? `"${file.name}" was already added.` : `Added "${file.name}".`);
      } catch (error) {
        toast(`Couldn't add "${file.name}": ${error.message}`, "error");
      }
    }
  } finally {
    $("#dropzone").removeAttribute("aria-busy");
    $("#file-input").value = "";
  }
  await refreshMaterial();
  poll();
}

function detectFormat(text) {
  const t = text.trim();
  if (/^WEBVTT/i.test(t)) return "vtt";
  if (/^\d+\s*\r?\n\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->/m.test(t)) return "srt";
  if (t.startsWith("{")) {
    try { JSON.parse(t); return "json"; } catch { return "python_dict"; }
  }
  return "plain";
}

async function addPaste(form) {
  const text = field(form, "text");
  if (!text.trim()) return;
  const firstLine = text.trim().split("\n")[0].slice(0, 60).trim();
  const title = field(form, "title").trim() || firstLine || "Pasted notes";
  const send = (format) => {
    const data = new FormData();
    data.append("name", title);
    data.append("payload", text);
    data.append("input_format", format);
    data.append("course_id", state.courseId);
    if (topicHint()) data.append("topic_hint", topicHint());
    return api("/sources/manual", { method: "POST", form: data });
  };
  const format = detectFormat(text);
  let result;
  try {
    result = await send(format);
  } catch (error) {
    // Text that merely starts with "{" is still fine as plain notes.
    if (error.status !== 400 || format === "plain") throw error;
    result = await send("plain");
  }
  toast(result.body.deduplicated ? "This text was already added." : `Added "${title}".`);
  form.reset();
  await refreshMaterial();
  poll();
}

async function addVideo(form) {
  const url = field(form, "url").trim();
  await api("/sources/youtube", { method: "POST", params: { url, course_id: state.courseId, topic_hint: topicHint() } });
  toast("Added the video. Getting its transcript…");
  form.reset();
  await refreshMaterial();
  poll();
}

// ---------- "Needs your help" cards (paste-back decisions)
function renderHelp() {
  syncHelp($("#material-help"), state.decisions.filter((d) => d.action !== "syllabus"));
  syncHelp($("#plan-help"), state.decisions.filter((d) => d.action === "syllabus"));
}

function syncHelp(container, decisions) {
  const ids = new Set(decisions.map((d) => String(d.id)));
  for (const card of $$(".help-card", container)) if (!ids.has(card.dataset.id)) card.remove();
  for (const d of decisions) {
    // Existing cards are left alone so a half-pasted reply is never wiped by a refresh.
    if ($(`.help-card[data-id="${d.id}"]`, container)) continue;
    copyTexts.set(`decision-${d.id}`, d.prompt);
    container.insertAdjacentHTML("beforeend", helpCard(d).s);
  }
}

const helpCard = (d) => html`
  <article class="card help-card" data-id="${d.id}">
    <span class="chip chip-needs_help">Needs your help</span>
    <h3>${d.label || "Your AI's help is needed"}</h3>
    <p class="help-later muted small">Up next, once the request above is answered.</p>
    <p class="help-intro muted">${WHY[d.task] || "Your AI needs to make a decision for this step."} This app doesn't use an AI by itself, so you pass the request along.</p>
    <ol class="help-steps">
      <li><div>
        <strong>Copy the request</strong>
        <div class="actions"><button class="btn btn-primary" type="button" data-copy="decision-${d.id}">Copy request</button></div>
        <details><summary>See the request</summary><pre class="preview">${d.prompt}</pre></details>
      </div></li>
      <li><div>
        <strong>Paste it into a new chat with your AI</strong>
        ${aiLinks()}
      </div></li>
      <li><div>
        <strong>Paste the AI's whole reply here</strong>
        <label class="sr-only" for="reply-${d.id}">The AI's reply</label>
        <textarea id="reply-${d.id}" rows="5" placeholder="Paste everything the AI answered, including the part in { curly brackets }"></textarea>
        <p class="help-error" role="alert" hidden></p>
        <div class="actions"><button class="btn btn-primary" type="button" data-action="send-reply" data-id="${d.id}">Send reply</button></div>
      </div></li>
    </ol>
  </article>`;

async function sendReply(id, button) {
  const card = button.closest(".help-card");
  const error = $(".help-error", card);
  const text = $("textarea", card).value;
  const fail = (message) => { error.textContent = message; error.hidden = false; };
  error.hidden = true;
  if (!text.trim()) return fail("Paste the AI's reply first.");
  await busy(button, async () => {
    try {
      await api(`/decisions/${id}/answer`, { method: "POST", json: { answer: text } });
    } catch (e) {
      if (e.status === 400) {
        return fail("That doesn't look like the AI's full answer. Copy its whole reply, including the part in { curly brackets }, "
          + "and paste it again. If the AI stopped halfway, ask it to continue first.");
      }
      throw e;
    }
    card.remove();
    toast("Thanks! Picking up where it left off.");
    await refreshMaterial();
    if (state.tab !== "material") await refreshPlan();
    poll();
  });
}

// ---------- Step 2: study plan
async function refreshPlan() {
  if (!state.courseId) return;
  const id = state.courseId;
  const plan = (await api("/syllabus", { params: { course_id: id } })).body;
  const current = plan.status === "ready" && plan.started
    ? (await api("/tutor/current", { params: { course_id: id } })).body.current : null;
  if (id !== state.courseId) return;
  state.plan = plan;
  state.current = current;
  renderPlan();
}

function renderPlan() {
  const plan = state.plan;
  const asked = state.decisions.some((d) => d.action === "syllabus");
  const hasReady = state.sources.some((s) => s.state === "ready");
  const sig = JSON.stringify([plan, asked, hasReady, state.current?.topic_id]);
  if (sig === signatures.plan) return;
  signatures.plan = sig;
  const el = $("#plan-body");
  const ready = plan?.status === "ready";

  if (!ready && asked) {
    setHtml(el, html`<div class="card"><p class="muted">Your study plan will appear here as soon as you send your AI's reply above.</p></div>`);
    return;
  }
  if (!ready && !hasReady) {
    setHtml(el, emptyCard("Your study plan is built from your material",
      "Add something in step 1 first. When it shows Ready, come back here.", "material", "Go to step 1"));
    return;
  }
  if (!ready) {
    setHtml(el, html`
      <div class="card">
        <h2>Create your study plan</h2>
        <p class="muted">All your material goes into an ordered list of topics, from the basics to the advanced parts. Each topic gets its own passages, so nothing is repeated.</p>
        ${state.mode === "manual" ? html`<p class="muted small">Your AI will be asked to help with this once.</p>` : ""}
        <button class="btn btn-primary btn-lg" type="button" data-action="make-plan">Create study plan</button>
      </div>`);
    return;
  }

  const topics = plan.topics;
  const count = (status) => topics.filter((t) => t.status === status).length;
  const done = count("done");
  const skipped = count("skipped");
  const open = count("pending") + count("in_progress");
  const begun = topics.some((t) => t.parts_sent > 0 || t.status === "done");
  const next = open === 0 ? "" : begun
    ? html`<div class="callout"><p><strong>${plural(open, "topic")} to go.</strong> Pick up where you left off.</p>
        <button class="btn btn-primary" type="button" data-go="study">Continue studying →</button></div>`
    : html`<div class="callout"><p><strong>Ready?</strong> Skip anything you already know, then start studying.</p>
        <button class="btn btn-primary" type="button" data-go="study">Start studying →</button></div>`;
  setHtml(el, html`
    <div class="card">
      <div class="plan-head">
        <div>
          <h2>Your study plan</h2>
          <p class="muted small">${plural(topics.length, "topic")} · ${done} done${skipped ? ` · ${skipped} skipped` : ""}</p>
        </div>
        <button class="btn" type="button" data-action="make-plan">Update plan</button>
      </div>
      ${plan.outdated ? html`
        <div class="callout warn">
          <p>You added material after making this plan. Update it to include the new material. Your progress is kept.</p>
        </div>` : ""}
      <ol class="topic-list">${topics.map(topicRow)}</ol>
    </div>
    ${next}`);
}

function topicRow(t) {
  const now = state.current?.topic_id === t.id && t.status !== "skipped";
  const status = now ? "in_progress" : t.status;
  const label = now ? "Studying now" : TOPIC_STATUS[t.status];
  const action = t.status === "skipped" ? ["pending", "Include"]
    : t.status === "done" ? ["pending", "Study again"] : ["skipped", "Skip"];
  return html`
    <li class="topic" data-status="${status}">
      <span class="topic-num" aria-hidden="true">${t.status === "done" && !now ? "✓" : t.number}</span>
      <div>
        <div class="topic-title"><span class="sr-only">Topic ${t.number}: </span>${t.title}</div>
        ${t.purpose ? html`<div class="muted small">${t.purpose}</div>` : ""}
        <div class="topic-meta">
          <span class="chip chip-${status}">${label}</span>
          ${t.prerequisites.length ? html`<span>Builds on ${t.prerequisites.join(", ")}</span>` : ""}
          ${t.parts_total > 1 ? html`<span>${t.parts_total} parts</span>` : ""}
        </div>
        ${t.concepts.length ? html`<div class="concepts">${t.concepts.slice(0, 6).map((c) => html`<span class="chip">${c}</span>`)}${
          t.concepts.length > 6 ? html`<span class="chip">+${t.concepts.length - 6} more</span>` : ""}</div>` : ""}
      </div>
      <div class="topic-actions">
        <button class="btn btn-sm" type="button" data-action="topic-status" data-id="${t.id}" data-status="${action[0]}"
          aria-label="${action[1]} topic ${t.number}">${action[1]}</button>
      </div>
    </li>`;
}

async function makePlan() {
  const { status, body } = await api("/syllabus", { method: "POST", json: { course_id: state.courseId } });
  if (status === 202) toast("Almost there: pass the request above to your AI to finish your plan.");
  else if (body.status === "empty") toast(body.message, "error");
  else toast("Your study plan is ready.");
  await refreshMaterial();
  await refreshPlan();
  if (status === 202) $("#plan-help").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- Step 3: study
async function refreshStudy() {
  if (state.plan?.status === "ready") {
    const id = state.courseId;
    const kickoff = (await api("/tutor/kickoff", { params: { course_id: id } })).body.text;
    if (id !== state.courseId) return;
    copyTexts.set("kickoff", kickoff);
  }
  renderStudy();
}

function renderStudy() {
  const el = $("#study-body");
  const plan = state.plan;
  if (plan?.status !== "ready") {
    setHtml(el, emptyCard("Studying starts from your study plan", "Create it in step 2 first.", "plan", "Go to step 2"));
    return;
  }
  const topics = plan.topics;
  const open = topics.filter((t) => t.status === "pending" || t.status === "in_progress").length;
  const begun = topics.some((t) => t.parts_sent > 0 || t.status === "done");
  const current = state.current;
  if (!current && begun && open === 0) {
    setHtml(el, html`
      <div class="card finished">
        <h2>You've worked through your whole study plan</h2>
        <p class="muted">Want to go over something again? Choose "Study again" next to a topic in your plan.</p>
        <button class="btn btn-primary" type="button" data-go="plan">Open study plan</button>
      </div>`);
    return;
  }
  if (!current && begun) {
    setHtml(el, html`
      <div class="card">
        <h2>Pick up where you left off</h2>
        <p class="muted">${plural(open, "topic")} still to study. Get the next one and paste it into your chat.</p>
        <div class="actions"><button class="btn btn-primary btn-lg" type="button" data-action="next">Get the next topic</button></div>
        <p class="muted small mt">New chat? <button class="btn-link" type="button" data-copy="kickoff">Copy the first message again</button>.</p>
      </div>`);
    return;
  }
  if (!current) {
    renderStart(el, topics);
    return;
  }
  copyTexts.set("current", current.text);
  const counted = topics.length - topics.filter((t) => t.status === "skipped").length;
  const done = topics.filter((t) => t.status === "done" && t.id !== current.topic_id).length;
  const pct = counted ? Math.round((done / counted) * 100) : 0;
  const more = current.part < current.parts;
  const last = !more && current.remaining_topics === 0;
  setHtml(el, html`
    <div class="card">
      <div class="study-progress">
        <p class="eyebrow">Topic ${current.number} of ${topics.length}${current.parts > 1 ? ` · Part ${current.part} of ${current.parts}` : ""}</p>
        <p class="muted small">${done} of ${counted} done</p>
      </div>
      <h2 class="part-title">${current.title}</h2>
      <div class="bar" role="progressbar" aria-label="Study progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><span style="width:${pct}%"></span></div>
      <p class="muted mt">Paste this into your chat with your AI. It holds the passages from your material for this ${current.parts > 1 ? "part" : "topic"}; your AI teaches from them and asks you a few questions at the end.</p>
      <div class="actions"><button class="btn btn-primary btn-lg" type="button" data-copy="current">Copy this ${current.parts > 1 ? "part" : "topic"}</button></div>
      <details><summary>See what you'll paste</summary><pre class="preview">${current.text}</pre></details>
    </div>
    <div class="callout">
      <p><strong>Finished with your AI?</strong> ${last ? "That was the last topic." : `Get the next ${more ? "part" : "topic"}.`}</p>
      <button class="btn btn-primary" type="button" data-action="next">${last ? "Finish" : more ? "Next part →" : "Next topic →"}</button>
    </div>
    <p class="muted small">Lost your chat? <button class="btn-link" type="button" data-copy="kickoff">Copy the first message again</button>.
      To skip topics or bring them back, use your <a href="#plan">study plan</a>.</p>`);
}

function renderStart(el, topics) {
  const choices = topics.filter((t) => t.status !== "done" && t.status !== "skipped");
  setHtml(el, html`
    <div class="card">
      <p class="eyebrow">First</p>
      <h2>Open a chat with your AI</h2>
      <p class="muted">Copy this first message into a new chat with ChatGPT, Gemini or Claude. It shares your study plan and asks the AI to check with you where to start.</p>
      <div class="actions">
        <button class="btn btn-primary" type="button" data-copy="kickoff">Copy first message</button>
        ${aiLinks()}
      </div>
      <details><summary>See the message</summary><pre class="preview">${copyTexts.get("kickoff") || ""}</pre></details>
    </div>
    <form class="card" data-form="start-line">
      <p class="eyebrow">Then</p>
      <h2>Tell us where you're starting</h2>
      <p class="muted">Once you've decided with your AI, it gives you a line like <code>START: 3; SKIP: 5, 7</code>. Paste it here.</p>
      <label class="sr-only" for="start-line">Line from your AI</label>
      <div class="row">
        <input id="start-line" name="line" required placeholder="START: 3; SKIP: 5, 7" autocomplete="off">
        <button class="btn btn-primary" type="submit">Start</button>
      </div>
      ${choices.length ? html`
        <details>
          <summary>Or choose it yourself</summary>
          <label for="start-topic">Start from</label>
          <div class="row">
            <select id="start-topic">${choices.map((t) => html`<option value="${t.number}">${t.number}. ${t.title}</option>`)}</select>
            <button class="btn" type="button" data-action="start-pick">Start here</button>
          </div>
          <p class="muted small">Topics before it are skipped. You can bring any topic back from your study plan.</p>
        </details>` : ""}
    </form>`);
}

async function startStudying(payload) {
  const { body } = await api("/tutor/start", { method: "POST", json: { course_id: state.courseId, ...payload } });
  await refreshPlan();
  if (!body.done) state.current = body;
  renderStudy();
  toast(body.done ? "Every topic is done or skipped." : `Starting with topic ${body.number}: ${body.title}`);
}

async function nextPart() {
  const { body } = await api("/tutor/next", { method: "POST", json: { course_id: state.courseId } });
  await refreshPlan();
  state.current = body.done ? null : body;
  renderStudy();
  $("#panel-study").scrollIntoView({ behavior: "smooth", block: "start" });
  if (!body.done) toast(`Ready to copy: ${body.title}${body.parts > 1 ? `, part ${body.part}` : ""}.`);
}

async function ask(form) {
  const question = field(form, "question").trim();
  if (!question) return;
  const { body } = await api("/study", { method: "POST", json: { question, course_id: state.courseId, top_k: 8 } });
  const out = $("#ask-result");
  if (body.answer) {
    setHtml(out, html`<div class="answer">${body.answer}</div>`);
    return;
  }
  copyTexts.set("ask", body.prompt);
  setHtml(out, html`
    <p class="muted small mt">We found the passages that answer this. Paste the message into your AI.</p>
    <div class="actions"><button class="btn btn-primary" type="button" data-copy="ask">Copy message for your AI</button></div>
    <details><summary>See the message</summary><pre class="preview">${body.prompt}</pre></details>`);
}

// ---------- Polling: quick while something is being processed, slow otherwise.
let pollTimer;
function poll() {
  clearTimeout(pollTimer);
  const active = state.sources.some((s) => s.state === "queued" || s.state === "working");
  const delay = document.hidden ? 20000 : active ? 2000 : 8000;
  pollTimer = setTimeout(async () => {
    try {
      if (state.courseId && !document.hidden) await refreshMaterial();
    } catch { /* try again next tick */ }
    poll();
  }, delay);
}

// ---------- Events
function onClick(event) {
  const el = event.target.closest("[data-action],[data-tab],[data-go],[data-copy],[data-add]");
  if (!el) return;
  const fail = (e) => toast(e.message, "error");
  if (el.dataset.tab) return void setTab(el.dataset.tab).catch(fail);
  if (el.dataset.go) {
    setTab(el.dataset.go).catch(fail);
    return void scrollTo({ top: 0, behavior: "smooth" });
  }
  if (el.dataset.copy) return void copy(el.dataset.copy, el);
  if (el.dataset.add) return setAddMode(el.dataset.add);
  const id = Number(el.dataset.id);
  switch (el.dataset.action) {
    case "new-course": return $("#course-dialog").showModal();
    case "close-dialog": return $("#course-dialog").close();
    case "retry":
      return void busy(el, async () => {
        await api(`/sources/${id}/retry`, { method: "POST" });
        await refreshMaterial();
        poll();
      });
    case "send-reply": return void sendReply(id, el);
    case "make-plan": return void busy(el, makePlan);
    case "topic-status":
      return void busy(el, async () => {
        await api(`/syllabus/topics/${id}/status`, { method: "POST", json: { status: el.dataset.status } });
        await refreshPlan();
      });
    case "start-pick": {
      const start = Number($("#start-topic").value);
      return void busy(el, () => startStudying({ start, skip: [] }));
    }
    case "next": return void busy(el, nextPart);
  }
}

function onSubmit(event) {
  const form = event.target.closest("form[data-form]");
  if (!form) return;
  event.preventDefault();
  const button = event.submitter || $("button[type=submit]", form);
  const run = {
    "create-course": () => createCourse(form),
    "add-paste": () => addPaste(form),
    "add-video": () => addVideo(form),
    "start-line": () => startStudying({ reply: field(form, "line") }),
    "ask": () => ask(form),
  }[form.dataset.form];
  if (run) busy(button, run);
}

function bindEvents() {
  document.addEventListener("click", onClick);
  document.addEventListener("submit", onSubmit);
  $("#course").addEventListener("change", (e) => {
    state.courseId = Number(e.target.value);
    openCourse().catch((err) => toast(err.message, "error"));
  });
  window.addEventListener("hashchange", () => {
    if (state.courseId) setTab(location.hash.slice(1)).catch((err) => toast(err.message, "error"));
  });
  $(".steps").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const next = TABS[(TABS.indexOf(state.tab) + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
    setTab(next).then(() => $(`#tab-${next}`).focus());
  });
  const dropzone = $("#dropzone");
  for (const type of ["dragenter", "dragover"]) dropzone.addEventListener(type, () => dropzone.classList.add("dragging"));
  for (const type of ["dragleave", "drop"]) dropzone.addEventListener(type, () => dropzone.classList.remove("dragging"));
  // Dropping a file anywhere should never navigate away from the page.
  for (const type of ["dragover", "drop"]) window.addEventListener(type, (e) => e.preventDefault());
  dropzone.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));
  $("#file-input").addEventListener("change", (e) => uploadFiles([...e.target.files]));
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.courseId) refreshMaterial().catch(() => {});
    poll();
  });
}

async function init() {
  bindEvents();
  try {
    state.mode = (await api("/health")).body.ai_mode || "manual";
  } catch { /* keep the default */ }
  for (const el of $$("[data-mode-note]")) el.textContent = MODE_NOTE[state.mode] || "";
  $("[data-ask-note]").textContent = state.mode === "gemini"
    ? "Ask a question about your material and get an answer based on it."
    : "Ask a question about your material. You'll get a message with the passages that answer it, ready to paste into your AI.";
  try {
    await loadCourses();
  } catch (error) {
    toast(error.message, "error");
  }
  $("#boot").remove();
  poll();
}

init();
