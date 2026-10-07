/* arXiv RAG Explorer front-end: vanilla JS, no build step, no external requests. */
"use strict";

(() => {
  // ------------------------------------------------------------------ icons (Lucide, ISC licence)
  const ICONS = {
    book: '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    history: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    code: '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    sparkles: '<path d="M9.94 15.5A2 2 0 0 0 8.5 14.06l-6.14-1.58a.5.5 0 0 1 0-.96L8.5 9.94A2 2 0 0 0 9.94 8.5l1.58-6.14a.5.5 0 0 1 .96 0L14.06 8.5A2 2 0 0 0 15.5 9.94l6.14 1.58a.5.5 0 0 1 0 .96L15.5 14.06a2 2 0 0 0-1.44 1.44l-1.58 6.14a.5.5 0 0 1-.96 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/>',
    image: '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21"/>',
    stop: '<rect width="14" height="14" x="5" y="5" rx="2"/>',
    copy: '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    external: '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    file: '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    layers: '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    braces: '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1"/><path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1"/>',
    lightbulb: '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    sort: '<path d="m3 16 4 4 4-4"/><path d="M7 20V4"/><path d="M11 4h10"/><path d="M11 8h7"/><path d="M11 12h4"/>',
    pen: '<path d="M12 20h9"/><path d="M16.38 3.62a1 1 0 0 1 3 3L7.37 18.64a2 2 0 0 1-.86.5l-2.87.84a.5.5 0 0 1-.62-.62l.84-2.87a2 2 0 0 1 .5-.86z"/>',
    shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    zap: '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    cpu: '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    hash: '<line x1="4" x2="20" y1="9" y2="9"/><line x1="4" x2="20" y1="15" y2="15"/><line x1="10" x2="8" y1="3" y2="21"/><line x1="16" x2="14" y1="3" y2="21"/>',
    alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    refresh: '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
    trash: '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',
    shuffle: '<path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.7-1.1 2-1.7 3.3-1.7H22"/><path d="m18 2 4 4-4 4"/><path d="M2 6h1.9c1.5 0 2.9.9 3.6 2.2"/><path d="M22 18h-5.9c-1.3 0-2.6-.7-3.3-1.8l-.5-.8"/><path d="m18 14 4 4-4 4"/>',
  };
  const icon = (name) => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;
  const hydrateIcons = (root = document) =>
    root.querySelectorAll("i[data-icon]").forEach((el) => { el.outerHTML = icon(el.dataset.icon); });

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);
  const fmtMs = (ms) => (ms == null ? "–" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
  const fmtNum = (n) => (n == null ? "–" : Number(n).toLocaleString());
  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage full / private mode */ } },
  };
  const timeAgo = (iso) => {
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(iso).toLocaleDateString();
  };

  // ------------------------------------------------------------------ state
  const S = {
    status: null, ready: false, running: false, controller: null,
    question: "", answerText: "", passages: [], plan: null, citations: [], response: null,
    genImage: false, renderQueued: false, reasoning: "",
  };
  const HISTORY_KEY = "rag-history-v1";
  const PREFS_KEY = "rag-prefs-v1";
  const CURATED = [
    "How can reward models combine human feedback with verifiable rewards?",
    "What techniques make diffusion models generate images in only a few steps?",
    "How do researchers detect hallucinations in large language models?",
    "How is Gaussian splatting used for 3D avatar reconstruction?",
    "How can LLM agents stay reliable on long-horizon tasks?",
    "What are new approaches to parameter-efficient fine-tuning of LLMs?",
  ];

  // ------------------------------------------------------------------ markdown (safe subset)
  const CARET = "⁣";

  function citeChips(inner) {
    const refs = [];
    for (const part of inner.split(/\s*[,;]\s*/)) {
      const [a, b] = part.split(/\s*[-–]\s*/).map(Number);
      if (!Number.isFinite(a)) continue;
      const hi = Number.isFinite(b) ? Math.min(b, a + 20) : a;
      for (let r = a; r <= hi; r++) if (r >= 1 && r <= S.passages.length && !refs.includes(r)) refs.push(r);
    }
    if (!refs.length) return null;
    return refs.map((r) => {
      const title = S.passages[r - 1] ? S.passages[r - 1].title : "";
      return `<button type="button" class="cite" data-ref="${r}" title="${esc(title)}">${r}</button>`;
    }).join("");
  }

  function inline(escaped) {
    return escaped
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?![*\w])/g, "$1<em>$2</em>")
      .replace(/\[(\d+(?:\s*[-–,;]\s*\d+)*)\]/g, (m, inner) => citeChips(inner) || m)
      .replace(CARET, '<span class="caret"></span>');
  }

  function renderMarkdown(src) {
    const lines = src.replace(/\r/g, "").split("\n");
    let html = "";
    let list = null;
    let para = [];
    const flush = () => { if (para.length) { html += `<p>${inline(esc(para.join(" ")))}</p>`; para = []; } };
    const close = () => { if (list) { html += `</${list}>`; list = null; } };
    for (const raw of lines) {
      const line = raw.trimEnd();
      let m;
      if (!line.trim()) { flush(); close(); continue; }
      if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { flush(); close(); html += "<hr>"; continue; }
      if ((m = line.match(/^\s*#{1,6}\s+(.*)$/))) { flush(); close(); html += `<h4>${inline(esc(m[1]))}</h4>`; continue; }
      if ((m = line.match(/^\s*[-*+•]\s+(.*)$/))) {
        flush(); if (list !== "ul") { close(); html += "<ul>"; list = "ul"; }
        html += `<li>${inline(esc(m[1]))}</li>`; continue;
      }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
        flush(); if (list !== "ol") { close(); html += "<ol>"; list = "ol"; }
        html += `<li>${inline(esc(m[1]))}</li>`; continue;
      }
      close();
      para.push(line.trim());
    }
    flush(); close();
    return html;
  }

  // ------------------------------------------------------------------ keyword highlighting
  const GENERIC = new Set("the and for with from that this are how what which why using based into their between model models method methods approach approaches data large language learning task tasks paper study new via can does".split(" "));

  function highlightWords(plan) {
    const words = new Set();
    for (const k of (plan && plan.keywords) || []) {
      for (let w of k.split(/[\s/]+/)) {
        w = w.replace(/[^\p{L}\p{N}-]/gu, "");
        if (w.length < 3 || GENERIC.has(w.toLowerCase())) continue;
        if (w.length > 4 && /s$/i.test(w)) w = w.slice(0, -1);
        words.add(w);
      }
    }
    return [...words].sort((a, b) => b.length - a.length);
  }

  function highlight(text, words) {
    if (!words.length) return esc(text);
    const pattern = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
    const re = new RegExp(`\\b(?:${pattern})\\w*`, "gi"); // whole word: stem + suffix
    let out = "";
    let last = 0;
    text.replace(re, (match, offset) => {
      out += esc(text.slice(last, offset)) + `<mark>${esc(match)}</mark>`;
      last = offset + match.length;
      return match;
    });
    return out + esc(text.slice(last));
  }

  function highlightJSON(obj) {
    const json = JSON.stringify(obj, null, 2).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return json.replace(
      /("(\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
      (m) => {
        let cls = "j-num";
        if (m.startsWith('"')) cls = m.endsWith(":") ? "j-key" : "j-str";
        else if (m === "true" || m === "false") cls = "j-bool";
        else if (m === "null") cls = "j-null";
        return `<span class="${cls}">${m}</span>`;
      },
    );
  }

  // ------------------------------------------------------------------ toasts
  function toast(message, type = "ok") {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `${icon(type === "error" ? "alert" : "check")}<div>${esc(message)}</div>`;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), type === "error" ? 7000 : 3500);
  }

  // ------------------------------------------------------------------ status polling
  function stepState(state) {
    return { starting: 0, waiting_for_ollama: 0, installing_runtime: 0, pulling_models: 1, indexing: 2, ready: 4, error: -1 }[state] ?? 0;
  }

  function applyStatus(s) {
    const wasReady = S.ready;
    S.status = s;
    S.ready = Boolean(s.ready);
    const pill = $("#status-pill");
    const pct = s.progress && s.progress.percent != null ? `${Math.round(s.progress.percent)}%` : "";
    const docs = s.index ? s.index.documents : null;
    const texts = {
      ready: docs != null ? `Ready · ${fmtNum(docs)} abstracts` : "Ready",
      indexing: `Indexing ${pct}`.trim(),
      pulling_models: `Downloading model ${pct}`.trim(),
      installing_runtime: `Installing Ollama ${pct}`.trim(),
      waiting_for_ollama: "Waiting for Ollama",
      starting: "Starting…",
      error: "Needs attention",
    };
    pill.dataset.state = s.state;
    $("#status-text").textContent = texts[s.state] || s.state;
    pill.title = s.message || "";

    $("#btn-ask").disabled = !S.ready || S.running;

    // Boot card
    const boot = $("#boot-card");
    const showBoot = !S.ready && !S.running;
    boot.classList.toggle("hidden", !showBoot);
    if (showBoot) {
      boot.classList.toggle("error", s.state === "error");
      $("#boot-title").textContent = s.state === "error" ? "Something needs your attention" : "Getting things ready";
      let msg = s.message || "";
      if (s.progress && s.state === "indexing") {
        const p = s.progress;
        if (p.total) msg += ` ${fmtNum(p.processed)} / ${fmtNum(p.total)}`;
        if (p.docs_per_s) msg += ` · ${p.docs_per_s} docs/s`;
        if (p.eta_s != null) msg += ` · about ${p.eta_s}s left`;
      }
      $("#boot-message").textContent = msg;
      const bar = $("#boot-bar");
      const hasPct = s.progress && s.progress.percent != null;
      bar.parentElement.classList.toggle("indeterminate", !hasPct && s.state !== "error");
      bar.style.width = hasPct ? `${s.progress.percent}%` : "0%";
      const current = s.state === "error" ? (s.ollama && s.ollama.version ? 2 : 0) : stepState(s.state);
      $$("#boot-steps li").forEach((li, i) => {
        li.classList.toggle("done", i < current);
        li.classList.toggle("active", i === current && s.state !== "error");
      });
    }

    // Copy that depends on the dataset/models
    if (s.index) {
      $("#hero-lead").textContent =
        `Grounded answers with citations from ${fmtNum(s.index.documents)} research abstracts (${s.index.dataset}), generated 100% locally by ${s.models.llm}.`;
    }
    $("#footer-models").textContent =
      `LangGraph · LangChain · Ollama (${s.models.llm} + ${s.models.embedding}) · Chroma · SQLite FTS5` +
      (s.models.reranker ? ` · ${s.models.reranker.split("/").pop()} reranker` : "");

    // Image toggle availability
    const img = s.image || {};
    const sw = $("#image-switch");
    const box = $("#gen-image");
    sw.classList.toggle("disabled", !img.available);
    box.disabled = !img.available;
    if (!img.available) box.checked = false;
    const setup = img.setup || {};
    const preparing = setup.state === "installing"
      ? ` · preparing local FLUX${setup.percent != null ? ` ${Math.round(setup.percent)}%` : ""}` : "";
    $("#image-provider").textContent = img.available ? `via ${img.label}${img.keyless ? " (keyless)" : ""}${preparing}` : "(not configured)";
    const chain = (img.chain || []).join(" → ");
    sw.title = img.available
      ? `Bonus: free image generation via ${chain || img.label}${img.model ? ` (${img.model})` : ""}; ` +
        (img.runs_after_answer ? "the local model starts once the answer is complete" : "runs in parallel with the text answer")
      : img.reason || "Image generation is not configured";

    if (!wasReady && S.ready) { loadExamples(); autoAskFromUrl(); }
    if ($("#dataset-modal").open) renderDatasetDetails();
  }

  async function pollStatus() {
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      applyStatus(await res.json());
    } catch {
      applyStatus({ state: "error", ready: false, message: "Cannot reach the server. Is it still running?", models: {}, image: {} });
    }
    setTimeout(pollStatus, S.ready ? 10000 : 1200);
  }

  // ------------------------------------------------------------------ shareable links (?q=...&image=1&k=5)
  function autoAskFromUrl() {
    const params = new URLSearchParams(location.search);
    const q = params.get("q");
    if (!q || S.autoAsked) return;
    S.autoAsked = true;
    const k = Number(params.get("k"));
    if (k >= 1 && k <= 10) { $("#topk").value = k; $("#topk-out").textContent = k; }
    if (params.get("image") === "1" && !$("#gen-image").disabled) $("#gen-image").checked = true;
    setQuery(q);
    ask(q);
  }

  function updateUrl(question) {
    const params = new URLSearchParams({ q: question });
    if (S.genImage) params.set("image", "1");
    try { history.replaceState(null, "", `?${params}`); } catch { /* ignore */ }
  }

  // ------------------------------------------------------------------ examples
  async function loadExamples() {
    const wrap = $("#examples");
    let questions = [];
    const dataset = (S.status && S.status.index && S.status.index.dataset) || "";
    if (/^arxiv_2\.9k/i.test(dataset)) {
      questions = [...CURATED].sort(() => Math.random() - 0.5).slice(0, 3);
    } else {
      try {
        const data = await (await fetch("/api/examples?n=3")).json();
        questions = data.titles.map((t) => `What is “${t}” about?`);
      } catch { /* examples are optional */ }
    }
    wrap.innerHTML =
      questions.map((q) => `<button type="button" class="chip" data-q="${esc(q)}">${icon("sparkles")}<span>${esc(q)}</span></button>`).join("") +
      `<button type="button" class="chip random" id="chip-random" title="Ask about a random paper from the dataset">${icon("shuffle")}<span>Random paper</span></button>`;
  }

  async function askRandom() {
    try {
      const data = await (await fetch("/api/examples?n=1")).json();
      if (!data.titles.length) return;
      const q = `What is the main contribution of the paper “${data.titles[0]}”?`;
      setQuery(q);
      ask(q);
    } catch { toast("Could not fetch a random paper", "error"); }
  }

  // ------------------------------------------------------------------ pipeline UI
  function setStep(step, status, ms) {
    const el = $(`.step[data-step="${step}"]`);
    if (!el) return;
    el.dataset.status = status;
    if (ms != null) $("small", el).textContent = fmtMs(ms);
  }

  function resetPipeline(withImage) {
    const afterAnswer = Boolean(S.status && S.status.image && S.status.image.runs_after_answer);
    const defaults = { understand: "query plan", retrieve: "hybrid search", rerank: "cross-encoder", generate: "grounded answer", image: afterAnswer ? "after answer" : "in parallel" };
    $$(".step").forEach((el) => { el.dataset.status = ""; $("small", el).textContent = defaults[el.dataset.step]; });
    $$(".image-only").forEach((el) => el.classList.toggle("hidden", !withImage));
    $("#pipeline").classList.remove("hidden");
  }

  // ------------------------------------------------------------------ rendering
  function resetResults() {
    $("#features").classList.add("hidden");
    $("#results").classList.remove("hidden");
    $("#answer").classList.add("placeholder");
    $("#answer").innerHTML = '<div class="skeleton-lines"><span></span><span></span><span></span></div>';
    $("#metrics").innerHTML = "";
    $("#reasoning").classList.add("hidden");
    $("#reasoning-text").textContent = "";
    $("#citations-card").classList.add("hidden");
    $("#analysis-card").classList.add("hidden");
    $("#image-card").classList.toggle("hidden", !S.genImage);
    if (S.genImage) {
      const label = (S.status && S.status.image && S.status.image.label) || "image API";
      const when = S.status && S.status.image && S.status.image.runs_after_answer ? "once the answer is ready" : "in parallel";
      $("#image-meta").textContent = label;
      $("#image-body").innerHTML = `<div class="image-frame loading">Generating an illustration with ${esc(label)} ${when}…</div>`;
    }
    $("#sources").innerHTML = '<div class="empty">Searching the index…</div>';
    $("#context-count").textContent = "";
    $("#json").innerHTML = "";
    $("#raw-json").classList.add("hidden");
  }

  function scheduleAnswerRender() {
    if (S.renderQueued) return;
    S.renderQueued = true;
    requestAnimationFrame(() => { S.renderQueued = false; renderAnswer(false); });
  }

  function renderAnswer(final) {
    const el = $("#answer");
    el.classList.remove("placeholder");
    el.innerHTML = renderMarkdown(S.answerText + (final ? "" : CARET));
  }

  function renderAnalysis(plan) {
    const queries = plan.queries_used && plan.queries_used.length ? plan.queries_used : plan.search_queries || [];
    const weights = plan.query_weights || [];
    const used = new Set((plan.bm25_keywords || []).map((k) => k.toLowerCase()));
    const planner = plan.method === "llm" ? "LLM structured output (JSON schema)" : "keyword heuristic (fallback)";
    const queryItems = queries.map((q, i) => {
      const w = weights[i];
      const badge = w != null ? `<span class="weight" title="Weight of this query in Reciprocal Rank Fusion">×${w}</span>` : "";
      return `<li>${esc(q)}${badge}</li>`;
    }).join("");
    const keywordChips = (plan.keywords || []).map((k) => {
      const isUsed = used.has(k.toLowerCase());
      const title = isUsed ? "Sent to BM25 keyword search" : "Not in the question, so not sent to BM25 (avoids guessed terms)";
      return `<span class="tag${isUsed ? "" : " unused"}" title="${title}">${esc(k)}</span>`;
    }).join("");
    $("#analysis").innerHTML = `
      ${plan.topic ? `<div class="row"><span class="label">Topic</span><span>${esc(plan.topic)}</span></div>` : ""}
      <div class="row"><span class="label">Search queries</span><div><ul>${queryItems}</ul>
        <div class="legend">${weights.slice(1).some((w) => w < 1)
          ? "Your question anchors retrieval; LLM rewrites only add recall (lower RRF weight)."
          : "Rewrites widen recall at full weight; the cross-encoder then re-scores every candidate against your question."}</div></div></div>
      ${keywordChips ? `<div class="row"><span class="label">Keywords</span><div><div class="tags">${keywordChips}</div>
        <div class="legend">Solid = used for BM25 · dashed = suggested by the planner but not in the question.</div></div></div>` : ""}
      <div class="row"><span class="label">Planner</span><span class="muted">${planner}</span></div>`;
    $("#analysis-meta").textContent = plan.ms != null ? fmtMs(plan.ms) : "";
    $("#analysis-card").classList.remove("hidden");
  }

  function renderSources() {
    const wrap = $("#sources");
    $("#context-count").textContent = S.passages.length ? `${S.passages.length}` : "";
    if (!S.passages.length) { wrap.innerHTML = '<div class="empty">No abstracts matched this question.</div>'; return; }
    const maxScore = Math.max(...S.passages.map((p) => p.score || 0)) || 1;
    const words = highlightWords(S.plan);
    wrap.innerHTML = S.passages.map((p) => {
      const rel = Math.max(4, Math.round((100 * (p.score || 0)) / maxScore));
      const id = p.url
        ? `<a href="${esc(p.url)}" target="_blank" rel="noopener" title="Open on arXiv">arXiv:${esc(p.doc_id)} ${icon("external")}</a>`
        : `<span>${esc(p.doc_id)}</span>`;
      const long = p.text.length > 420;
      return `<article class="source" id="src-${p.ref}" data-ref="${p.ref}" style="animation-delay:${(p.ref - 1) * 50}ms">
        <div class="source-head"><span class="ref">${p.ref}</span>
          <div class="source-title"><h3>${esc(p.title)}</h3><div class="source-meta">${id}</div></div></div>
        <div class="relevance">
          <div class="bar" title="Relevance relative to the top result (Reciprocal Rank Fusion: ${p.score})"><span style="width:${rel}%"></span></div>
          ${p.similarity != null ? `<span class="badge sim" title="Cosine similarity to the closest search query">cos ${Number(p.similarity).toFixed(2)}</span>` : ""}
          ${p.rerank_rank ? `<span class="badge rerank" title="Rank given by the cross-encoder reranker (score ${p.rerank_score})">rerank #${p.rerank_rank}</span>` : ""}
          ${p.dense_rank ? `<span class="badge semantic" title="Best rank in dense (vector) search">semantic #${p.dense_rank}</span>` : ""}
          ${p.keyword_rank ? `<span class="badge keyword" title="Rank in BM25 keyword search">keyword #${p.keyword_rank}</span>` : ""}
        </div>
        <p class="source-text${long ? " clamp" : ""}">${highlight(p.text, words)}</p>
        ${long ? '<button class="more" type="button">Show more</button>' : ""}
      </article>`;
    }).join("");
  }

  function markCited(citations) {
    const refs = new Set(citations.map((c) => c.ref));
    $$(".source").forEach((card) => {
      const cited = refs.has(Number(card.dataset.ref));
      card.classList.toggle("cited", cited);
      const rel = $(".relevance", card);
      if (cited && rel && !$(".badge.cited", rel)) rel.insertAdjacentHTML("beforeend", '<span class="badge cited">cited</span>');
    });
  }

  function renderCitations(citations) {
    S.citations = citations || [];
    $("#citations-card").classList.toggle("hidden", !S.citations.length);
    $("#citations-count").textContent = S.citations.length;
    $("#citations").innerHTML = S.citations.map((c) => {
      const id = c.url
        ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">arXiv:${esc(c.doc_id)} ${icon("external")}</a>`
        : esc(c.doc_id);
      return `<li data-ref="${c.ref}"><span class="ref">${c.ref}</span><div><div class="cit-title">${esc(c.title)}</div><div class="cit-id">${id}</div></div></li>`;
    }).join("");
    markCited(S.citations);
  }

  function renderImage(data) {
    const card = $("#image-card");
    card.classList.remove("hidden");
    const body = $("#image-body");
    if (data.error || !data.images || !data.images.length) {
      body.innerHTML = `<div class="notice warn">${icon("alert")}<div><b>No illustration this time.</b> ${esc(data.error || "The provider returned no image.")}<br><span class="small">The text answer is unaffected.</span></div></div>`;
      return;
    }
    const img = data.images[0];
    $("#image-meta").textContent = `${img.provider}${img.model ? ` · ${img.model}` : ""} · ${fmtMs(img.latency_ms)}`;
    body.innerHTML = `
      <div class="image-frame"><img src="${esc(img.url)}" alt="${esc(`Illustration for: ${S.question}`)}"></div>
      <div class="image-caption">
        <details><summary>Image prompt</summary><p>${esc(img.prompt)}</p></details>
        <a href="${esc(img.url)}" target="_blank" rel="noopener">Open full size ${icon("external")}</a>
      </div>`;
    const el = $("img", body);
    el.addEventListener("load", () => el.classList.add("loaded"));
    if (el.complete) el.classList.add("loaded");
  }

  function renderMetrics(m) {
    if (!m) return;
    const items = [
      ["clock", "Answer in", fmtMs(m.latency_ms)],
      ["zap", "First token", fmtMs(m.time_to_first_token_ms)],
      ["lightbulb", "Plan", fmtMs(m.analysis_ms)],
      ["search", "Retrieval", fmtMs(m.retrieval_ms)],
      ["sort", "Rerank", m.rerank_ms != null ? fmtMs(m.rerank_ms) : null],
      ["hash", "Tokens", m.prompt_tokens != null ? `${fmtNum(m.prompt_tokens)} → ${fmtNum(m.completion_tokens)}` : null],
      ["zap", "Speed", m.tokens_per_second != null ? `${m.tokens_per_second} tok/s` : null],
      ["image", "Image", m.image_ms != null ? fmtMs(m.image_ms) : null],
      ["cpu", "Model", m.model],
    ];
    $("#metrics").innerHTML = items
      .filter(([, , v]) => v != null && v !== "–")
      .map(([ic, label, v]) => `<span class="metric">${icon(ic)}${label} <b>${esc(v)}</b></span>`).join("");
  }

  function renderJSON(obj) {
    $("#raw-json").classList.remove("hidden");
    $("#json").innerHTML = highlightJSON(obj);
  }

  function showError(message) {
    toast(message, "error");
    $$(".step").forEach((el) => { if (el.dataset.status === "running") el.dataset.status = "error"; });
    const el = $("#answer");
    if (!S.answerText) {
      el.classList.remove("placeholder");
      el.innerHTML = `<div class="notice error">${icon("alert")}<div>${esc(message)}</div></div>`;
    }
  }

  // ------------------------------------------------------------------ asking
  async function* readSSE(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        let event = "message";
        const data = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
        }
        if (data.length) {
          try { yield { event, data: JSON.parse(data.join("\n")) }; } catch (e) { console.warn("Bad SSE payload", e); }
        }
      }
    }
  }

  function handleEvent(event, data) {
    switch (event) {
      case "step":
        setStep(data.step, data.status === "done" ? "done" : "running", data.status === "done" ? data.ms : null);
        if (data.step === "retrieve" && data.status === "done" && data.candidates) {
          $(`.step[data-step="retrieve"] small`).textContent = `${fmtMs(data.ms)} · ${data.candidates} found`;
        }
        break;
      case "analysis":
        S.plan = data;
        setStep("understand", "done", data.ms);
        renderAnalysis(data);
        break;
      case "retrieval":
        S.passages = data.passages || [];
        if (data.reranked) setStep("rerank", "done", data.ms);
        else setStep("rerank", "skipped");
        renderSources();
        break;
      case "thinking":
        S.reasoning += data.text;
        $("#reasoning").classList.remove("hidden");
        $("#reasoning-text").textContent = S.reasoning;
        break;
      case "token":
        S.answerText += data.text;
        scheduleAnswerRender();
        break;
      case "answer":
        S.answerText = data.answer;
        setStep("generate", "done", data.ms);
        renderAnswer(true);
        renderCitations(data.citations);
        break;
      case "image":
        setStep("image", data.error ? "error" : "done", data.ms);
        renderImage(data);
        break;
      case "final":
        S.response = data;
        renderMetrics(data.metrics);
        renderJSON(data);
        break;
      case "error":
        showError(data.message || "Something went wrong");
        break;
      default:
        break;
    }
  }

  function setRunning(running) {
    S.running = running;
    $("#btn-ask").disabled = running || !S.ready;
    $("#btn-ask").classList.toggle("loading", running);
    $("#btn-ask span").textContent = running ? "Answering…" : "Answer";
    $("#btn-stop").classList.toggle("hidden", !running);
  }

  async function ask(question) {
    question = (question || "").trim();
    if (!question || S.running) return;
    if (!S.ready) { toast("The index is not ready yet. Please wait a moment.", "error"); return; }
    Object.assign(S, { question, answerText: "", passages: [], plan: null, citations: [], response: null, reasoning: "" });
    S.genImage = $("#gen-image").checked && !$("#gen-image").disabled;
    setRunning(true);
    updateUrl(question);
    resetPipeline(S.genImage);
    resetResults();
    const pipeline = $("#pipeline");
    if (pipeline.getBoundingClientRect().top > window.innerHeight * 0.7) pipeline.scrollIntoView({ behavior: "smooth", block: "start" });

    S.controller = new AbortController();
    let completed = false;
    try {
      const res = await fetch("/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ query: question, top_k: Number($("#topk").value), generate_image: S.genImage }),
        signal: S.controller.signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = typeof err.detail === "string" ? err.detail : err.detail ? JSON.stringify(err.detail) : `HTTP ${res.status}`;
        throw new Error(detail);
      }
      for await (const { event, data } of readSSE(res)) {
        handleEvent(event, data);
        if (event === "final") completed = true;
      }
    } catch (err) {
      if (err.name === "AbortError") {
        toast("Stopped.");
        $$(".step").forEach((el) => { if (el.dataset.status === "running") el.dataset.status = ""; });
        if (S.answerText) renderAnswer(true);
      } else {
        showError(err.message || String(err));
      }
    } finally {
      S.controller = null;
      setRunning(false);
      if (completed) saveHistory();
    }
  }

  // ------------------------------------------------------------------ history
  function saveHistory() {
    const items = store.get(HISTORY_KEY, []).filter((h) => h.q !== S.question);
    items.unshift({ id: Date.now(), q: S.question, at: new Date().toISOString(), r: S.response });
    store.set(HISTORY_KEY, items.slice(0, 25));
    renderHistory();
  }

  function renderHistory() {
    const items = store.get(HISTORY_KEY, []);
    $("#history-list").innerHTML = items.length
      ? items.map((h) => `<li data-id="${h.id}"><div class="h-q">${esc(h.q)}</div><div class="h-meta">${timeAgo(h.at)} · ${h.r.citations.length} citations · ${fmtMs(h.r.metrics.latency_ms)}</div></li>`).join("")
      : '<li class="empty" style="border:0;cursor:default">Your questions will appear here.</li>';
  }

  function showResponse(question, r) {
    Object.assign(S, { question, answerText: r.answer, passages: r.sources || [], response: r, reasoning: "" });
    S.plan = r.query_analysis;
    S.genImage = Boolean(r.images && r.images.length) || Boolean(r.image_error);
    setQuery(question);
    resetPipeline(S.genImage);
    $$(".step").forEach((el) => { el.dataset.status = "done"; });
    const m = r.metrics || {};
    setStep("understand", "done", m.analysis_ms);
    setStep("retrieve", "done", m.retrieval_ms);
    setStep("rerank", m.rerank_ms != null ? "done" : "skipped", m.rerank_ms);
    setStep("generate", "done", m.generation_ms);
    resetResults();
    renderAnalysis({ ...r.query_analysis, ms: m.analysis_ms });
    renderSources();
    renderAnswer(true);
    renderCitations(r.citations);
    if (S.genImage) { setStep("image", r.image_error ? "error" : "done", m.image_ms); renderImage({ images: r.images, error: r.image_error }); }
    renderMetrics(m);
    renderJSON(r);
  }

  function openDrawer(open) {
    $("#history-drawer").classList.toggle("open", open);
    $("#history-drawer").setAttribute("aria-hidden", String(!open));
    $("#drawer-backdrop").classList.toggle("hidden", !open);
    if (open) renderHistory();
  }

  // ------------------------------------------------------------------ dataset dialog
  function renderDatasetDetails() {
    const s = S.status || {};
    const ix = s.index;
    const rows = [];
    if (ix) {
      const built = new Date(ix.built_at);
      rows.push(
        ["File", `<code>${esc(ix.dataset)}</code>`],
        ["Location", `<span class="small">${esc(ix.dataset_path)}</span>`],
        ["Documents", `${fmtNum(ix.documents)} indexed · ${fmtNum(ix.chunks)} chunks`],
        ["Skipped lines", `${fmtNum(ix.duplicates)} duplicates · ${fmtNum(ix.malformed)} malformed · ${fmtNum(ix.missing_text)} without abstract`],
        ["SHA-256", `<code>${esc(ix.sha256.slice(0, 16))}…</code>`],
        ["Embeddings", `${esc(s.models.embedding)} · ${ix.embedding_dimensions || "?"} dims`],
        ["Vector store", esc(ix.vector_store)],
        ["Keyword index", esc(ix.keyword_index)],
        ["Reranker", s.models.reranker ? esc(s.models.reranker)
          : s.models.reranker_state === "loading" ? "loading (first run downloads ~90 MB); answers work meanwhile"
          : `off${s.models.reranker_error ? ` (${esc(s.models.reranker_error)})` : ""}`],
        ["Built", `${built.toLocaleString()} in ${ix.build_seconds}s`],
      );
    } else {
      rows.push(["Status", esc(s.message || "No index yet")]);
    }
    rows.push(
      ["LLM", `${esc((s.models || {}).llm || "?")} via Ollama ${esc((s.ollama || {}).version || "")}`],
      ["Images", esc((s.image && (s.image.available ? (s.image.chain || [s.image.label]).join(" → ") : s.image.reason)) || "–")],
    );
    $("#dataset-details").innerHTML = `<dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
  }

  function uploadDataset(file) {
    if (!file) return;
    if (!/\.jsonl$/i.test(file.name)) { toast("Please choose a .jsonl file", "error"); return; }
    const bar = $("#upload-progress");
    bar.classList.remove("hidden");
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/dataset");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) $("span", bar).style.width = `${(100 * e.loaded) / e.total}%`; };
    xhr.onload = () => {
      bar.classList.add("hidden");
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) {
        toast(`Uploaded ${body.dataset}: the old index was discarded and a new one is being built.`);
        $("#dataset-modal").close();
        S.ready = false;
        pollNow();
      } else {
        toast(body.detail || `Upload failed (HTTP ${xhr.status})`, "error");
      }
    };
    xhr.onerror = () => { bar.classList.add("hidden"); toast("Upload failed", "error"); };
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  }

  async function rebuildIndex() {
    const res = await fetch("/api/reindex", { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (res.ok) {
      toast("Rebuilding the index from scratch…");
      $("#dataset-modal").close();
      S.ready = false;
      pollNow();
    } else {
      toast(body.detail || `Could not rebuild (HTTP ${res.status})`, "error");
    }
  }

  async function pollNow() {
    try { applyStatus(await (await fetch("/api/status", { cache: "no-store" })).json()); } catch { /* next poll will retry */ }
  }

  // ------------------------------------------------------------------ misc UI
  function setQuery(q) {
    const ta = $("#query");
    ta.value = q;
    autoSize();
  }

  function autoSize() {
    const ta = $("#query");
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 320)}px`;
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("rag-theme", theme); } catch { /* ignore */ }
    $("#btn-theme").innerHTML = icon(theme === "dark" ? "sun" : "moon");
  }

  function focusSource(ref, flash) {
    const card = document.getElementById(`src-${ref}`);
    if (!card) return;
    if (flash) {
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1300);
    }
  }

  async function copyText(text, what) {
    try { await navigator.clipboard.writeText(text); toast(`${what} copied`); } catch { toast("Copy failed", "error"); }
  }

  // ------------------------------------------------------------------ wiring
  function bind() {
    $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("#query").value); });
    $("#query").addEventListener("input", autoSize);
    $("#query").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask($("#query").value); }
    });
    $("#btn-stop").addEventListener("click", () => S.controller && S.controller.abort());

    const prefs = store.get(PREFS_KEY, {});
    if (prefs.topK) $("#topk").value = prefs.topK;
    if (prefs.image) $("#gen-image").checked = true;
    $("#topk-out").textContent = $("#topk").value;
    $("#topk").addEventListener("input", () => {
      $("#topk-out").textContent = $("#topk").value;
      store.set(PREFS_KEY, { ...store.get(PREFS_KEY, {}), topK: Number($("#topk").value) });
    });
    $("#gen-image").addEventListener("change", () => store.set(PREFS_KEY, { ...store.get(PREFS_KEY, {}), image: $("#gen-image").checked }));

    $("#examples").addEventListener("click", (e) => {
      const chip = e.target.closest(".chip");
      if (!chip) return;
      if (chip.id === "chip-random") { askRandom(); return; }
      setQuery(chip.dataset.q);
      ask(chip.dataset.q);
    });

    // Citation chips <-> context cards
    $("#answer").addEventListener("click", (e) => { const c = e.target.closest(".cite"); if (c) focusSource(c.dataset.ref, true); });
    $("#answer").addEventListener("mouseover", (e) => {
      const c = e.target.closest(".cite");
      $$(".source.hover").forEach((el) => el.classList.remove("hover"));
      if (c) { const card = document.getElementById(`src-${c.dataset.ref}`); if (card) card.classList.add("hover"); }
    });
    $("#answer").addEventListener("mouseleave", () => $$(".source.hover").forEach((el) => el.classList.remove("hover")));
    $("#citations").addEventListener("click", (e) => {
      if (e.target.closest("a")) return;
      const li = e.target.closest("li");
      if (li) focusSource(li.dataset.ref, true);
    });
    $("#sources").addEventListener("click", (e) => {
      const more = e.target.closest(".more");
      if (!more) return;
      const text = $(".source-text", more.parentElement);
      const clamped = text.classList.toggle("clamp");
      more.textContent = clamped ? "Show more" : "Show less";
    });

    $("#btn-copy-answer").addEventListener("click", () => copyText(S.answerText, "Answer"));
    $("#btn-copy-json").addEventListener("click", (e) => { e.preventDefault(); if (S.response) copyText(JSON.stringify(S.response, null, 2), "JSON"); });
    $("#btn-download-json").addEventListener("click", (e) => {
      e.preventDefault();
      if (!S.response) return;
      const url = URL.createObjectURL(new Blob([JSON.stringify(S.response, null, 2)], { type: "application/json" }));
      const a = Object.assign(document.createElement("a"), { href: url, download: "answer.json" });
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    // History drawer
    $("#btn-history").addEventListener("click", () => openDrawer(true));
    $("#btn-close-history").addEventListener("click", () => openDrawer(false));
    $("#drawer-backdrop").addEventListener("click", () => openDrawer(false));
    $("#btn-clear-history").addEventListener("click", () => { store.set(HISTORY_KEY, []); renderHistory(); });
    $("#history-list").addEventListener("click", (e) => {
      const li = e.target.closest("li[data-id]");
      if (!li) return;
      const item = store.get(HISTORY_KEY, []).find((h) => String(h.id) === li.dataset.id);
      if (item) { showResponse(item.q, item.r); openDrawer(false); }
    });

    // Dataset dialog
    const openDataset = () => { renderDatasetDetails(); $("#dataset-modal").showModal(); };
    $("#btn-dataset").addEventListener("click", openDataset);
    $("#status-pill").addEventListener("click", openDataset);
    $("#btn-reindex").addEventListener("click", rebuildIndex);
    $("#file-input").addEventListener("change", (e) => { uploadDataset(e.target.files[0]); e.target.value = ""; });
    const dz = $("#dropzone");
    ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", (e) => uploadDataset(e.dataTransfer.files[0]));

    // Theme + keyboard shortcuts
    setTheme(document.documentElement.dataset.theme || "light");
    $("#btn-theme").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") openDrawer(false);
      const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
      if (e.key === "/" && !typing) { e.preventDefault(); $("#query").focus(); }
    });
  }

  hydrateIcons();
  bind();
  renderHistory();
  pollStatus();
  $("#query").focus();
})();
