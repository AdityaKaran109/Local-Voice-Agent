// Shared voice client: mic capture + VAD, push-to-talk, barge-in, audio playback,
// browser Web Speech fallbacks, provider settings, transcript rendering.

const CFG_KEY = "voicelab.cfg";
const DEFAULTS = { stt: "whisper", llm: "ollama", tts: "kokoro", voice: "", input: "handsfree", bargeIn: true };

export function loadCfg() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(CFG_KEY) || "{}") }; } catch { return { ...DEFAULTS }; }
}
function saveCfg(cfg) { try { localStorage.setItem(CFG_KEY, JSON.stringify(cfg)); } catch {} }

const NAV = [
  ["index.html", "Home"],
  ["playground.html", "1 · Voice stack"],
  ["realtime.html", "2 · Voice-to-voice"],
  ["game.html", "3 · Voice in app"],
  ["agent.html", "4 · Voice for agent"],
  ["caller.html", "5 · Voice as tool"],
  ["evals.html", "6 · Evals"],
  ["vocalbridge.html", "Vocal Bridge"],
];

export function renderNav() {
  const here = location.pathname.split("/").pop() || "index.html";
  const el = document.getElementById("nav");
  el.innerHTML = `<a class="brand" href="index.html">🎙️ Voice Agent Lab</a><nav>${NAV.map(
    ([href, label]) => `<a href="${href}" class="${href === here ? "active" : ""}">${label}</a>`).join("")}</nav>`;
}

// ---------- WAV encoding ----------
function encodeWav(float32, sampleRate) {
  const buf = new ArrayBuffer(44 + float32.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + float32.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, float32.length * 2, true);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

const WORKLET = `class Rec extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('rec',Rec);`;

// ---------- Voice client ----------
export class VoiceClient {
  constructor(mode, { onMessage, onState } = {}) {
    this.mode = mode;
    this.onMessage = onMessage || (() => {});
    this.onState = onState || (() => {});
    this.cfg = loadCfg();
    this.extra = {};
    this.playCtx = null;
    this.nextTime = 0;
    this.sources = [];
    this.playChain = Promise.resolve();
    this.micOn = false;
  }

  connect() {
    return new Promise((resolve) => {
      this.ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/${this.mode}`);
      this.ws.onopen = () => { this.sendConfig(); this.onState("ready"); resolve(); };
      this.ws.onclose = () => { this.onState("disconnected"); setTimeout(() => this.connect(), 1500); };
      this.ws.onmessage = (e) => this._handle(JSON.parse(e.data));
    });
  }

  send(obj) { if (this.ws?.readyState === 1) this.ws.send(typeof obj === "string" || obj instanceof ArrayBuffer ? obj : JSON.stringify(obj)); }
  sendConfig(patch = {}) {
    Object.assign(this.cfg, patch);
    saveCfg(this.cfg);
    this.send({ type: "config", stt: this.cfg.stt, llm: this.cfg.llm, tts: this.cfg.tts, voice: this.cfg.voice || null, ...this.extra });
  }
  setExtra(extra) { Object.assign(this.extra, extra); this.sendConfig(); }
  sendText(text) { this.stopPlayback(); this.send({ type: "text", text }); }
  sendAction(data) { this.stopPlayback(); this.send({ type: "action", ...data }); }
  reset() { this.stopPlayback(); this.send({ type: "reset" }); }

  _handle(msg) {
    if (msg.type === "audio") this._playAudio(msg.data);
    else if (msg.type === "speak") this._speakBrowser(msg.text, msg.role);
    else if (msg.type === "interrupted") this.stopPlayback();
    else if (msg.type === "status") this.onState(msg.text);
    if (msg.type === "assistant" && msg.role === "assistant") setTimeout(() => !this.isPlaying() && this.onState(this.micOn ? "listening" : "ready"), 50);
    this.onMessage(msg);
  }

  // ---------- playback ----------
  _ctx() { if (!this.playCtx) this.playCtx = new AudioContext(); if (this.playCtx.state === "suspended") this.playCtx.resume(); return this.playCtx; }
  _playAudio(b64) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    // chain decodes so sentences always play in order
    this.playChain = this.playChain.then(async () => {
      const ctx = this._ctx();
      const buf = await ctx.decodeAudioData(bytes.buffer);
      const src = ctx.createBufferSource();
      src.buffer = buf; src.connect(ctx.destination);
      const at = Math.max(ctx.currentTime + 0.02, this.nextTime);
      src.start(at); this.nextTime = at + buf.duration;
      this.sources.push(src);
      src.onended = () => { this.sources = this.sources.filter((s) => s !== src); if (!this.isPlaying()) this.onState(this.micOn ? "listening" : "ready"); };
      this.onState("speaking");
    }).catch((e) => console.error("playback", e));
  }
  _speakBrowser(text, role) {
    const u = new SpeechSynthesisUtterance(text);
    const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith("en"));
    const idx = { assistant: 0, callee: 1, caller: 2 }[role] ?? 0;
    if (voices.length) u.voice = voices[(idx * 3) % voices.length];
    u.onstart = () => this.onState("speaking");
    u.onend = () => !speechSynthesis.speaking && this.onState(this.micOn ? "listening" : "ready");
    speechSynthesis.speak(u);
  }
  isPlaying() { return (this.playCtx && this.playCtx.currentTime < this.nextTime - 0.01) || speechSynthesis.speaking; }
  stopPlayback() {
    this.sources.forEach((s) => { try { s.stop(); } catch {} });
    this.sources = []; this.nextTime = 0; this.playChain = Promise.resolve();
    speechSynthesis.cancel();
  }
  bargeIn() {
    if (this.cfg.bargeIn && this.isPlaying()) { this.stopPlayback(); this.send({ type: "interrupt" }); }
  }

  // ---------- microphone ----------
  async startMic() {
    if (this.micOn) return;
    this.micOn = true;
    this._ctx(); // unlock audio output on the same user gesture
    if (this.cfg.stt === "browser") return this._startBrowserSTT();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    this.micCtx = new AudioContext({ sampleRate: 16000 });
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await this.micCtx.audioWorklet.addModule(url);
    const srcNode = this.micCtx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.micCtx, "rec");
    srcNode.connect(this.node);
    this.sr = this.micCtx.sampleRate;
    this._resetVad();
    this.node.port.onmessage = (e) => this._onFrame(e.data);
    this.onState("listening");
  }
  stopMic() {
    this.micOn = false;
    this.recognition?.stop(); this.recognition = null;
    this.node?.disconnect(); this.stream?.getTracks().forEach((t) => t.stop()); this.micCtx?.close();
    this.node = this.stream = this.micCtx = null;
    this.onState("ready");
  }
  _resetVad() { this.vad = { speaking: false, frames: [], pre: [], loud: 0, quietMs: 0, noise: 0.005, ptt: false }; }

  // push-to-talk
  pttDown() { if (!this.micOn) return; this.bargeIn(); this.vad.ptt = true; this.vad.frames = []; this.onState("user-speaking"); }
  pttUp() { if (!this.vad?.ptt) return; this.vad.ptt = false; this._flush(); }

  _onFrame(f) {
    const v = this.vad;
    const frameMs = (f.length / this.sr) * 1000;
    if (this.cfg.input === "ptt") { if (v.ptt) v.frames.push(f); return; }
    let sum = 0; for (let i = 0; i < f.length; i++) sum += f[i] * f[i];
    const rms = Math.sqrt(sum / f.length);
    this.onLevel?.(rms);
    const threshold = Math.max(0.015, v.noise * 3) * (this.isPlaying() ? 2.5 : 1); // stricter while the bot talks (echo)
    if (!v.speaking) {
      v.pre.push(f); if (v.pre.length > 40) v.pre.shift(); // ~320ms pre-roll
      if (rms > threshold) { v.loud += frameMs; } else { v.loud = 0; v.noise = 0.97 * v.noise + 0.03 * rms; }
      if (v.loud > 120) { v.speaking = true; v.frames = [...v.pre]; v.quietMs = 0; this.bargeIn(); this.onState("user-speaking"); }
    } else {
      v.frames.push(f);
      v.quietMs = rms > threshold * 0.7 ? 0 : v.quietMs + frameMs;
      if (v.quietMs > (this.silenceMs || 700)) { v.speaking = false; v.loud = 0; this._flush(); }
    }
  }
  _flush() {
    const frames = this.vad.frames; this.vad.frames = []; this.vad.pre = [];
    const len = frames.reduce((n, f) => n + f.length, 0);
    if (len / this.sr < 0.35) { this.onState("listening"); return; }
    const all = new Float32Array(len); let o = 0; frames.forEach((f) => { all.set(f, o); o += f.length; });
    this.stopPlayback();
    this.send(encodeWav(all, this.sr));
    this.onState("transcribing");
  }

  _startBrowserSTT() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { alert("This browser has no Web Speech API. Use Chrome/Safari or pick another STT."); this.micOn = false; return; }
    const r = new SR(); r.continuous = true; r.interimResults = true; r.lang = "en-US";
    r.onspeechstart = () => { this.bargeIn(); this.onState("user-speaking"); };
    r.onresult = (e) => {
      const res = e.results[e.results.length - 1];
      if (res.isFinal && res[0].transcript.trim()) this.sendText(res[0].transcript.trim());
    };
    r.onend = () => { if (this.micOn && this.recognition === r) r.start(); };
    r.start(); this.recognition = r; this.onState("listening");
  }
}

// ---------- settings bar ----------
// Primary choices are open source and Google Gemini; other cloud APIs sit behind "More providers".
export const PRESETS = {
  oss: { label: "Open source (local)", stt: "whisper", llm: "ollama", tts: "kokoro" },
  gemini: { label: "Google Gemini", stt: "gemini", llm: "gemini", tts: "gemini" },
};
const isPrimary = (p) => p.kind !== "cloud" || p.id.startsWith("gemini");

export async function renderSettings(el, client, { showInput = true } = {}) {
  const cat = await (await fetch("/api/providers")).json();
  window.__providers = cat;
  const cfg = client.cfg;
  const usable = (k, id) => cat[k].some((p) => p.id === id && p.available);
  const presetOk = (name) => ["stt", "llm", "tts"].every((k) => usable(k, PRESETS[name][k]));
  // fall back to an available provider if the saved one isn't usable
  for (const k of ["stt", "llm", "tts"]) if (!usable(k, cfg[k])) cfg[k] = (cat[k].find((p) => p.available) || cat[k][0]).id;
  const currentPreset = () => Object.keys(PRESETS).find((n) => ["stt", "llm", "tts"].every((k) => cfg[k] === PRESETS[n][k])) || "custom";

  const opt = (p, sel) => `<option value="${p.id}" ${p.id === sel ? "selected" : ""} ${p.available ? "" : "disabled"} title="${p.reason || ""}">
      ${p.label}${p.kind === "open-source" ? " · OSS" : p.kind === "browser" ? " · browser" : ""}${p.available ? "" : " — " + p.reason}</option>`;
  const list = (k) => cat[k].filter((p) => cfg.moreProviders || isPrimary(p) || p.id === cfg[k]);

  el.innerHTML = `
    <label class="stack">Stack <select data-k="preset">
      ${Object.entries(PRESETS).map(([n, p]) => `<option value="${n}" ${presetOk(n) ? "" : "disabled"}>${p.label}${presetOk(n) ? "" : n === "gemini" ? " — add GEMINI_API_KEY" : " — unavailable"}</option>`).join("")}
      <option value="custom">Custom mix</option></select></label>
    <label>STT <select data-k="stt"></select></label>
    <label>LLM <select data-k="llm"></select></label>
    <label>TTS <select data-k="tts"></select></label>
    <label>Voice <select data-k="voice"></select></label>
    ${showInput ? `<label>Input <select data-k="input"><option value="handsfree">Hands-free (VAD)</option><option value="ptt">Push-to-talk (hold Space)</option></select></label>
    <label class="check"><input type="checkbox" data-k="bargeIn"> Barge-in</label>` : ""}
    <label class="check muted"><input type="checkbox" data-k="moreProviders"> More providers</label>`;

  const sel = (k) => el.querySelector(`[data-k="${k}"]`);
  const fill = () => {
    for (const k of ["stt", "llm", "tts"]) sel(k).innerHTML = list(k).map((p) => opt(p, cfg[k])).join("");
    const voices = (cat.tts.find((p) => p.id === cfg.tts) || {}).voices || [];
    if (!voices.includes(cfg.voice)) cfg.voice = voices[0] || "";
    sel("voice").innerHTML = voices.length ? voices.map((v) => `<option ${v === cfg.voice ? "selected" : ""}>${v}</option>`).join("") : `<option value="">default</option>`;
    sel("voice").disabled = !voices.length;
    sel("preset").value = currentPreset();
  };
  fill();
  if (showInput) { sel("input").value = cfg.input; sel("bargeIn").checked = cfg.bargeIn; }
  sel("moreProviders").checked = !!cfg.moreProviders;

  el.querySelectorAll("[data-k]").forEach((s) => s.addEventListener("change", () => {
    const k = s.dataset.k;
    const wasOn = client.micOn;
    const before = cfg.stt;
    if (k === "preset") {
      if (s.value !== "custom") Object.assign(cfg, { stt: PRESETS[s.value].stt, llm: PRESETS[s.value].llm, tts: PRESETS[s.value].tts });
    } else cfg[k] = s.type === "checkbox" ? s.checked : s.value;
    if (k !== "voice") fill();
    if (wasOn && (k === "input" || cfg.stt !== before)) { client.stopMic(); client.startMic(); }
    client.sendConfig();
    el.dispatchEvent(new CustomEvent("change-config", { detail: cfg }));
  }));
  client.sendConfig();
  return cat;
}

// ---------- transcript ----------
const ROLE_LABEL = { user: "You", assistant: "Agent", caller: "📞 Our agent", callee: "📞 Business" };
const plain = (s) => String(s).replace(/\*\*|__/g, ""); // LLMs add markdown bold; show it as plain text

export class Transcript {
  constructor(el) { this.el = el; this.open = {}; }
  clear() { this.el.innerHTML = ""; this.open = {}; }
  _bubble(role, text = "") {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.innerHTML = `<span class="who">${ROLE_LABEL[role] || role}</span><span class="text"></span>`;
    div.querySelector(".text").textContent = plain(text);
    this.el.appendChild(div); this.el.scrollTop = this.el.scrollHeight;
    return div;
  }
  note(html, cls = "") {
    const div = document.createElement("div"); div.className = `note ${cls}`; div.innerHTML = html;
    this.el.appendChild(div); this.el.scrollTop = this.el.scrollHeight; return div;
  }
  handle(msg) {
    if (msg.type === "transcript") { this._bubble(msg.role, msg.text); }
    else if (msg.type === "delta") {
      if (!this.open[msg.role]) this.open[msg.role] = this._bubble(msg.role);
      const t = this.open[msg.role].querySelector(".text");
      t.textContent = plain(t.textContent + msg.text);
      this.el.scrollTop = this.el.scrollHeight;
    } else if (msg.type === "assistant") {
      const b = this.open[msg.role] || this._bubble(msg.role);
      if (msg.text) b.querySelector(".text").textContent = plain(msg.text);
      delete this.open[msg.role];
    } else if (msg.type === "interrupted") {
      Object.values(this.open).forEach((b) => b.classList.add("cut"));
      this.open = {};
    } else if (msg.type === "error") { this.note(`⚠️ ${msg.message}`, "error"); }
    else if (msg.type === "event" && msg.name === "tool_call") {
      this.note(`🔧 <code>${escapeHtml(`${msg.tool}(${JSON.stringify(msg.arguments || {})})`)}</code>`, "tool");
    }
  }
}
export function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }

// ---------- conversation panel (mic button, text box, status) ----------
export async function setupConversation({ mode, panel, settingsEl, onEvent, onMessage, extra }) {
  panel.innerHTML = `
    <div class="convo-head"><span class="status" id="status"><i></i><b>connecting…</b></span>
      <span class="level"><span id="level"></span></span>
      <button class="ghost" id="reset">Reset</button></div>
    <div class="transcript" id="transcript"></div>
    <div class="controls">
      <button class="mic" id="mic" title="Start/stop microphone">🎙️ <span>Start talking</span></button>
      <form id="textform"><input id="textin" placeholder="…or type a message" autocomplete="off"><button>Send</button></form>
    </div>`;
  const statusEl = panel.querySelector("#status");
  const transcript = new Transcript(panel.querySelector("#transcript"));
  const client = new VoiceClient(mode, {
    onState: (s) => { statusEl.dataset.state = s; statusEl.querySelector("b").textContent = s; },
    onMessage: (msg) => {
      transcript.handle(msg);
      if (msg.type === "event") onEvent?.(msg, { client, transcript });
      onMessage?.(msg, { client, transcript });
    },
  });
  if (extra) client.extra = extra;
  const level = panel.querySelector("#level");
  client.onLevel = (rms) => { level.style.width = Math.min(100, rms * 600) + "%"; };
  await client.connect();
  if (settingsEl) await renderSettings(settingsEl, client);

  const mic = panel.querySelector("#mic");
  mic.onclick = async () => {
    if (client.micOn) { client.stopMic(); mic.classList.remove("on"); mic.querySelector("span").textContent = "Start talking"; }
    else {
      try { await client.startMic(); mic.classList.add("on"); mic.querySelector("span").textContent = client.cfg.input === "ptt" ? "Mic on · hold Space" : "Listening… (click to stop)"; }
      catch (e) { transcript.note(`⚠️ Microphone error: ${escapeHtml(e.message)}`, "error"); client.micOn = false; }
    }
  };
  addEventListener("keydown", (e) => { if (e.code === "Space" && client.cfg.input === "ptt" && !e.repeat && document.activeElement.tagName !== "INPUT") { e.preventDefault(); client.pttDown(); } });
  addEventListener("keyup", (e) => { if (e.code === "Space" && client.cfg.input === "ptt") { e.preventDefault(); client.pttUp(); } });
  panel.querySelector("#textform").onsubmit = (e) => {
    e.preventDefault(); client._ctx();
    const input = panel.querySelector("#textin");
    if (input.value.trim()) { client.sendText(input.value.trim()); input.value = ""; }
  };
  panel.querySelector("#reset").onclick = () => { client.reset(); transcript.clear(); };
  return { client, transcript };
}
