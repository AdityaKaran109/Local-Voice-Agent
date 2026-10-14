// Gemini Live API client for the browser: mic -> 16 kHz PCM -> WebSocket -> 24 kHz PCM playback.
// The server only mints a short-lived ephemeral token; audio flows browser <-> Google directly.

const WORKLET = `class Rec extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('rec',Rec);`;
const WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained";

const b64 = (buf) => { let s = ""; const b = new Uint8Array(buf); for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]); return btoa(s); };

export class GeminiLive {
  constructor({ onEvent }) { this.onEvent = onEvent; this.nextTime = 0; this.sources = []; }

  async start({ instructions, voice, tools }) {
    const r = await fetch("/api/gemini-live/token", { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    const { token, model } = await r.json();
    this.tools = tools;
    this.ws = new WebSocket(`${WS_URL}?access_token=${encodeURIComponent(token)}`);
    await new Promise((res, rej) => { this.ws.onopen = res; this.ws.onerror = () => rej(new Error("WebSocket failed to open")); });
    this.ws.onclose = (e) => this.onEvent({ type: "closed", reason: e.reason || `code ${e.code}` });
    this.ws.onmessage = async (e) => this._handle(JSON.parse(typeof e.data === "string" ? e.data : await e.data.text()));
    const setup = {
      model: `models/${model}`,
      generationConfig: { responseModalities: ["AUDIO"], speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } } },
      systemInstruction: { parts: [{ text: instructions }] },
      inputAudioTranscription: {},
      outputAudioTranscription: {},
    };
    if (tools?.length) setup.tools = [{ functionDeclarations: tools }];
    this.ws.send(JSON.stringify({ setup }));
    this.onEvent({ type: "log", text: `connecting to ${model}…` });
  }

  async _startMic() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    this.micCtx = new AudioContext({ sampleRate: 16000 });
    await this.micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" })));
    const node = new AudioWorkletNode(this.micCtx, "rec");
    this.micCtx.createMediaStreamSource(this.stream).connect(node);
    let pending = [], n = 0;
    node.port.onmessage = (e) => {
      pending.push(e.data); n += e.data.length;
      if (n < 1600) return; // send ~100 ms chunks
      const pcm = new Int16Array(n); let o = 0;
      for (const f of pending) for (let i = 0; i < f.length; i++) pcm[o++] = Math.max(-1, Math.min(1, f[i])) * 0x7fff;
      pending = []; n = 0;
      if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ realtimeInput: { audio: { data: b64(pcm.buffer), mimeType: "audio/pcm;rate=16000" } } }));
    };
  }

  _play(base64, mime) {
    if (!this.outCtx) this.outCtx = new AudioContext();
    const rate = +(mime.match(/rate=(\d+)/)?.[1] || 24000);
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    const pcm = new Int16Array(bytes.buffer, 0, bytes.length >> 1);
    const buf = this.outCtx.createBuffer(1, pcm.length, rate);
    const ch = buf.getChannelData(0); for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = this.outCtx.createBufferSource(); src.buffer = buf; src.connect(this.outCtx.destination);
    const at = Math.max(this.outCtx.currentTime + 0.03, this.nextTime);
    src.start(at); this.nextTime = at + buf.duration; this.sources.push(src);
    src.onended = () => (this.sources = this.sources.filter((s) => s !== src));
  }
  _stopPlayback() { this.sources.forEach((s) => { try { s.stop(); } catch {} }); this.sources = []; this.nextTime = 0; }

  async _handle(msg) {
    if (msg.setupComplete) { await this._startMic(); this.onEvent({ type: "ready" }); return; }
    if (msg.toolCall) {
      const responses = [];
      for (const fc of msg.toolCall.functionCalls || []) {
        this.onEvent({ type: "tool", name: fc.name, args: fc.args });
        const r = await fetch(`/api/tools/${fc.name}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(fc.args || {}) });
        responses.push({ id: fc.id, name: fc.name, response: { result: await r.json() } });
      }
      this.ws.send(JSON.stringify({ toolResponse: { functionResponses: responses } }));
      return;
    }
    const sc = msg.serverContent;
    if (!sc) { if (msg.goAway) this.onEvent({ type: "log", text: "server will close soon (goAway)" }); return; }
    if (sc.interrupted) { this._stopPlayback(); this.onEvent({ type: "interrupted" }); }
    if (sc.inputTranscription?.text) this.onEvent({ type: "user", text: sc.inputTranscription.text });
    if (sc.outputTranscription?.text) this.onEvent({ type: "assistant_delta", text: sc.outputTranscription.text });
    for (const p of sc.modelTurn?.parts || []) {
      if (p.inlineData?.data) { this.onEvent({ type: "audio" }); this._play(p.inlineData.data, p.inlineData.mimeType || ""); }
    }
    if (sc.turnComplete) this.onEvent({ type: "turn_complete" });
  }

  stop() {
    this._stopPlayback();
    this.ws?.close(); this.stream?.getTracks().forEach((t) => t.stop()); this.micCtx?.close();
    this.ws = this.stream = this.micCtx = null;
  }
}
