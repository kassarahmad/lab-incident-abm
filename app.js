// Browser side: sends commands to the Python server and draws its state.
// No simulation logic here.
const $ = (id) => document.getElementById(id);
const CELL = 20;
const TILE = { "#": "#2b3440", ".": "#1b232c", "B": "#5b4a36", "C": "#7b2cbf", "E": "#2ec4b6", "S": "#3a86ff" };
const PERF = { "cfp": "#ffd60a", "propose": "#90e0ef", "accept-proposal": "#2ec4b6", "reject-proposal": "#8b98a5",
               "refuse": "#8b98a5", "inform": "#adb5bd", "inform-done": "#2ec4b6", "failure": "#ff3b5c", "request": "#ffd60a" };

let plan = [], session = null, state = null, playing = false, busy = false, timer = null;

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) } : {});
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

async function init() {
  const cfg = await api("/api/config");
  plan = cfg.floorPlan;
  const canvas = $("grid");
  canvas.width = plan[0].length * CELL;
  canvas.height = plan.length * CELL;
  for (const p of cfg.presets) $("preset").add(new Option(p.name, p.id));
  $("seed").value = cfg.defaultSeed;
  $("play").onclick = () => setPlaying(!playing);
  $("stepBtn").onclick = () => { setPlaying(false); advance(1); };
  $("reset").onclick = reset;
  $("preset").onchange = reset;
  $("seed").onchange = reset;
  $("speed").oninput = () => playing && schedule();
  await reset();
}

async function reset() {
  setPlaying(false);
  try {
    state = await api("/api/reset", { preset: $("preset").value, seed: Number($("seed").value) || 0 });
    session = state.session;
    $("error").textContent = "";
    draw();
  } catch (e) { $("error").textContent = e.message; }
}

async function advance(n) {
  if (busy || !session) return;
  busy = true;
  try {
    state = await api("/api/step", { session, n });
    draw();
    if (!state.running) setPlaying(false);
  } catch (e) { $("error").textContent = e.message; setPlaying(false); }
  busy = false;
}

function schedule() {
  clearInterval(timer);
  const speed = Number($("speed").value);
  timer = setInterval(() => advance(speed > 10 ? speed - 9 : 1), speed > 10 ? 120 : 600 / speed);
}

function setPlaying(on) {
  playing = on;
  $("play").textContent = on ? "Pause" : "Start";
  if (on) schedule(); else clearInterval(timer);
}

function draw() {
  const ctx = $("grid").getContext("2d");
  const W = plan[0].length, H = plan.length;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const t = plan[y][x];
    ctx.fillStyle = state.burned[x][y] && t !== "#" ? "#3d2b1f" : TILE[t];
    ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
    const s = state.smoke[x][y];
    if (s > 2 && t !== "#") { ctx.fillStyle = `rgba(150,160,170,${Math.min(0.75, s / 70)})`; ctx.fillRect(x * CELL, y * CELL, CELL, CELL); }
    const f = state.fire[x][y];
    if (f > 0) {
      ctx.fillStyle = `hsla(${45 - f * 0.4}, 100%, ${45 + f * 0.15}%, ${0.55 + f / 250})`;
      ctx.fillRect(x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2);
    }
  }
  for (const d of state.detectors) {
    ctx.fillStyle = d.on ? "#ff4fa3" : "#6c757d";
    ctx.fillRect(d.x * CELL + 7, d.y * CELL + 7, 6, 6);
  }
  ctx.strokeStyle = "#ffd60a"; ctx.lineWidth = 2;
  for (const c of Object.values(state.contracts)) ctx.strokeRect(c[0] * CELL + 2, c[1] * CELL + 2, CELL - 4, CELL - 4);
  for (const p of state.people) {
    const cx = p.x * CELL + CELL / 2, cy = p.y * CELL + CELL / 2;
    ctx.beginPath();
    if (p.t === "R") {
      ctx.fillStyle = "#ff3b5c";
      ctx.moveTo(cx, cy - 8); ctx.lineTo(cx + 7, cy + 6); ctx.lineTo(cx - 7, cy + 6); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "9px monospace"; ctx.textAlign = "center"; ctx.fillText(p.charges, cx, cy + 5);
    } else {
      ctx.fillStyle = p.s === "i" ? "#000" : p.s === "w" ? "#4cc9f0" : "#a2d2ff";
      ctx.arc(cx, cy, 5.5, 0, 2 * Math.PI); ctx.fill();
      if (p.s === "e") { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke(); }
    }
  }

  const sm = state.summary, h = state.history;
  $("stepLabel").textContent = `step ${state.step}`;
  $("alarmLabel").textContent = state.alarm ? "ALARM" : "no alarm";
  $("alarmLabel").className = "pill" + (state.alarm ? " alarm" : "");
  $("alarmDot").className = "dot" + (state.alarm ? " on" : "");
  $("modeLabel").textContent = state.coordination;
  $("kBurning").textContent = h.Burning.at(-1);
  $("kBurned").textContent = sm.burned_area;
  $("kEvac").textContent = sm.evacuated;
  $("kMsg").textContent = sm.messages;

  chart($("chartFire"), [[h.Burning, "#ffb703"], [h.BurnedArea, "#8d6e63"]]);
  chart($("chartEvac"), [[h.Evacuated, "#4cc9f0"], [h.Exposure.map((v) => v * 100), "#ff3b5c"]]);

  $("responders").tBodies[0].innerHTML = state.people.filter((p) => p.t === "R")
    .map((r) => `<tr><td>${r.id}</td><td>${r.charges}</td><td>${r.option}</td><td>${r.task ? r.task.join(",") : "–"}</td></tr>`).join("");
  $("log").innerHTML = state.messages.slice().reverse()
    .map((m) => `<li>t${m.step} <b style="color:${PERF[m.p] || "#fff"}">${m.p}</b> ${m.from} → ${m.to}</li>`).join("");
}

function chart(canvas, series) {
  const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#26313d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(30, 6); ctx.lineTo(30, H - 14); ctx.lineTo(W - 4, H - 14); ctx.stroke();
  const n = Math.max(30, ...series.map(([d]) => d.length));
  const max = Math.max(1, ...series.flatMap(([d]) => d));
  ctx.fillStyle = "#8b98a5"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
  ctx.fillText(Math.round(max), 26, 12); ctx.fillText("0", 26, H - 14);
  for (const [d, color] of series) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    d.forEach((v, i) => {
      const x = 30 + (i / (n - 1)) * (W - 36), y = H - 14 - (v / max) * (H - 22);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
}

init();
