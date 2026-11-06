"use strict";
const workspaces = {
  inference: [
    "Inference",
    "Inspect model sessions, independent windows and synchronized ticks. Production model loading, sessions and token streaming are planned for milestone 5.",
  ],
  agents: [
    "Agents",
    "Persistent projects, specialist tasks and scoped tool execution are planned for milestone 6.",
  ],
  training: [
    "Training",
    "Full-parameter supervised fine-tuning, validation and resumable checkpoints are planned for milestone 3.",
  ],
  evaluation: [
    "Evaluation",
    "Replay correctness, behavioral outcomes and checkpoint comparisons are planned for milestone 7.",
  ],
};
let status = null;
const set = (id, text) => {
  document.getElementById(id).textContent = text;
};
function workspace() {
  const key = Object.hasOwn(workspaces, location.hash.slice(1))
    ? location.hash.slice(1)
    : "inference";
  const [title, description] = workspaces[key];
  document.getElementById("tiny-panel").hidden = ![
    "inference",
    "training",
  ].includes(key);
  document.getElementById("echo-panel").hidden = ![
    "inference",
    "training",
  ].includes(key);
  document.querySelectorAll("nav a").forEach((a) => {
    if (a.hash === `#${key}`) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  set("workspace-title", title);
  set("workspace-description", description);
  set(
    "capability",
    status
      ? `Backend: ${status.capabilities[key] ?? "unavailable"}`
      : "Server state unavailable",
  );
  set(
    "capability-detail",
    "Use the CLI for the implemented CPU reference and dataset validator. Execution controls become available when their backend is implemented.",
  );
}
async function poll() {
  try {
    const [s, d] = await Promise.all([
      fetch("/api/window-v1/system/status"),
      fetch("/api/window-v1/system/devices"),
    ]);
    if (!s.ok || !d.ok) throw new Error(`HTTP ${s.status}/${d.status}`);
    status = await s.json();
    const devices = await d.json();
    set("connection", "● Connected");
    set("server", status.status);
    set("version", `Tesslot ${status.application_version}`);
    set("uptime", `${Math.floor(status.uptime_ms / 1000)} s`);
    const cpu = devices.devices.find((d) => d.kind === "cpu");
    set(
      "cpu",
      cpu?.available_parallelism == null
        ? "Unavailable"
        : `${cpu.available_parallelism} threads`,
    );
    set("platform", cpu ? `${cpu.os} / ${cpu.architecture}` : "Unavailable");
    set("gpu", status.capabilities.cuda);
    set("protocol", `${status.api_version}\n${status.architecture_version}`);
    set("updated", `Last response ${new Date().toLocaleTimeString()}`);
  } catch (error) {
    status = null;
    set("connection", "● Disconnected");
    set("server", "Unavailable");
    for (const id of ["uptime", "cpu", "gpu", "version", "platform"])
      set(id, "—");
    set("protocol", "Server state unavailable");
    set("updated", `Status unavailable: ${error.message}`);
  }
  workspace();
  setTimeout(poll, 3000);
}
window.addEventListener("hashchange", workspace);
workspace();
poll();

const tinyBase = "/api/window-v1/";
async function tinyAction(path, body) {
  try {
    const r = await fetch(tinyBase + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const v = await r.json();
    if (!r.ok) throw new Error(v.message || JSON.stringify(v));
    set("tiny-output", JSON.stringify(v, null, 2));
  } catch (e) {
    set("tiny-output", e.message);
  }
}
document.getElementById("tiny-generate").onclick = () =>
  tinyAction("inference/experiments/tiny-qwen/generate", {
    prompt: document.getElementById("tiny-prompt").value,
    max_tokens: 16,
  });
document.getElementById("tiny-train").onclick = () =>
  tinyAction("training/experiments/tiny-qwen/jobs", {
    steps: Number(document.getElementById("tiny-steps").value),
  });
for (const [id, path] of [
  ["tiny-stop", "stop"],
  ["tiny-save", "checkpoint"],
  ["tiny-load", "restore"],
])
  document.getElementById(id).onclick = () =>
    tinyAction("training/experiments/tiny-qwen/" + path);
async function tinyPoll() {
  try {
    const r = await fetch(tinyBase + "inference/experiments/tiny-qwen/status");
    if (!r.ok) throw new Error("Status unavailable");
    const v = await r.json();
    set(
      "tiny-enabled",
      v.enabled
        ? v.busy
          ? "Experiment running"
          : "Experiment ready"
        : "Start the server with --tiny-experiment to enable.",
    );
    for (const id of ["tiny-generate", "tiny-train", "tiny-save"])
      document.getElementById(id).disabled = !v.enabled || v.busy;
    document.getElementById("tiny-load").disabled =
      !v.enabled || v.busy || !v.checkpoint_available;
    document.getElementById("tiny-stop").disabled = !v.enabled || !v.busy;
    if (v.enabled)
      set(
        "tiny-metrics",
        v.error || JSON.stringify(v.metrics.slice(-8), null, 2),
      );
  } catch (e) {
    set("tiny-enabled", e.message);
    for (const id of [
      "tiny-generate",
      "tiny-train",
      "tiny-save",
      "tiny-load",
      "tiny-stop",
    ])
      document.getElementById(id).disabled = true;
  }
  setTimeout(tinyPoll, 1000);
}
tinyPoll();

function showEcho(report) {
  if (!report?.replay) return;
  const r = report.replay;
  set("echo-source", r.source);
  set("echo-result", r.echo);
  set(
    "echo-accuracy",
    `Free-running accuracy: ${(100 * r.accuracy).toFixed(1)}% · step ${report.step} · teacher-forced loss ${report.loss.toFixed(4)} · source-masked loss ${report.source_masked_loss.toFixed(4)} · seed ${report.seed}`,
  );
  const tbody = document.getElementById("echo-ticks");
  tbody.replaceChildren();
  const token = (id) =>
    id == null
      ? "—"
      : id >= 32 && id < 127
        ? String.fromCharCode(id)
        : `\\x${id.toString(16).padStart(2, "0")}`;
  for (const tick of r.ticks) {
    const row = document.createElement("tr");
    for (const value of [
      tick.tick,
      token(tick.source_token),
      token(tick.echo_token),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    tbody.append(row);
  }
}
async function echoAction(path, body) {
  try {
    const r = await fetch(tinyBase + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(result.message || JSON.stringify(result));
    set(
      "echo-output",
      result.replay
        ? `Replayed source and echo at step ${result.step}.`
        : JSON.stringify(result, null, 2),
    );
    showEcho(result);
  } catch (e) {
    set("echo-output", e.message);
  }
}
const echoSeed = () => Number(document.getElementById("echo-seed").value);
document.getElementById("echo-generate").onclick = () =>
  echoAction("inference/experiments/tiny-qwen-echo/generate", {
    seed: echoSeed(),
  });
document.getElementById("echo-train").onclick = () =>
  echoAction("training/experiments/tiny-qwen-echo/jobs", {
    steps: Number(document.getElementById("echo-steps").value),
    seed: echoSeed(),
  });
for (const [id, path] of [
  ["echo-stop", "stop"],
  ["echo-save", "checkpoint"],
  ["echo-load", "restore"],
])
  document.getElementById(id).onclick = () =>
    echoAction("training/experiments/tiny-qwen-echo/" + path);
async function echoPoll() {
  try {
    const r = await fetch(
      tinyBase + "inference/experiments/tiny-qwen-echo/status",
    );
    if (!r.ok) throw new Error("Echo status unavailable");
    const v = await r.json();
    set(
      "echo-enabled",
      !v.enabled
        ? "Start the server with --tiny-experiment to enable."
        : v.busy
          ? "Echo experiment running"
          : "Echo experiment ready",
    );
    for (const id of ["echo-generate", "echo-train", "echo-save"])
      document.getElementById(id).disabled = !v.enabled || v.busy;
    document.getElementById("echo-load").disabled =
      !v.enabled || v.busy || !v.checkpoint_available;
    document.getElementById("echo-stop").disabled = !v.enabled || !v.busy;
    if (v.enabled) {
      set(
        "echo-metrics",
        v.error || JSON.stringify(v.metrics.slice(-8), null, 2),
      );
      showEcho(v.echo_result);
    }
  } catch (e) {
    set("echo-enabled", e.message);
    for (const id of [
      "echo-generate",
      "echo-train",
      "echo-save",
      "echo-load",
      "echo-stop",
    ])
      document.getElementById(id).disabled = true;
  }
  setTimeout(echoPoll, 1000);
}
echoPoll();
