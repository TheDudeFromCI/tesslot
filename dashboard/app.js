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
