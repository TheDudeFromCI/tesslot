// Opt-in browser check against serve --tiny-experiment on port 8081.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.TESSLOT_CHROMIUM_PATH,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("http://127.0.0.1:8081");
    await page.waitForFunction(
      () => !document.querySelector("#tiny-generate").disabled,
    );
    await page.click("#tiny-generate");
    await page.waitForFunction(() =>
      document.querySelector("#tiny-output").textContent.includes("token_ids"),
    );
    await page.fill("#tiny-steps", "4");
    await page.click("#tiny-train");
    await page.waitForFunction(() =>
      document
        .querySelector("#tiny-metrics")
        .textContent.includes("gradient_norm"),
    );
    await page.waitForFunction(
      () => !document.querySelector("#tiny-save").disabled,
    );
    await page.click("#tiny-save");
    await page.waitForFunction(() =>
      document
        .querySelector("#tiny-output")
        .textContent.includes('"restored": false'),
    );
    await page.waitForFunction(
      () => !document.querySelector("#tiny-load").disabled,
    );
    await page.click("#tiny-load");
    await page.waitForFunction(() =>
      document
        .querySelector("#tiny-output")
        .textContent.includes('"restored": true'),
    );
    await page.screenshot({
      path: "/tmp/tesslot-tiny-qwen.png",
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    console.log(
      "Actual browser generation, training metrics and checkpoint save/restore passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
