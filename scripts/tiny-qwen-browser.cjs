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
    await page.waitForFunction(
      () => !document.querySelector("#echo-generate").disabled,
    );
    await page.click("#echo-generate");
    await page.waitForFunction(
      () =>
        document.querySelector("#echo-source").textContent === "jbkquuxmrgvc",
    );
    await page.fill("#echo-steps", "400");
    await page.click("#echo-train");
    await page.waitForFunction(
      () =>
        document.querySelector("#echo-accuracy").textContent.includes("100.0%"),
      null,
      { timeout: 90000 },
    );
    await page.waitForFunction(
      () => !document.querySelector("#echo-save").disabled,
    );
    await page.click("#echo-save");
    await page.waitForFunction(() =>
      document
        .querySelector("#echo-output")
        .textContent.includes('"restored": false'),
    );
    await page.waitForFunction(
      () => !document.querySelector("#echo-load").disabled,
    );
    await page.click("#echo-load");
    await page.waitForFunction(() =>
      document
        .querySelector("#echo-output")
        .textContent.includes('"restored": true'),
    );
    await page.waitForFunction(
      () => !document.querySelector("#echo-generate").disabled,
    );
    await page.click("#echo-generate");
    await page.waitForFunction(() =>
      document
        .querySelector("#echo-output")
        .textContent.includes("Replayed source and echo"),
    );
    assert.equal(
      await page.locator("#echo-result").textContent(),
      "jbkquuxmrgvc",
    );
    assert.equal(await page.locator("#echo-ticks tr").count(), 13);
    await page.waitForFunction(
      () => !document.querySelector("#corpus-train").disabled,
    );
    await page.fill("#corpus-steps", "600");
    await page.click("#corpus-train");
    await page.waitForFunction(
      () =>
        document
          .querySelector("#corpus-evaluation")
          .textContent.includes("step 600"),
      null,
      { timeout: 120000 },
    );
    await page.waitForFunction(
      () => !document.querySelector("#corpus-test").disabled,
    );
    await page.click("#corpus-test");
    await page.waitForFunction(() =>
      document
        .querySelector("#corpus-output")
        .textContent.includes("Completed test evaluation"),
    );
    assert.equal(
      await page.locator("#corpus-accuracy").textContent(),
      "100.0%",
    );
    assert.equal(await page.locator("#corpus-cases tr").count(), 16);
    assert.equal(
      await page.locator("#corpus-cases tr td").first().textContent(),
      "9223372036854775808",
    );
    await page.waitForFunction(
      () => !document.querySelector("#corpus-save").disabled,
    );
    await page.click("#corpus-save");
    await page.waitForFunction(() =>
      document
        .querySelector("#corpus-output")
        .textContent.includes('"restored":false'),
    );
    await page.waitForFunction(
      () => !document.querySelector("#corpus-load").disabled,
    );
    await page.click("#corpus-load");
    await page.waitForFunction(() =>
      document
        .querySelector("#corpus-output")
        .textContent.includes('"restored":true'),
    );
    await page.waitForFunction(
      () => !document.querySelector("#corpus-test").disabled,
    );
    await page.click("#corpus-test");
    await page.waitForFunction(() =>
      document
        .querySelector("#corpus-output")
        .textContent.includes("Completed test evaluation"),
    );
    assert.equal(
      await page.locator("#corpus-accuracy").textContent(),
      "100.0%",
    );
    await page.screenshot({
      path: "/tmp/tesslot-tiny-qwen.png",
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    console.log(
      "Browser generation, ABC/single-pattern/corpus training, held-out test evaluation and all checkpoint workflows passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
