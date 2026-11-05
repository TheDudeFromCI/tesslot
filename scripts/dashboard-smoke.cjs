// Optional development check: npm-provided Playwright and installed Chromium.
// Run against an already running local server: node scripts/dashboard-smoke.cjs
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.TESSLOT_CHROMIUM_PATH,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1365, height: 950 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("http://127.0.0.1:8080");
    await page
      .locator("#connection")
      .filter({ hasText: "Connected" })
      .waitFor();
    assert.equal(await page.locator("#server").textContent(), "ready");
    assert.match(await page.locator("#cpu").textContent(), /threads/);
    for (const title of ["Agents", "Training", "Evaluation", "Inference"]) {
      await page.getByRole("link", { name: new RegExp(title) }).click();
      await page.waitForFunction(
        (title) =>
          document.querySelector("#workspace-title").textContent === title,
        title,
      );
      assert.equal(await page.locator("#workspace-title").textContent(), title);
      assert.match(
        await page.locator("#capability").textContent(),
        /unimplemented/,
      );
    }
    await page.evaluate(() => {
      location.hash = "toString";
    });
    await page.waitForFunction(
      () =>
        document.querySelector("#workspace-title").textContent === "Inference",
    );
    await page.screenshot({
      path: "/tmp/tesslot-dashboard.png",
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "mobile horizontal overflow",
    );
    await page.route("**/system/status", (route) => route.abort());
    await page
      .locator("#connection")
      .filter({ hasText: "Disconnected" })
      .waitFor();
    assert.equal(await page.locator("#server").textContent(), "Unavailable");
    await page.unroute("**/system/status");
    await page
      .locator("#connection")
      .filter({ hasText: "Connected" })
      .waitFor();
    assert.deepEqual(errors, []);
    console.log(
      "Dashboard: live status, four workspaces, invalid hash, mobile width, disconnect and reconnect passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
