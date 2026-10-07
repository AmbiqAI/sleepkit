import { test, expect } from "@playwright/test";
test("installation tabs switch and notebook outputs remain available", async ({
  page,
}) => {
  await page.goto("quickstart/");
  await page.getByRole("tab", { name: "PyPI install" }).click();
  await expect(
    page
      .getByRole("tabpanel")
      .filter({ hasText: "pip install sleepkit" })
      .first(),
  ).toBeVisible();
  await page.goto("guides/train-detect-model/");
  await expect(
    page.getByRole("link", { name: "Download notebook" }),
  ).toHaveAttribute("href", "/sleepkit/notebooks/train-detect-model.ipynb");
  const figures = page.locator('img[src*="/notebooks/"]');
  await expect(figures).toHaveCount(3);
  for (const img of await figures.all())
    expect(
      await img.evaluate(
        (e: HTMLImageElement) => e.complete && e.naturalWidth > 0,
      ),
    ).toBeTruthy();
});
test("API search narrows results", async ({ page }) => {
  await page.goto("reference/");
  const search = page.getByRole("searchbox", { name: "Search Python API" });
  await expect(search).toBeVisible();
  await search.fill("CmidssDataset");
  await expect(
    page.getByRole("link", { name: "CmidssDataset", exact: true }).first(),
  ).toBeVisible();
});
test("mobile document stays within viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await expect(
    page.getByRole("heading", {
      name: "From sleep signals to Edge AI.",
      exact: true,
    }),
  ).toBeVisible();
});

test("global search finds authored content", async ({ page }) => {
  await page.goto("");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page.locator(".pagefind-ui__search-input").fill("staging");
  await expect(page.locator(".pagefind-ui__result-link").first()).toBeVisible();
});
test("diagrams and chart embeds render", async ({ page }) => {
  await page.goto("features/");
  await expect(page.locator(".sl-markdown-content svg").first()).toBeVisible();
  await page.goto("guides/stage-ablation/");
  await expect(page.locator("iframe")).toHaveCount(5);
  await expect(
    page
      .locator("iframe")
      .first()
      .contentFrame()
      .locator(".plotly .main-svg")
      .first(),
  ).toBeVisible();
});

test("mobile navigation has a single Home link and task-specific staging guides", async ({
  page,
}) => {
  await page.setViewportSize({ width: 835, height: 890 });
  await page.goto("");
  await page.locator('[data-helia-section-dropdown] summary').click();
  await page.getByRole('navigation', { name: 'Choose section' }).getByRole('link', { name: 'Tasks', exact: true }).click();
  await page.getByRole('button', { name: 'Open navigation', exact: true }).click();
  await expect(page.locator('[data-helia-sidebar-heading]')).toHaveText('Tasks');
  await page
    .getByRole("link", { name: "Saved-feature baseline", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Saved-feature sleep staging baseline",
      exact: true,
    }),
  ).toBeVisible();
});
test("large configuration has a preview, full copyable code and exact download", async ({
  page,
  request,
}) => {
  await page.goto("features/");
  const example = page.locator(".config-example").first();
  await expect(example.locator(".config-preview")).toBeVisible();
  await expect(example.locator("details")).not.toHaveAttribute("open", "");
  const link = example.getByRole("link", { name: "Download", exact: true });
  const response = await request.get((await link.getAttribute("href"))!);
  expect(response.ok()).toBeTruthy();
  const downloaded = await response.text();
  expect(JSON.parse(downloaded).name).toBe("sd-2-tcn-sm");
  await example.locator("summary").click();
  await expect(example.locator(".config-preview")).toBeHidden();
  await expect(example.locator(".expressive-code")).toBeVisible();
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);
  await example.getByTitle("Copy to clipboard", { exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()))
    .toBe(downloaded.trim());
  await example.locator("summary").click();
  await expect(example.locator(".config-preview")).toBeVisible();
});
test("class tables and image sizes survive conversion", async ({ page }) => {
  await page.goto("tasks/detect/");
  const content = page.locator(".sl-markdown-content");
  await expect(
    content.getByRole("cell", { name: "AWAKE", exact: true }),
  ).toBeVisible();
  await expect(
    content.getByRole("cell", { name: "SLEEP", exact: true }),
  ).toBeVisible();
  await expect(
    content.locator('img[alt="Wrist-based Sleep Classification"]'),
  ).toHaveAttribute("width", "540");
  expect(await content.innerText()).not.toContain("{ width");
});

test("navigation scopes historical routes and installation contains only commands", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  for (const [route, label] of [
    ["datasets/cmidss/", "User guide"],
    ["staging-baseline/", "Tasks"],
    ["huggingface-artifacts/", "Reference"],
    ["usage/python/", "Getting started"],
  ]) {
    await page.goto(route);
    await expect(page.locator("[data-helia-sidebar-heading]")).toHaveText(
      label,
    );
    await expect(
      page.locator('nav[aria-label="Primary"] a[aria-current="page"]'),
    ).toHaveText(label);
  }
  await page.goto("");
  await expect(page.locator('nav[aria-label="Primary"] a')).toHaveCount(5);
  await expect(page.locator("helia-ascii-terminal")).toHaveCount(0);
  await page.goto("quickstart/");
  await page.getByRole("tab", { name: "Git clone", exact: true }).click();
  const content = page.getByRole("tabpanel").filter({ hasText: "git clone" });
  await expect(content).toBeVisible();
  await expect(content).not.toContainText("--->");
});

test("review fixes preserve notebook downloads and readable mobile diagrams", async ({
  page,
  request,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("guides/train-detect-model/");
  await expect(
    page.getByRole("heading", {
      name: "Train Sleep Detection Model",
      exact: true,
    }),
  ).toHaveCount(1);
  expect(await page.locator("main").textContent()).not.toContain("\x1b");
  const alternate = page.getByRole("link", {
    name: "previous documentation copy",
    exact: true,
  });
  expect(
    (await request.get((await alternate.getAttribute("href"))!)).ok(),
  ).toBeTruthy();
  await page.goto("features/");
  const diagram = page.locator(".sl-markdown-content svg").first();
  const bounds = await diagram.boundingBox();
  expect(bounds!.height).toBeGreaterThan(400);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.goto("models/");
  await expect(page.locator('a[download$=".json"]')).toHaveCount(0);
});
