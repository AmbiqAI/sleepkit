import { internalPages, isPrivateModule } from "./public-docs.mjs";
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { resolve, relative, join } from "node:path";
import { parseHTML } from "linkedom";
import assert from "node:assert/strict";
const root = resolve("dist");
const files = (dir) =>
  readdirSync(dir).flatMap((n) =>
    statSync(join(dir, n)).isDirectory() ? files(join(dir, n)) : [join(dir, n)],
  );
const errors = [];
const cache = new Map();
const doc = (file) => {
  if (!cache.has(file))
    cache.set(file, parseHTML(readFileSync(file, "utf8")).document);
  return cache.get(file);
};
for (const file of files(root).filter((f) => f.endsWith(".html"))) {
  const from = new URL(
    "/sleepkit/" + relative(root, file).replace(/index\.html$/, ""),
    "https://ambiqai.github.io",
  );
  for (const a of doc(file).querySelectorAll(
    "a[href], img[src], iframe[src]",
  )) {
    const raw = a.getAttribute("href") ?? a.getAttribute("src");
    if (!raw || raw.startsWith("mailto:")) continue;
    const url = new URL(raw, from);
    if (url.origin !== from.origin || !url.pathname.startsWith("/sleepkit/"))
      continue;
    const path = resolve(
      root,
      decodeURIComponent(url.pathname.slice("/sleepkit/".length)),
    );
    const target = [path, join(path, "index.html"), path + ".html"].find(
      (p) => existsSync(p) && statSync(p).isFile(),
    );
    if (!target) errors.push(`${relative(root, file)} -> ${raw}`);
    else if (
      url.hash &&
      target.endsWith(".html") &&
      !doc(target).getElementById(decodeURIComponent(url.hash.slice(1)))
    )
      errors.push(`${relative(root, file)} -> missing anchor ${raw}`);
  }
}
const index = JSON.parse(readFileSync("src/data/api-index.json", "utf8"));
assert(index.rows.length >= 200, "API coverage unexpectedly reduced");
assert(index.rows.every((row) => !isPrivateModule(row.module)), "Private API in catalog");
const apiModel = JSON.parse(readFileSync("dist/reference/api/reference.json", "utf8"));
function checkPublic(module) {
  assert(!isPrivateModule(module.path), `Private module published: ${module.path}`);
  (module.submodules ?? []).forEach(checkPublic);
}
apiModel.modules.forEach(checkPublic);
for (const file of files(root)) {
  const route = relative(root, file).replaceAll("\\", "/");
  assert(!route.split("/").some((part) => part.startsWith("_") && !part.startsWith("_astro")), `Private route published: ${route}`);
}
assert(existsSync("dist/llms-full.txt"), "Missing LLM export");
assert(existsSync("dist/pagefind/pagefind.js"), "Missing search index");
assert(
  existsSync("dist/notebooks/train-detect-model.ipynb"),
  "Missing notebook download",
);
if (errors.length)
  throw Error(
    `Broken internal links (${errors.length}):\n${[...new Set(errors)].join("\n")}`,
  );
console.log(
  `Verified internal links across ${cache.size} pages, ${index.rows.length} API symbols and discovery exports.`,
);

for (const path of files(resolve("src/content/docs")).filter((p) => /\.mdx?$/.test(p))) {
  const rel = relative(resolve("src/content/docs"), path).replace(/(?:index)?\.mdx?$/, "");
  assert(existsSync(join(root, rel, "index.html")), `Missing authored route: ${rel}`);
}
for (const path of internalPages) {
  const rel = path.replace(/(?:index)?\.md$/, "");
  assert(!existsSync(join(root, rel, "index.html")), `Internal page published: ${rel}`);
}
const notebook = JSON.parse(
  readFileSync("../notebooks/train-detect-model.ipynb", "utf8"),
);
assert.deepEqual(
  JSON.parse(readFileSync("dist/notebooks/train-detect-model.ipynb", "utf8")),
  notebook,
  "Notebook download changed",
);
const reference = readFileSync(
  "dist/reference/api/sleepkit/defines/index.md",
  "utf8",
);
assert(
  reference.includes("TaskParams"),
  "API Markdown lost parameter documentation",
);

for (const file of files(root).filter((f) => f.endsWith("/index.html"))) {
  const content = doc(file).querySelector(".sl-markdown-content");
  if (!content) continue;
  const prose = content.cloneNode(true);
  prose
    .querySelectorAll("pre, code, script, style")
    .forEach((node) => node.remove());
  assert(
    !/(?:^|\n)\s*(?:!!!|\?\?\?|===|--8<--)/m.test(prose.textContent),
    `Unconverted Material block: ${relative(root, file)}`,
  );
  assert(
    !/\{\s*width\s*=/.test(prose.textContent),
    `Leaked image attribute: ${relative(root, file)}`,
  );
  for (const p of prose.querySelectorAll("p"))
    assert(
      !/^\s*\|.+\|/s.test(p.textContent),
      `Unrendered table: ${relative(root, file)}`,
    );
}
for (const route of [
  "tasks/detect",
  "tasks/stage",
  "zoo/detect",
  "zoo/stage",
]) {
  assert(
    doc(join(root, route, "index.html")).querySelector(
      ".sl-markdown-content table tbody tr",
    ),
    `Missing model or class table: ${route}`,
  );
}
assert(
  !doc(join(root, "quickstart/index.html")).querySelector(
    ".sl-markdown-content aside starlight-tabs",
  ),
  "Installation tabs should not be in callouts",
);
assert(
  readFileSync("dist/features/index.md", "utf8").includes('"class_names"'),
  "Folded configuration missing from Markdown export",
);
assert(
  readFileSync("dist/llms-full.txt", "utf8").includes('"class_names"'),
  "Folded configuration missing from LLM export",
);

for (const file of files(root).filter(
  (f) => f.endsWith("/index.html") && f !== join(root, "index.html"),
)) {
  const document = doc(file);
  if (!document.querySelector(".sl-markdown-content")) continue;
  assert(
    document.querySelector("[data-helia-sidebar-heading]"),
    `Page has no assigned section: ${relative(root, file)}`,
  );
}

for (const file of files(root).filter((f) => f.endsWith("/index.html"))) {
  const document = doc(file);
  const prose = document.querySelector("main")?.cloneNode(true);
  if (!prose) continue;
  prose
    .querySelectorAll("pre, code, script, style")
    .forEach((node) => node.remove());
  assert(
    !/:(?:material|simple|fontawesome|octicons)-[\w-]+:/.test(
      prose.textContent,
    ),
    `Unconverted icon shortcode: ${relative(root, file)}`,
  );
  assert(
    !/:(?:material|simple|fontawesome|octicons)-[\w-]+:/.test(document.title),
    `Icon shortcode in page title: ${relative(root, file)}`,
  );
}

for (const file of files(join(root, "examples")).filter((f) =>
  f.endsWith(".json"),
)) {
  JSON.parse(readFileSync(file, "utf8"));
}
const notebookMarkdown = readFileSync(
  "src/content/docs/guides/train-detect-model.md",
  "utf8",
);
assert(
  !notebookMarkdown.includes("\x1b"),
  "Notebook contains terminal control escapes",
);
assert(
  !notebookMarkdown.includes("file:///workspaces/"),
  "Notebook leaks terminal hyperlink targets",
);
assert(
  !notebookMarkdown.includes("## Train Sleep Detection Model"),
  "Notebook duplicates the page title",
);
assert.deepEqual(
  readFileSync("../notebooks/archive/previous-docs-train-detect-model.ipynb"),
  readFileSync("dist/notebooks/previous-docs-train-detect-model.ipynb"),
);
assert(
  !existsSync("public/examples/models-index-1.json"),
  "Model fragment should not be offered as a runnable configuration",
);
