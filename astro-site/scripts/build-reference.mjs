import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
const repo = resolve("..");
const run = (cmd, args) =>
  execFileSync(cmd, args, {
    cwd: repo,
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
const commit = run("git", ["rev-parse", "HEAD"]).trim();
mkdirSync(".cache", { recursive: true });
const dump = JSON.parse(
  run("uv", [
    "tool",
    "run",
    "--from",
    "griffe==1.7.3",
    "griffe",
    "dump",
    "sleepkit",
    "--docstyle",
    "google",
    "-f",
  ]),
);
function clean(node) {
  if (node.docstring) {
    node.docstring.value = node.docstring.value.replace(
      /:(?:material|simple|fontawesome|octicons)-[\w-]+:/g,
      "",
    );
    for (const s of node.docstring.parsed ?? [])
      if (s.kind === "text")
        s.value = s.value.replace(/:(?:material|simple|fontawesome|octicons)-[\w-]+:/g, "");
  }
  for (const child of Object.values(node.members ?? {})) clean(child);
}
clean(dump.sleepkit);
writeFileSync(".cache/griffe.json", JSON.stringify(dump));
run(process.execPath, [
  resolve("node_modules/@ambiqai/helia-ui/scripts/pyref.mjs"),
  "--input",
  resolve(".cache/griffe.json"),
  "--out",
  resolve("src/content/docs/reference/api"),
  "--public",
  resolve("public"),
  "--base",
  "/sleepkit/",
  "--site",
  "https://ambiqai.github.io",
  "--source-root",
  repo,
  "--source-url",
  `https://github.com/AmbiqAI/sleepkit/blob/${commit}/{path}#L{line}`,
  "--commit",
  commit,
  "--quiet",
]);
const model = JSON.parse(
  readFileSync("public/reference/api/reference.json", "utf8"),
);
const modules = [];
function visit(m) {
  modules.push(m);
  (m.submodules ?? []).forEach(visit);
}
model.modules.forEach(visit);
const route = (m) =>
  "reference/api/" + m.path.toLowerCase().replaceAll(".", "/");
const rows = modules.flatMap((m) =>
  (m.symbols ?? [])
    .filter((s) => ["class", "function"].includes(s.kind))
    .map((s) => ({
      id: s.id,
      name: s.name,
      kind: s.kind,
      module: m.path,
      href: `/sleepkit/${route(m)}/#${s.id}`,
      summary: (s.description ?? "").split("\n")[0],
      facets: { kind: [s.kind], category: [m.path.split(".")[1] ?? "package"] },
    })),
);
writeFileSync(
  "src/data/api-sidebar.json",
  JSON.stringify(modules.map((m) => ({ label: m.path, slug: route(m) }))),
);
writeFileSync(
  "src/data/api-index.json",
  JSON.stringify({
    rows,
    filters: [
      { id: "kind", label: "Symbol type", values: ["class", "function"] },
      {
        id: "category",
        label: "Category",
        values: [...new Set(rows.flatMap((r) => r.facets.category))].sort(),
      },
    ],
  }),
);
writeFileSync(
  "src/data/redirects.json",
  JSON.stringify(
    Object.fromEntries(
      modules.map((m) => [
        "/api/" + m.path.replaceAll(".", "/"),
        "/sleepkit/" + route(m) + "/",
      ]),
    ),
  ),
);
writeFileSync(
  "src/content/docs/reference/index.mdx",
  `---\ntitle: Python API catalog\ndescription: Search sleepKIT classes, functions, signatures and parameter documentation.\n---\nimport ReferenceBrowser from '@ambiqai/helia-ui/astro/ReferenceBrowser';\nimport catalog from '../../../data/api-index.json';\n\nSearch classes and functions by name or module. Documentation is generated from the Python source without importing training dependencies.\n\n<ReferenceBrowser rows={catalog.rows} filters={catalog.filters} label="Search Python API" itemsLabel="APIs" pageSize={20} />\n`,
);
console.log(
  `Generated ${modules.length} API modules and ${rows.length} searchable symbols.`,
);
