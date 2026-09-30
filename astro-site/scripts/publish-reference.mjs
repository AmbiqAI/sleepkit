import { readFileSync, writeFileSync } from "node:fs";
import {
  RENDER_DEFAULTS,
  buildIndex,
  renderModuleMarkdown,
} from "../node_modules/@ambiqai/helia-ui/scripts/lib/reference-render.mjs";

const model = JSON.parse(
  readFileSync("dist/reference/api/reference.json", "utf8"),
);
const options = {
  ...RENDER_DEFAULTS,
  base: "/sleepkit/",
  site: "https://ambiqai.github.io",
};
const index = buildIndex(model, options);
let full = readFileSync("dist/llms-full.txt", "utf8");
function publish(module) {
  const page = `${options.base}reference/api/${module.path.toLowerCase().replaceAll(".", "/")}/`;
  const markdown = renderModuleMarkdown(module, { index, options });
  const marker = `<!-- ${options.site}${page} -->`;
  const start = full.indexOf(marker);
  if (start < 0) throw new Error(`Missing API export section: ${page}`);
  const next = full.indexOf("\n<!-- https://", start + marker.length);
  full =
    full.slice(0, start) +
    marker +
    "\n\n" +
    markdown +
    "\n" +
    (next < 0 ? "" : full.slice(next));
  writeFileSync(`dist/${page.slice(options.base.length)}index.md`, markdown);
  (module.submodules ?? []).forEach(publish);
}
model.modules.forEach(publish);
writeFileSync("dist/llms-full.txt", full);
