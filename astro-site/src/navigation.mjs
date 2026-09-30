import sidebar from "./data/sidebar.json" with { type: "json" };
import apiSidebar from "./data/api-sidebar.json" with { type: "json" };
const page = (label, slug) => ({ label, slug });
const group = (label, items) => ({ label, items, collapsed: false });
const original = (label) =>
  sidebar
    .find((item) => item.label === label)
    .items.map((item) => ({
      ...item,
      slug: item.slug?.replace(/\.ipynb$/, ""),
    }));
const apiGroups = Object.entries(
  Object.groupBy(apiSidebar, (item) => item.label.split(".")[1] || "Package"),
).map(([label, items]) => ({ label, items, collapsed: true }));
export const sections = [
  { label: "Home", href: "/sleepkit/", sidebar: false },
  {
    label: "Getting started",
    href: "/sleepkit/quickstart/",
    sidebar: [
      page("Install and quickstart", "quickstart"),
      page("Command line", "usage/cli"),
      page("Python usage", "usage/python"),
      page("Workflow overview", "modes"),
    ],
  },
  {
    label: "User guide",
    href: "/sleepkit/guides/",
    sidebar: [
      page("Overview", "guides"),
      group("Run an experiment", [
        page("Configuration", "modes/configuration"),
        page("Download data", "modes/download"),
        page("Train", "modes/train"),
        page("Evaluate", "modes/evaluate"),
        page("Export", "modes/export"),
        page("Demo", "modes/demo"),
      ]),
      group("Datasets", original("Datasets")),
      group("Feature extraction", original("Features")),
      group("Model architectures", original("Models")),
      group("Tutorials", [
        page("Train a detection model", "guides/train-detect-model"),
        page("Staging ablation", "guides/stage-ablation"),
      ]),
    ],
  },
  {
    label: "Tasks",
    href: "/sleepkit/tasks/",
    sidebar: [
      page("Overview", "tasks"),
      group("Sleep detection", [
        page("Task guide", "tasks/detect"),
        page("Golden experiment", "detection-golden"),
        page("Profiling", "detection-profiling"),
      ]),
      group("Sleep staging", [
        page("Task guide", "tasks/stage"),
        page("Saved-feature baseline", "staging-baseline"),
        page("Train a staging model", "staging-training"),
      ]),
      page("Sleep apnea", "tasks/apnea"),
      page("Custom tasks", "tasks/byot"),
    ],
  },
  {
    label: "Reference",
    href: "/sleepkit/reference/",
    sidebar: [
      page("Python API catalog", "reference"),
      group("Model zoo", [
        ...original("Model Zoo"),
        page("Model artifacts", "huggingface-artifacts"),
      ]),
      group("Licensing", [
        page("Model licensing", "model-licensing-policy"),
      ]),
      ...apiGroups,
    ],
  },
];
const flatten = (items) =>
  items.flatMap((item) => (item.items ? flatten(item.items) : [item]));
export const sectionByPath = Object.fromEntries(
  sections.flatMap((section) =>
    section.sidebar === false
      ? [[section.href, section.href]]
      : flatten(section.sidebar).map((item) => [
          item.slug !== undefined ? `/sleepkit/${item.slug}/` : item.link,
          section.href,
        ]),
  ),
);
