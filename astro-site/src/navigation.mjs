import apiSidebar from "./data/api-sidebar.json" with { type: "json" };
const page = (label, slug) => ({ label, slug });
const group = (label, items) => ({ label, items, collapsed: false });
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
      group("Datasets", [
  {
    "label": "Datasets",
    "slug": "datasets"
  },
  {
    "label": "CMIDSS",
    "slug": "datasets/cmidss"
  },
  {
    "label": "MESA",
    "slug": "datasets/mesa"
  },
  {
    "label": "YSYW",
    "slug": "datasets/ysyw"
  },
  {
    "label": "STAGES",
    "slug": "datasets/stages"
  },
  {
    "label": "Synthetic",
    "slug": "datasets/synthetic"
  },
  {
    "label": "BYOD",
    "slug": "datasets/byod"
  }
]),
      group("Feature extraction", [
  {
    "label": "Features",
    "slug": "features"
  },
  {
    "label": "FS-W-PA-14",
    "slug": "features/fs_w_pa_14"
  },
  {
    "label": "FS-C-EAR-9",
    "slug": "features/fs_c_ear_9"
  },
  {
    "label": "FS-W-A-5",
    "slug": "features/fs_w_a_5"
  },
  {
    "label": "FS-H-E-10",
    "slug": "features/fs_h_e_10"
  },
  {
    "label": "FS-W-P-5",
    "slug": "features/fs_w_p_5"
  },
  {
    "label": "BYOFS",
    "slug": "features/byofs"
  }
]),
      group("Model architectures", [
  {
    "label": "Models",
    "slug": "models"
  },
  {
    "label": "BYOM",
    "slug": "models/byom"
  }
]),
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
        ...[
  {
    "label": "Model Zoo",
    "slug": "zoo"
  },
  {
    "label": "Detect",
    "slug": "zoo/detect"
  },
  {
    "label": "Stage",
    "slug": "zoo/stage"
  },
  {
    "label": "Apnea",
    "slug": "zoo/apnea"
  }
],
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
