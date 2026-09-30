export const internalPages = new Set([
  "shared-foundation.md",
  "reusable-blocks.md",
  "detection-preparation-performance.md",
  "licenses/ambiq-device-model-license-draft.md",
]);

export const isPrivateModule = (path) =>
  path.split(".").some((part) => part.startsWith("_"));
