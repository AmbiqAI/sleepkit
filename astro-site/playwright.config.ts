import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:8775/sleepkit/" },
  webServer: {
    command:
      "node node_modules/@ambiqai/helia-ui/scripts/serve-dist.mjs --port 8775 --base /sleepkit --dist dist",
    url: "http://127.0.0.1:8775/sleepkit/",
    reuseExistingServer: false,
  },
});
