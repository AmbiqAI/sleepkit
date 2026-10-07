# Home-page consistency handoff

Goal: align sleepKIT with the KIT documentation sites and physioKIT.

State:
- Done locally in `codex/kit-home-consistency`: replace the hero dot with the existing product icon, render the links below the hero with the shared helia-ui Button, and set `header.titleRegularPrefix` so the prefix is regular and KIT stays bold. The home body follows the shared order: overview, four task cards with small pink icons, a short uv installation path, and four shared navigation cards. Detailed mode, dataset and model material lives on dedicated pages; legacy home anchors remain.
- Verified: Astro check and build pass. Output contract and route checks pass. All 17 Playwright tests pass. The first PR CI run exposed three tests that still searched the shortened home page for the old installation tabs. They now test the Quickstart page, where those tabs remain. Desktop, 785px and 390px views were inspected in light and dark mode, with no horizontal overflow. The shared navigation cards respond to body clicks and keyboard focus.
- Preview: http://127.0.0.1:8776/sleepkit/
- Tracked by AmbiqAI/sleepkit#55. The site PR is pending; no release or deployment has occurred.

Decisions:
- Reuse the repository's existing product icon and keep its current hero color and destination links.
- Use the shared Button component for the links under the hero; do not introduce a site-specific pill design.
- The header option is implemented in the local helia-ui worktree `/Users/adam.page/Ambiq/helia/helia-ui-header-prefix` but is not released. Preview dependencies have a local, untracked copy of that implementation. The consumer package pin stays at its current immutable version until a new helia-ui tag exists.

Next:
1. Open a draft site PR linked to issue #55 and request review.
2. After helia-ui#193 is merged and released, pin its immutable tag, regenerate the lockfile, rebuild, and review both themes.
3. Resolve review and CI findings, then ask Adam for approval. Do not merge without that approval.

Gotcha: `npm ci` resets the preview-only helia-ui copy in `node_modules`; restore or use the released tag before rebuilding.
