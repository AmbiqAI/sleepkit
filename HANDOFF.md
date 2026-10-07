# Home-page consistency handoff

Goal: align sleepKIT with the KIT documentation sites and physioKIT.

State:
- Done locally in `codex/kit-home-consistency`: replace the hero dot with the existing product icon, render the links below the hero with the shared helia-ui Button, and set `header.titleRegularPrefix` so the prefix is regular and KIT stays bold. The home body follows the shared order: overview, four task cards with small pink icons, a short uv installation path, and four shared navigation cards. Detailed mode, dataset and model material lives on dedicated pages; legacy home anchors remain.
- Verified: Astro check and build pass. Output contract and route checks pass. All 20 Playwright tests pass. The first PR CI run exposed three tests that still searched the shortened home page for the old installation tabs. They now test the Quickstart page, where those tabs remain. Desktop, 785px and 390px views were inspected in light and dark mode, with no horizontal overflow. Browser regression checks verify readable shared quick buttons in both themes and navigation from card-body clicks.
- Preview: http://127.0.0.1:8776/sleepkit/
- Tracked by AmbiqAI/sleepkit#55. Draft PR #56 is open; no release or deployment has occurred.

Decisions:
- Reuse the repository's existing product icon and keep its current hero color and destination links.
- Use the shared Button component for the links under the hero; do not introduce a site-specific pill design.
- Pin helia-ui v0.1.0-alpha.24, published from f7158bf, with an npm 11.19.0 lockfile. Clean installs reproduce the shared header without a local patch.

Next: finish checks against the released package, final review and CI, merge under Adam's authorization, then verify the Pages deployment.

Not-found handling: restrict the product hero to the home route so unknown routes show the 404 page; built output and browser checks cover the fallback.


Release validation: clean installation of alpha.24 passed. Final check/build/output checks pass, with 20 rendered acceptance checks passing against the clean released dependency. User authorized merging after green CI.
