# sleepKIT landing consistency

Goal: issue #55, align the landing page with heartKIT, compressionKIT and physioKIT. PR #56 is ready for review on codex/kit-home-consistency. Adam authorized merge after green CI.

Done: product hero icon, regular sleep prefix and bold KIT navbar suffix, shared quick buttons, overview, task cards with small pink icons and neutral borders, short installation path and full-card documentation links. Historical home anchors remain. The custom hero only renders on the home route.

Verified: immutable helia-ui v0.1.0-alpha.24 pin and npm 11.19.0 lockfile, clean install, Astro check/build/output and 20 browser tests. Light/dark mobile and desktop screenshots inspected. Tests exercise title weights, logo, readable buttons, card-body navigation and absent home hero on 404. Full PR CI passed before the final regression assertion; CI on this assertion must finish before merge.

Review: prior dependency and installation-test findings resolved. Final review requested a more explicit 404 assertion and consistent handoff status; both addressed.

Next: verify final CI, complete review, squash merge and confirm Pages deployment. No publication of this landing change is claimed yet.

Preview: http://127.0.0.1:8776/sleepkit/. Builds use generated API and notebook material; training is not run. A clean install needs no package patch.
