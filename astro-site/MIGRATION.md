# Migration review

Tracking: AmbiqAI/sleepkit#43. Baseline: main e80e431.

## Preserved

- Existing documentation content and routes from docs/ and mkdocs.yml, with navigation refinements requested during review.
- Installation tabs, admonitions, Markdown tables, math, five Mermaid diagrams, and embedded Plotly figures.
- Training notebook code, three saved PNG figures, logs and text tables from the repository copy selected during review. The documentation and repository notebook copies have identical code cells but different saved outputs and metadata. The documentation copy has two PNG outputs; the repository copy has three. Neither original is changed.
- Existing authored routes, plus redirects from generated MkDocs API routes to source-generated reference pages.
- License documents and model-weight licensing boundaries.

## Narrow repairs

Two Python docstrings had unterminated example fences. Added closing fences and corrected sk.util to sk.utils in the plotting example. No runtime behavior changed.

The old datasets/factory and features/factory API links now redirect to their defining package modules. A notebook feature-guide link is normalized for its rendered route.

## Review fixes

- Fixed snippet expansion that inserted blank lines between table rows. Added regression coverage for table whitespace and nested snippet indentation.
- Normalize mixed-case Material callouts, unwrap example/install wrappers, and retain collapsed/expanded details. Guard every rendered page against leaked Material syntax, table text and image-width attributes.
- Long JSON/YAML and Python examples have a compact preview, native expansion, syntax-highlighted full code, copy and download. Full examples remain in Markdown and LLM exports.
- Removed duplicated Home navigation. Grouped staging and detection guides under their tasks. Shared architecture and component experiments, inherited from main PRs #33 and #36, are under Maintainer notes.
- Reworked Modes overview and corrected stale task-parameter/import snippets and orphaned annotation markers. No runtime behavior changed.

Validation: Astro check has no errors or warnings; build, output checks, four converter tests and ten browser tests pass. A browser sweep of all 59 authored Markdown pages at 390px found no viewport overflow or broken images. Inspected navigation, tabs, tables, modes and configuration screenshots, including dark mobile configuration. Clipboard content matches the JSON download.

## Existing gaps retained for follow-up

The synthetic dataset page no longer embeds the absent assets/segmentation_example.html; it explains synthesis and custom dataset integration. The detect page also names assets/sleep-detect-demo.html in commented content; that inactive content remains excluded.

Broader task/API/content inconsistencies belong to the existing product-readiness issue #18. This migration does not establish model validity or rerun notebook training, dataset acquisition, or hardware deployment.

## Deployment boundary

PRs build and test the site; main and manual main dispatch publish Pages. The package release workflow no longer invokes the docs workflow. Package publishing jobs and version metadata are unchanged. No production deployment has been performed for this branch.

## Navigation and installation refinement

Five major sections: Home, Getting started, User guide, Tasks and Reference. Every content page has section membership while preserving existing URLs. The local section matcher is resolved only for HELIA's section-matching imports so its header and sidebar use the same route assignment. A build guard checks section coverage; browser checks confirm both the selected navbar link and scoped sidebar on historical routes. Mobile retains access to all five sections with the current section expanded.

Landing installation examples are copyable Bash commands with no Termynal progress artifacts . Ten browser tests pass.

## Public documentation boundary

Private implementation API modules and their redirects are excluded. Shared architecture proposals, reusable-component experiments, feature-preparation implementation notes and unadopted draft license terms stay in the repository and are not published. Public task workflows and model licensing guidance remain. The public site now contains 55 authored Markdown pages, one notebook and 110 API modules (216 catalog symbols). Output checks reject internal routes and private modules in the catalog/reference model.

## Canonical sources

The one-time MkDocs adapter has been retired. Authored pages are now in `src/content/docs/`, navigation in `src/navigation.mjs`, and static assets in `public/`. API and notebook generation remain. See README.md for source ownership.
