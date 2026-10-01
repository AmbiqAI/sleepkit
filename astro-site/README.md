# sleepKIT documentation site

The Astro/Starlight site reads the authored Markdown in `../docs` and the saved notebook in `../notebooks` and generates Python reference pages with Griffe. The Python package is inspected statically; no training dependencies or notebook execution are required.

Use Node 24, Python 3.12 and uv:

```sh
npm ci
npx playwright install chromium
npm run dev -- --port 8774
```

Validation:

```sh
npm run check
npm run build
npm run check:output
npm test
```

Edit authored Markdown/MDX in `src/content/docs/`, navigation in `src/navigation.mjs`, static redirects in `src/redirects.json`, and static assets in `public/`. These are canonical sources and are never replaced by the build.

`prepare:docs` generates only downloadable configuration examples, notebook guides/assets, and Python API pages/data. API output under `src/content/docs/reference/`, notebook `.md` pages under `guides/`, `src/data/`, and `public/{reference,notebooks,examples}/` are ignored. Edit Python docstrings or the notebooks in `../notebooks/` for those outputs. There is no MkDocs configuration or Markdown conversion step.

The notebook page uses `notebooks/train-detect-model.ipynb`, preserving its saved outputs, three figures and downloadable original. The earlier documentation copy is retained in `notebooks/archive/` as an archive download. Neither source is overwritten. Historical notebook outputs do not establish validation against the latest package.

The documentation workflow checks pull requests and publishes main to GitHub Pages. It can also be dispatched manually on main. It does not publish Python packages. Package publishing remains in the release workflows; documentation no longer depends on a package release.

Public reference generation excludes underscore-prefixed implementation modules before rendering, so they do not appear in pages, catalog, search or machine-readable exports. `scripts/public-docs.mjs` lists repository-only maintainer/proposal pages omitted from publication. Their sources remain in the repository.

Golden-contract evidence remains canonical under `../docs/evidence/` because experiment declarations reference those paths. The build copies it to `public/evidence/`; do not edit the generated copy.
