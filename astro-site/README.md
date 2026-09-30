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

`prepare:docs` regenerates `src/content/docs`, `src/data` and `public`. Do not edit those directories. Edit Markdown under `../docs`, Python docstrings, or the owning scripts. `mkdocs.yml` supplies the existing navigation during migration. `.cache/conversion.json` records syntax conversions and items requiring visual inspection.

The notebook page uses `notebooks/train-detect-model.ipynb`, preserving its saved outputs, three figures and downloadable original. The earlier `docs/guides/` copy is retained as an archive download. Neither source is overwritten. Historical notebook outputs do not establish validation against the latest package.

The documentation workflow checks pull requests and publishes main to GitHub Pages. It can also be dispatched manually on main. It does not publish Python packages. Package publishing remains in the release workflows; documentation no longer depends on a package release.

Public reference generation excludes underscore-prefixed implementation modules before rendering, so they do not appear in pages, catalog, search or machine-readable exports. `scripts/public-docs.mjs` lists repository-only maintainer/proposal pages omitted from publication. Their sources remain in the repository.
