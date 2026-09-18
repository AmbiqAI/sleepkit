# Changelog

## 1.0.0a1 (unreleased)

- Introduce composable raw-record preprocessing with explicit signal semantics,
  deterministic caching, subject splits, and training-only fitted normalization.
- Add a synthetic staging recipe, independent LiteRT evaluation/inference,
  validated INT8 bundles, and Hugging Face staging with explicit upload.
- Upgrade TensorFlow to 2.21 and Keras to 3.15; separate optional dependencies
  and preserve historical artifacts and the legacy CLI.
- Add core-only and training/export CI checks. Historical physiological feature
  recipes and baseline release metadata are still being migrated.


## [0.11.1](https://github.com/AmbiqAI/sleepkit/compare/v0.11.0...v0.11.1) (2026-01-18)


### Bug Fixes

* publish to PyPI without reusable workflow ([#15](https://github.com/AmbiqAI/sleepkit/issues/15)) ([10a68d1](https://github.com/AmbiqAI/sleepkit/commit/10a68d13e103f78f7dc6d726738a31eeba70316d))

## [0.11.0](https://github.com/AmbiqAI/sleepkit/compare/v0.10.0...v0.11.0) (2026-01-18)


### Features

* add unit tests for metrics and utils ([c73dbe4](https://github.com/AmbiqAI/sleepkit/commit/c73dbe4c94016774ea8a684b0a6f98f2dde082e0))

## [0.10.0](https://github.com/AmbiqAI/sleepkit/compare/v0.9.0...v0.10.0) (2026-01-16)


### Features

* Update to Helia ecosystem ([#8](https://github.com/AmbiqAI/sleepkit/issues/8)) ([0e9cc55](https://github.com/AmbiqAI/sleepkit/commit/0e9cc556bee146a053e6417e4c2b3122ba74d2d4))
