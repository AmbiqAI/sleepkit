# Changelog

## [0.12.0](https://github.com/AmbiqAI/sleepkit/compare/v0.11.1...v0.12.0) (2026-10-07)


### Features

* Add detection error diagnostics and exact historical comparison ([#29](https://github.com/AmbiqAI/sleepkit/issues/29)) ([19badc2](https://github.com/AmbiqAI/sleepkit/commit/19badc2b753694ff97e4dc3e4dd2d50fecf1e015))
* add explicit saved-feature staging replay ([#39](https://github.com/AmbiqAI/sleepkit/issues/39)) ([aa6c368](https://github.com/AmbiqAI/sleepkit/commit/aa6c3684e58f4904a6c85b85183d0cd974b337d8))
* Add golden membership baseline and stage profiling ([#34](https://github.com/AmbiqAI/sleepkit/issues/34)) ([2952aa2](https://github.com/AmbiqAI/sleepkit/commit/2952aa27b1d6fc509d94b3a68d9e8dc907931586))
* add reproducible staging training and a golden experiment ([#41](https://github.com/AmbiqAI/sleepkit/issues/41)) ([c6b67ea](https://github.com/AmbiqAI/sleepkit/commit/c6b67eaef591ed7032c80aa8165fe59cc3b53a0b))
* Add train-only int8 conversion and verified release bundle ([#30](https://github.com/AmbiqAI/sleepkit/issues/30)) ([43f93a3](https://github.com/AmbiqAI/sleepkit/commit/43f93a3f8aafadd42cc2556c74989a32de3bc95a))
* Prove reusable experiment blocks with a portable signal recipe ([#36](https://github.com/AmbiqAI/sleepkit/issues/36)) ([f3a8637](https://github.com/AmbiqAI/sleepkit/commit/f3a8637a76acba232f0aec9207b3bf1a77891a87))
* Stage licensed releases from verified experiment bundles ([#32](https://github.com/AmbiqAI/sleepkit/issues/32)) ([444fb19](https://github.com/AmbiqAI/sleepkit/commit/444fb19f6a93d0ec4ff2b9ee1164303ce2e36dc2))
* Verify scoring evidence and report first frozen detection experiment ([#28](https://github.com/AmbiqAI/sleepkit/issues/28)) ([fb52ca3](https://github.com/AmbiqAI/sleepkit/commit/fb52ca3bca804cbe12c28206d5ffdd3ec927c01d))


### Bug Fixes

* bind comparison checks to evaluated recipe bytes ([#38](https://github.com/AmbiqAI/sleepkit/issues/38)) ([73a858e](https://github.com/AmbiqAI/sleepkit/commit/73a858ea497eef7994235f5aece909fa2105830c))
* **deps:** bound helia-edge below 0.8 ([#53](https://github.com/AmbiqAI/sleepkit/issues/53)) ([9a1302f](https://github.com/AmbiqAI/sleepkit/commit/9a1302fd0eee7119cc9d6a882c3b339da88aa1e2)), closes [#52](https://github.com/AmbiqAI/sleepkit/issues/52)
* **staging:** validate the model interface in staging evaluate ([#42](https://github.com/AmbiqAI/sleepkit/issues/42)) ([e80e431](https://github.com/AmbiqAI/sleepkit/commit/e80e431288d92e96b8d753666e8df6e65f51a96b)), closes [#40](https://github.com/AmbiqAI/sleepkit/issues/40)


### Performance Improvements

* Vectorize detection feature preparation with exact baseline parity ([#35](https://github.com/AmbiqAI/sleepkit/issues/35)) ([11584fb](https://github.com/AmbiqAI/sleepkit/commit/11584fb7f80be5e2207e5a60b89d3034650f3d56))


### Documentation

* adopt official Ambiq footer artwork ([#54](https://github.com/AmbiqAI/sleepkit/issues/54)) ([ef0f61b](https://github.com/AmbiqAI/sleepkit/commit/ef0f61beeca956c09dcfe0d7fc77db1d4bdc533a))
* adopt shared mobile navigation and compact terminals ([#46](https://github.com/AmbiqAI/sleepkit/issues/46)) ([e86a6e0](https://github.com/AmbiqAI/sleepkit/commit/e86a6e018b7205217a89ae1297cba1a7654369b7))
* align sleepKIT landing page with KIT sites ([#56](https://github.com/AmbiqAI/sleepkit/issues/56)) ([16ae4e0](https://github.com/AmbiqAI/sleepkit/commit/16ae4e040c6760beb96681bdb24067d1a1cf75fa))
* Define shared KIT architecture and multibackend heliaEDGE direction ([#33](https://github.com/AmbiqAI/sleepkit/issues/33)) ([6d20ef7](https://github.com/AmbiqAI/sleepkit/commit/6d20ef7e5ddd9fda34b0dc3b464b2557d9d7e5e0))
* Document BSD tooling and per-model license choices ([#31](https://github.com/AmbiqAI/sleepkit/issues/31)) ([5117cc0](https://github.com/AmbiqAI/sleepkit/commit/5117cc020563a84e8fda2c8baae5a07078c1b172))
* make Astro content canonical and retire MkDocs adapter ([#49](https://github.com/AmbiqAI/sleepkit/issues/49)) ([ca3db58](https://github.com/AmbiqAI/sleepkit/commit/ca3db58f597215fb69e3381657a9a757cc57c671))
* migrate sleepKIT to Astro ([#44](https://github.com/AmbiqAI/sleepkit/issues/44)) ([18639ad](https://github.com/AmbiqAI/sleepkit/commit/18639ad611491df06b554a324360c32076f1fd23))
* remove landing TOC and align mobile navigation ([#58](https://github.com/AmbiqAI/sleepkit/issues/58)) ([c972bd6](https://github.com/AmbiqAI/sleepkit/commit/c972bd6819191b68ae31d9e290b5f32a722450f3))

## [0.11.1](https://github.com/AmbiqAI/sleepkit/compare/v0.11.0...v0.11.1) (2026-01-18)


### Bug Fixes

* publish to PyPI without reusable workflow ([#15](https://github.com/AmbiqAI/sleepkit/issues/15)) ([10a68d1](https://github.com/AmbiqAI/sleepkit/commit/10a68d13e103f78f7dc6d726738a31eeba70316d))

## [0.11.0](https://github.com/AmbiqAI/sleepkit/compare/v0.10.0...v0.11.0) (2026-01-18)


### Features

* add unit tests for metrics and utils ([c73dbe4](https://github.com/AmbiqAI/sleepkit/commit/c73dbe4c94016774ea8a684b0a6f98f2dde082e0))

## [0.10.0](https://github.com/AmbiqAI/sleepkit/compare/v0.9.0...v0.10.0) (2026-01-16)


### Features

* Update to Helia ecosystem ([#8](https://github.com/AmbiqAI/sleepkit/issues/8)) ([0e9cc55](https://github.com/AmbiqAI/sleepkit/commit/0e9cc556bee146a053e6417e4c2b3122ba74d2d4))
