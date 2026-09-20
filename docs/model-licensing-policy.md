# Code, site and model licensing

sleepKIT's code, recipes and Ambiq-authored documentation/site content use
**BSD-3-Clause**, except material carrying a separate notice. Third-party assets,
dependencies and datasets retain their own terms. The repository's
[LICENSE](https://github.com/AmbiqAI/sleepkit/blob/main/LICENSE) remains unchanged.
Generic training, preprocessing, evaluation and export code can be used on any
hardware, including to train independent models.

Our intent is to make most model releases publicly downloadable. Each new model
release has its own explicit license covering named weight files and learned
preprocessing state. A public download, successful conversion or BSD-licensed
training script does not establish unrestricted model-use rights. This policy does
not retract rights granted in previous releases or relicense historical artifacts.

## Model release options

| Option | When to use it | Commercial use | Hardware |
| --- | --- | --- | --- |
| BSD-3-Clause model | All relevant upstream rights permit a permissive release | Permitted by this license | Any |
| Custom Ambiq device model license | Upstream rights permit the restriction and the release explicitly opts in | Proposed terms allow Ambiq deployment and host development/evaluation | Ambiq embedded execution; general-purpose host evaluation allowed |
| Dataset-compatible research license | Dataset or parent-model terms constrain the release | Follow the exact upstream terms; NC is not commercial permission | No extra restriction that conflicts with those terms |

The [custom Ambiq terms](licenses/ambiq-device-model-license-draft.md) are a draft,
not an active license. They are intended for otherwise compatible models, not an
addendum to a Creative Commons license. Each release chooses terms explicitly;
there is no default hardware restriction on every sleepKIT model.

Describe models with noncommercial or hardware restrictions as **public research
weights** or **source-available models**, and state the restriction prominently.
Reserve open-source claims for releases that actually meet the relevant definition.
The [Open Source Definition](https://opensource.org/osd) requires unrestricted
fields of endeavor and technology neutrality; public access alone is insufficient.

## Dataset terms and intended use

Assess the dataset's access agreement, license, competition rules and any parent
weights together. Multi-dataset models need compatible rights across all sources.
Preserve required attribution and notices. Permission to redistribute a model does
not automatically permit redistribution of source recordings or derived features.

**Noncommercial and nonproduction are different.** Under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en),
commercial purpose matters; internal R&D or a free customer demo is not automatically
noncommercial. Conversely, a noncommercial use is not defined by whether it is a
prototype. A model license cannot cure unauthorized data access or training.

Do not attach an Ambiq-only condition to material governed by CC BY-NC-SA:
sections 2(a)(5)(C) and 3(b)(3) prohibit additional downstream restrictions. If that
restriction is essential, use data and parent weights permitting it, or obtain
separate permission covering the intended use. Do not weaken an upstream constraint
by selecting a more permissive label.

Whether particular weights are an adaptation of training data is a separate,
fact-dependent question. Our conservative policy is to honor relevant dataset
conditions through the model release rather than assume training removes them.
[Creative Commons' AI guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
distinguishes this cautious approach from conclusions about when permission is
legally required. We do not claim every model automatically inherits its dataset's
copyright license.

License permission, task validity and device readiness are recorded separately.
For example, a commercially permitted model may still lack clinical validation,
MCU operator support, measured RAM usage or acceptable latency.

## Record the decision with each release

Use a short human-readable decision record in the model card or accompanying
release documentation. This is evidence for a maintainer decision, not a new
pipeline configuration language. Record:

- Exact model version, covered filenames/hashes, parent weights and licenses.
- Dataset versions and provenance; primary terms URLs, retrieval date and a local
  snapshot/hash where terms are mutable or require authentication.
- How data access and the actual training/evaluation purpose satisfy those terms;
  unresolved facts remain explicit.
- Selected license and attribution, commercial-use conditions, permitted hardware
  and any separate permissions relied on.
- Release status, decision date and responsible maintainer; links to evaluation
  and deployment limitations, separately from permission to use the model.

A maintainer can make and record an evidence-based decision without a standing
legal-review requirement. A concrete incompatibility or missing material permission
must be resolved before publishing affected artifacts. If it cannot be resolved,
keep that release local or train a replacement on compatible data.

For Hugging Face, include the operative license text and the same license in the
model card and bundle manifest. Use standard IDs such as `bsd-3-clause` or
`cc-by-nc-sa-4.0`. For a finalized custom license, use `license: other` plus
`license_name` and a link to its text as described in
[Hugging Face's license documentation](https://huggingface.co/docs/hub/repositories-licenses).
List any differently licensed files explicitly. Do not use a BSD badge to describe
restricted weights. The publisher's license-presence check is not a rights review.

## Current CMIDSS membership candidate

Status on 2026-09-20: **staged locally; license candidate identified; not published**.
This is the five-epoch nightly-period-membership experiment and its fixed int8
conversion, not clinical sleep/wake classification. See the
[conversion evidence and file inventory](detection-int8.md).

| Decision field | Current evidence |
| --- | --- |
| Candidate | `membership-tcn-seed0`, fixed int8 conversion dated 2026-09-20 |
| Int8 SHA-256 | `46c8cdc4f10461818959c4100b9232c2d6ba09bad7ba94f89a357949d8854a13` |
| Intended scope | Keras/float/int8 weights and fitted preprocessing; enumerate all exact release hashes when restaging |
| Dataset | Child Mind Institute — Detect Sleep States (CMIDSS), Kaggle competition 53666 |
| Primary terms | [Competition rules](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/rules), retrieved 2026-09-20 |
| Rules snapshot | Published rules page 239161; content SHA-256 `7713a36bc653d75702a408e2207d2c3a11990562e94cb97ee842d2dfe727a0f5` |
| Data conditions | CC BY-NC-SA 4.0; noncommercial access/use; competition rules take precedence over conflicting CC terms; restrictions on sharing competition data |
| Proposed model terms | Standard CC BY-NC-SA 4.0 as a conservative release policy, **without an Ambiq hardware restriction** |
| Open facts | Document actual data-access basis and training/evaluation purpose, competition submission/public-code-sharing history and existing grants; establish model redistribution basis and required attribution |
| Release decision | Pending those facts; no operative model license has been applied to the staged bundle |

The rules' LGPL 2.1 winner-license clause does not by itself license this model.
Their separate section 8(B) public-code-sharing provision also needs an
applicability check against what was submitted or shared; it is not an automatic
grant for these weights. The staged bundle excludes source recordings, subject identifiers and actual
calibration arrays. That helps avoid redistributing data but does not alone establish
release rights. Preserve the original experimental bundle; any licensed release
must be restaged with explicit scope, notices and updated integrity metadata.

Historical sleepKIT models remain subject to their existing grants and source terms.
Their access, training provenance and per-artifact license scope need individual
review before making new commercial-use claims or republishing them under new terms.
