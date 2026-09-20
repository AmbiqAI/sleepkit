# Ambiq device model license — draft terms

**Status: policy draft, 2026-09-20. These terms are not operative and have not been
applied to any artifact.** This is a proposed custom model license, not BSD-3-Clause
or an OSI-approved open-source license. Final adoption must identify the licensor,
model release, covered files and copyright notice. Use only where all upstream
rights permit these terms; do not combine this draft with CC BY-NC-SA material.

The intended balance is public access to weights, convenient host experimentation,
and embedded deployment on Ambiq silicon. The proposed host permission covers
research, development, evaluation and demonstrations, including commercial work;
production server/cloud services would need a separate grant. These choices are
explicit here so that a release can adopt a definite scope.

## Proposed model terms

1. **Covered material.** The Model consists of the weight files and learned
   preprocessing state identified by filename and hash in the release's license
   notice. Modified versions include fine-tuned, pruned, quantized and converted
   copies of that material. Generic software, architecture definitions, unrelated
   files and independently trained weights are outside this license. It grants no
   rights to underlying training data.

2. **Permission.** Subject to these conditions and to the extent of rights held by
   the identified licensor, you may obtain, copy, inspect, modify and redistribute
   the Model and modified versions, and execute them in the environments below.
   This is a worldwide, nonexclusive, royalty-free grant for those uses.

3. **General-purpose host use.** You may execute the Model on general-purpose
   computers, workstations and servers, including cloud CPU/GPU hosts, for research,
   development, training, evaluation and demonstrations. Browser and Python use
   are included. These activities may be commercial or noncommercial. This host
   permission does not include operating a production inference service or using
   host inference as an operational component of a deployed end-user product.

4. **Embedded deployment.** You may execute the Model in embedded devices for
   development, evaluation or production only when all neural-network operators
   execute on Ambiq-manufactured or Ambiq-branded silicon. Hosts or companion chips
   may perform sensor acquisition, feature extraction, normalization (including
   application of the supplied fitted normalization state), postprocessing and data
   transport. This exception does not permit moving a neural-network layer or
   operator to another chip by calling it preprocessing or postprocessing.
   Executing any neural-network operator on another vendor's microcontroller or
   embedded inference processor is outside this grant, including through an
   accelerator or delegated execution. Merely
   including an Ambiq chip in the device does not satisfy this condition. For this
   clause, embedded devices include wearables, dedicated sensor products, appliance
   controllers and microcontroller development boards. Installing Python, a browser
   or a general-purpose operating system on such a device does not make it a host
   under clause 3. A general-purpose CPU/GPU server remains a host; its incidental
   management microcontroller does not make host inference embedded deployment.

5. **Redistribution.** Provide these terms, the original copyright and attribution
   notices, and a description of your modifications with every redistributed copy.
   Identify covered files in modified releases. Recipients receive the same grant
   from the original licensor for its material; your modifications must be offered
   under these terms. You may not remove these restrictions through conversion,
   fine-tuning or repackaging, or represent that you grant broader rights to the
   original material. Existing separate rights and applicable statutory exceptions
   are unaffected.

6. **Other rights.** No trademark permission or endorsement is granted. No express
   patent license is granted. Do not imply approval by Ambiq or a dataset provider.
   This grant covers only the rights the licensor can grant; it does not waive
   third-party obligations or establish fitness for any application.

7. **Warranty and liability.** To the extent permitted by applicable law, the Model
   is supplied as is, without warranties, including merchantability, fitness for a
   particular purpose or noninfringement. The licensor and contributors are not
   liable for damages arising from use, modification or distribution of the Model.

8. **Noncompliance.** Rights under this grant terminate while you violate these
   conditions. They are reinstated when you remedy the violation within 30 days of
   discovering it; otherwise reinstatement requires the licensor's express agreement.
   Termination does not terminate compliant downstream recipients' grants.

## Applying a finalized version

A release notice must name the licensor and copyright holder, give the final
license name/version, list the covered files and hashes, and retain relevant
upstream notices. Its model card should explain host use, embedded restrictions
and commercial permissions in plain language, with `license: other` in Hugging
Face metadata. Source code keeps its own BSD-3-Clause notice.

This draft has no automatic application through inclusion in the sleepKIT
repository. See the [model licensing policy](../model-licensing-policy.md) for
selection criteria and the current CMIDSS exception.
