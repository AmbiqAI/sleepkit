# Sleep Apnea

## <span class="sk-h2-span">Overview</span>

Sleep apnea detection is the process of identifying hypopnea/apnea events. This task is useful for identifying sleep disorders and for monitoring sleep quality.

## <span class="sk-h2-span">Sleep Apnea Types</span>

There are three main types of sleep apnea: obstructive, central, and mixed. Obstructive sleep apnea (OSA) is the most common type of sleep apnea. It occurs when the throat muscles relax and block the airway during sleep. Central sleep apnea (CSA) occurs when the brain fails to send the proper signals to the muscles that control breathing. Mixed sleep apnea is a combination of both obstructive and central sleep apnea. Hypopnea is a partial blockage of the airway that results in shallow breathing.

=== "Obstructive Sleep Apnea"

    Obstructive sleep apnea (OSA) is the most common type of sleep apnea. It occurs when the throat muscles relax and block the airway during sleep. This leads to pauses in breathing and can result in low oxygen levels in the blood. OSA is often associated with snoring and daytime sleepiness.

=== "Central Sleep Apnea"

    Central sleep apnea (CSA) occurs when the brain fails to send the proper signals to the muscles that control breathing. This results in pauses in breathing during sleep. CSA is less common than OSA and is often associated with heart failure or stroke. Clinically, central sleep apnea is defined as a lack of respiratory effort for at least 10 seconds.

=== "Mixed Sleep Apnea"

    Mixed sleep apnea is a combination of both obstructive and central sleep apnea. It occurs when there is a blockage of the airway and a failure of the brain to send the proper signals to the muscles that control breathing. Mixed sleep apnea is less common than OSA or CSA. Clinically, mixed sleep apnea is defined as a combination of obstructive and central apneas.

=== "Hyopnea"

    Hypopnea is a partial blockage of the airway that results in shallow breathing. It is less severe than apnea but can still disrupt sleep and lead to daytime sleepiness. Clinically, hypopnea is defined as a reduction in airflow of at least 30% for at least 10 seconds, accompanied by a decrease in oxygen saturation of at least 3%.


## <span class="sk-h2-span">Model Zoo</span>

The following table provides the latest performance and accuracy results for pre-trained models. Additional result details can be found in [Model Zoo → Apnea](../zoo/apnea.md).


--8<-- "assets/zoo/apnea/apnea-model-zoo-table.md"

---

## <span class="sk-h2-span">Target Classes</span>

Below outlines the classes available for apnea classification. When training a model, the number of classes, mapping, and names must be provided.

--8<-- "assets/tasks/apnea/apnea-classes.md"

## <span class="sk-h2-span">References</span>

* [ApSense: Data-driven Algorithm in PPG-based Sleep Apnea Sensing](https://doi.org/10.48550/arXiv.2306.10863)
* [Multimodal Sleep Apnea Detection with Missing or Noisy Modalities](https://arxiv.org/pdf/2402.17788v1.pdf)
