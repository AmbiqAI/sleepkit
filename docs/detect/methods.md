# Methods & Materials

## <span class="sk-h2-span">Datasets</span>

For training the included models, we utilize the [CMIDSS](../datasets.md/#cmidss-dataset) dataset from the Child Mind Institute - Detect Sleep States Kaggle competition. For this dataset, we leverage only the data from a 3-axis accelerometer.

---

## <span class="sk-h2-span">Feature Extraction</span>

In this task, we are interested in capturing the general trends in a subject's physical activity. Specifically from a 3-axis accelerometer, we compute the mean and st. dev of ENMO and z-angle. ENMO provides a more consistent measure of physical activity across different age groups and body types. Also, ENMO is more suitable than actigraphy `counts` as the latter is manufacturer specific. The z-angle is a measure of the angle of the wrist with respect to the ground. The z-angle has proven to be very valuable in distinguishing between inactivity bouts and sleep ([2018 Hees](https://doi.org/10.1038/s41598-018-31266-z)). Furthermore, we provide temporal embedding of the time of day to account for the 24-hour circadian cycle.

---

## <span class="sk-h2-span">Feature Normalization</span>

Since physical activity can vary greatly between subjects and even within a single subject, we normalize the features across the temporal window fed into the model. For the pre-trained models, we typically pass 4 hours of data to better capture long-term trends.

---

## <span class="sk-h2-span">Training Procedure</span>

For training the models, we utilize the following setup. We utilize a focal loss function [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf) to deal with the significant class imbalance. Rather than artificially oversampling or undersampling we provide class weight factors to penalize under-represented class losses more. The model is trained using Adam optimizer [Kingma et al., 2014](https://arxiv.org/pdf/1412.6980.pdf). We also employ a cosine decay learning rate scheduler with restarts [Loshchilov et al., 2016](https://arxiv.org/pdf/1608.03983.pdf) to improve the model's generalization ability. Additionally, we use early stopping to prevent overfitting based on loss metric.

---

## <span class="sk-h2-span">Evaluation Metrics</span>

For each dataset, 20% of the data is held out for validation and 20% of the data is held out for testing. The remaining 60% of the data is used for training. There is no mixing of subjects between the training, validation, and test sets. Furthermore, the test set is held fixed while training and validation are randomly split during training. We evaluate the models performance using a variety of metrics including loss, accuracy, F1 score, average precision (AP).

---

## <span class="sk-h2-span">Sleep Detection Classes</span>

Below outlines the grouping of sleep detection task.

=== "2-Stage"

    | CLASS | STAGES          |
    | ---- | ---------------- |
    | 0- WAKE | W                |
    | 1- SLEEP | N1, N2, N3, REM |
