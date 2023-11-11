# Methods & Materials

## <span class="sk-h2-span">Datasets</span>

For training the model, we utilize the [MESA dataset](../datasets.md/#mesa-dataset). For this dataset, we leverage PPG, SpO2, and leg sensor data. Typically we would compute SpO2 directly from two or more PPG signals but only 1 PPG signal is provided in this dataset. Furthermore, we use the provided leg sensor as a proxy for general body movement. Ideally, capturing accelerometer on the wrist would provide better insights as we can extract movement as well as positioning (e.g. [z-angle](https://doi.org/10.1038/s41598-018-31266-z)).

---

## <span class="sk-h2-span">Feature Extraction</span>

For sleep stage classification, we are interested in capturing both short and long-term trends in a subject's physiological signals. In particular, as a subject transitions from one sleep stage to another, we expect to see changes in the subject's physiological signals. For example, as a subject transitions from wake to sleep, we might expect to see a decrease in heart rate and an increase in respiratory rate. Similarly, as a subject transitions from light sleep to deep sleep, we expect to see a decrease in heart rate and respiratory rate. While the general trend with AI is to use raw sensory input, we believe that feeding the model with pre-processed features provides a more robust and interpretable model. By extracting sensor agnostic features, we can better understand the model's decision making process and provide more accurate and actionable insights to the user. Furthermore, we beleive this will make the models more robust to sensor noise and drift and will be able to generalize to new subjects and environments.

In general, we compute features derived from the circulatory, respiratory, and muscular system. We leverage Ambiq's [PhysioKit Python package](https://ambiqai.github.io/physiokit) to extract these physiological features from raw sensory data. The following table provides a list of these features computed over sliding windows (e.g. 30 seconds).

* Inter-beat interval (IBI) Mean
* IBI St. Deviation
* IBI Median
* Heart Rate Variability (HRV) LF/HF Ratio
* SpO2 mean
* SpO2 St. Deviation
* SpO2 Median
* Movement mean
* Movement St. Deviation
* Movement Median
* Respiratory Rate (BPM)
* Quality of Signal (QoS)

Please refer to [sleepkit/features.py](https://github.com/AmbiqAI/sleepkit/tree/main/sleepkit/features.py) to examine the full list of features computed along with their implementations.

---

## <span class="sk-h2-span">Feature Normalization</span>

Physiological signals can vary greatly between subjects and even within a single subject. For example, a subject's heart rate can vary from 40 BPM to 100 BPM. In order to account for this, we normalize the features to have a mean of 0 and a standard deviation of 1 over an entire nights recording. This allows the model to learn the general trends in the data rather than the absolute values. For example, a subject's heart rate may be 80 BPM while another subject's heart rate may be 60 BPM. However, the general trend of the heart rate is the same for both subjects. By normalizing the features, the model can learn the general trend of the heart rate rather than the absolute value.

---

## <span class="sk-h2-span">Training Procedure</span>

For training the models, we utilize the following setup. We utilize a focal loss function [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf) to deal with the significant class imbalance. Rather than artificially oversampling or undersampling we provide class weight factors to penalize under-represented class losses more. The model is trained using Adam optimizer [Kingma et al., 2014](https://arxiv.org/pdf/1412.6980.pdf). We also employ a cosine decay learning rate scheduler with restarts [Loshchilov et al., 2016](https://arxiv.org/pdf/1608.03983.pdf) to improve the model's generalization ability. Additionally, we use early stopping to prevent overfitting based on loss metric.

---

## <span class="sk-h2-span">Evaluation Metrics</span>

For each dataset, 20% of the data is held out for validation and 20% of the data is held out for testing. The remaining 60% of the data is used for training. There is no mixing of subjects between the training, validation, and test sets. Furthermore, the test set is held fixed while training and validation are randomly split during training. We evaluate the models performance using a variety of metrics including loss, accuracy, F1 score, average precision (AP).

---

## <span class="sk-h2-span">Sleep Stage Classes</span>

Below outlines the grouping of sleep stages used for 2, 3, 4, and 5 stage sleep classification tasks.

=== "2-Stage"

    | CLASS | STAGES          |
    | ---- | ---------------- |
    | 0- WAKE | W                |
    | 1- SLEEP | N1, N2, N3, REM |

=== "3-Stage"
    | CLASS | STAGES          |
    | ---- | ---------------- |
    | 0- WAKE | W                |
    | 1- NREM | N1, N2, N3       |
    | 2- REM  | REM              |

=== "4-Stage"
    | CLASS | STAGES          |
    | ---- | ---------------- |
    | 0- WAKE | W                |
    | 1- CORE | N1, N2           |
    | 2- DEEP | N3               |
    | 3- REM  | REM              |

=== "5-Stage"
    | CLASS | STAGES          |
    | ---- | ---------------- |
    | 0- WAKE | W                |
    | 1- N1   | N1               |
    | 2- N2   | N2               |
    | 3- N3   | N3               |
    | 4- REM  | REM              |

---
