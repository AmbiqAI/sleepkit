# CMIDSS Dataset

## <span class="sk-h2-span">Overview</span>

This dataset comes from the Child Mind Institute - Detect Sleep States (CMIDSS) Kaggle competition. The dataset comprises 300 subjects with over 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wakeup, the end of sleep. While the original data contains 3-axis accelerometer data, this dataset only contains the euclidean norm minus one (ENMO) and z-angle reported every 5 seconds.

More info available on [PhysioNet website](https://physionet.org/content/challenge-2018/1.0.0)

## <span class="sk-h2-span">Funding</span>

The data was provided by the Healthy Brain Network, a landmark mental health study based in New York City that will help children around the world. In the Healthy Brain Network, families, community leaders, and supporters are partnering with the Child Mind Institute to unlock the secrets of the developing brain. In addition to generous support provided by the Kaggle team, financial support has been provided by the Stavros Niarchos Foundation (SNF) as part of its Global Health Initiative (GHI) through the SNF Global Center for Child and Adolescent Mental Health at the Child Mind Institute.

## <span class="sk-h2-span">Licensing</span>

The CMIDSS dataset is available for non-commercial use under [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## <span class="sk-h2-span">Supported Tasks</span>

* [Sleep Detect](../tasks/detect.md)
* [Sleep Stage](../tasks/stage.md)
* [Sleep Apnea](../tasks/apnea.md)
* [Sleep Arousal](../tasks/arousal.md)

## <span class="sk-h2-span">Installation</span>

The CMIDSS dataset is available for download from Kaggle. Please note, a user account and permission to access the dataset is required. Once the requirements are met, the dataset can be downloaded using the `kaggle` package. Please note, the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables need to be set to the user's username and key. The dataset can be downloaded using the following command:

download from the [NSRR website](https://sleepdata.org/datasets/stages). Please note, a user account and permission to access the dataset is required. In addition, the [NSSR command line gem](https://github.com/nsrr/nsrr-gem) needs to be installed and exposed to the command line. Once both requirements are met, the dataset can be downloaded using the `sleepkit` package. Please note, the `NSSR_TOKEN` environment variable needs to be set to the user's token. The dataset can be downloaded using the following command:


```bash

export NSSR_TOKEN="INSERT_TOKEN_HERE"

sleepkit -m download \
         -c '{ "ds_path": "./datasets", "datasets": ["stages"], "progress": true }'
```
