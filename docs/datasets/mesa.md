# MESA Dataset

## Overview

Multi-Ethnic Study of Atherosclerosis (MESA) is an NHLBI-sponsored 6-center collaborative longitudinal investigation of factors associated with the development of subclinical cardiovascular disease and the progression of subclinical to clinical cardiovascular disease in 6,814 black, white, Hispanic, and Chinese-American men and women initially ages 45-84 at baseline in 2000-2002. There have been four follow-up exams to date, in the years 2003-2004, 2004-2005, 2005-2007, and 2010-2011. Between 2010-2012, 2,237 participants also were enrolled in a Sleep Exam (MESA Sleep) which included full overnight unattended polysomnography, 7-day wrist-worn actigraphy, and a sleep questionnaire. The objectives of the sleep study are to understand how variations in sleep and sleep disorders vary across gender and ethnic groups and relate to measures of subclinical atherosclerosis.

More info available on [NSRR website](https://sleepdata.org/datasets/mesa)

## Funding

The Multi-Ethnic Study of Atherosclerosis (MESA) Sleep Ancillary study was funded by NIH-NHLBI Association of Sleep Disorders with Cardiovascular Health Across Ethnic Groups (RO1 HL098433). MESA is supported by NHLBI funded contracts HHSN268201500003I, N01-HC-95159, N01-HC-95160, N01-HC-95161, N01-HC-95162, N01-HC-95163, N01-HC-95164, N01-HC-95165, N01-HC-95166, N01-HC-95167, N01-HC-95168 and N01-HC-95169 from the National Heart, Lung, and Blood Institute, and by cooperative agreements UL1-TR-000040, UL1-TR-001079, and UL1-TR-001420 funded by NCATS. The National Sleep Research Resource was supported by the National Heart, Lung, and Blood Institute (R24 HL114473, 75N92019R002).

## Licensing

The MESA dataset is available for non-commercial and commercial use.

## Supported Tasks

* [Sleep Detect](../tasks/detect.md)
* [Sleep Stage](../tasks/stage.md)
* [Sleep Apnea](../tasks/apnea.md)
* [Sleep Arousal](../tasks/arousal.md)

## Installation

The MESA dataset is available for download from the [NSRR website](https://sleepdata.org/datasets/mesa). Please note, a user account and permission to access the dataset is required. Once granted permission, the dataset can be downloaded using the `sleepkit` package. Please note, the `NSRR_TOKEN` environment variable must be set to the user's token prior to downloading. The dataset can be downloaded using the following command:

```bash

export NSRR_TOKEN="INSERT_TOKEN_HERE"

sleepkit -m download \
         -c '{ "ds_path": "./datasets", "datasets": ["mesa"], "progress": true }'
```
