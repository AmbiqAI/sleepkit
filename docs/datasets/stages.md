# STAGES Dataset

## <span class="sk-h2-span">Overview</span>

The Stanford Technology Analytics and Genomics in Sleep (STAGES) study is a prospective cross-sectional, multi-site study involving 20 data collection sites from six centers including Stanford University, Bogan Sleep Consulting, Geisinger Health, Mayo Clinic, MedSleep, and St. Luke's Hospital. The project has collected data on 1,500 adult/adolescent patients evaluated for sleep disorders, including:

* Objective nocturnal sleep polysomnography (PSG) recordings (EEGs, chin and leg EMGs, nasal and oral breathing, chest movements, leg movements, position, EKG). This is the gold standard for sleep evaluation in sleep clinics.
* Comprehensive subjective sleep symptoms assessment through an on-line sleep questionnaire called the Alliance Sleep Questionnaire (ASQ, developed over 5 years by 5 academic institutions). The questionnaire also includes key medical history questions.
* Continuous actigraphy over several weeks to get untransformed activity counts (a Huami/Xiaomi actigraph device has been identified and validated against PSG and clinical gold standard Actiwatch)
* 3-D facial scans to extract craniofacial features predictive of sleep apnea (not available on NSRR)
* On-line neuropsychological assessments and psychovigilance tests (impact of sleep disorders)
* Medical record data (not available on NSRR)

All data, samples, analytic tools, and supporting documentation will be made publicly available for use by any interested researcher, provided a request is submitted to the National Sleep Research Resource and approved.

The National Sleep Research Resource is grateful to the STAGES team for sharing these data.

More info available on [NSRR website](https://sleepdata.org/datasets/stages)

## <span class="sk-h2-span">Funding</span>

This research has been conducted using the STAGES - Stanford Technology, Analytics and Genomics in Sleep Resource funded by the Klarman Family Foundation. The investigators of the STAGES study contributed to the design and implementation of the STAGES cohort and/or provided data and/or collected biospecimens, but did not necessarily participate in the analysis or writing of this report. The full list of STAGES investigators can be found at the project website.

The National Sleep Research Resource was supported by the U.S. National Institutes of Health, National Heart Lung and Blood Institute (R24 HL114473, 75N92019R002).

## <span class="sk-h2-span">Licensing</span>

The STAGES dataset is available for non-commercial and commercial use.

## <span class="sk-h2-span">Supported Tasks</span>

* [Sleep Detect](../tasks/detect.md)
* [Sleep Stage](../tasks/stage.md)
* [Sleep Apnea](../tasks/apnea.md)
* [Sleep Arousal](../tasks/arousal.md)

## <span class="sk-h2-span">Installation</span>

The STAGES dataset is available for download from the [NSRR website](https://sleepdata.org/datasets/stages). Please note, a user account and permission to access the dataset is required. Once granted permission, the dataset can be downloaded using the `sleepkit` package. Please note, the `NSRR_TOKEN` environment variable must be set to the user's token prior to downloading. The dataset can be downloaded using the following command:

```bash

export NSRR_TOKEN="INSERT_TOKEN_HERE"

sleepkit -m download \
         -c '{ "ds_path": "./datasets", "datasets": ["stages"], "progress": true }'
```
