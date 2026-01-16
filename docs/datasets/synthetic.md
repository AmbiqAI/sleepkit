# Synthetic Data

### Overview

By leveraging [physioKIT](https://ambiqai.github.io/physiokit/), we are able to generate synthetic data for a variety of physiological signals, including ECG, PPG, and respiration. In addition to the signals, the tool also provides corresponding landmark fiducials and segmentation annotations. While not a replacement for real-world data, synthetic data can be useful in conjunction with real-world data for training and testing the models.

Please visit [physioKIT](https://ambiqai.github.io/physiokit/) for more details.


## Funding

NA

## Licensing

The tool is available under BSD-3-Clause License.

## Supported Tasks

* [Detect](../tasks/detect.md)



## Usage

!!! Example Python

    ```py linenums="1"
    import physiokit as pk

    heart_rate = 64 # BPM
    sample_rate = 1000 # Hz
    signal_length = 10*sample_rate # 10 seconds

    # Generate NSR synthetic ECG signal
    ecg, segs, fids = pk.ecg.synthesize(
        signal_length=signal_length,
        sample_rate=sample_rate,
        heart_rate=heart_rate,
        leads=1,
        preset=pk.ecg.EcgPreset.NSR,
        p_multiplier=1.5,
        t_multiplier=1.2,
        noise_multiplier=0.2
    )

    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/segmentation_example.html"
    </div>
