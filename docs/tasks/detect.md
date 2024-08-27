# :material-sleep: Sleep Detection Task

## <span class="sk-h2-span">Overview</span>

The objective of sleep detection is to identify periods of sustained sleep over the course of several days or weeks. The de facto standard for long-term, ambulatory sleep detection is actigraphy, which is a method of monitoring gross motor activity using an accelerometer. However, actigraphy is not a reliable method for sleep detection as it employs very simple heuristics to determine sleep. Often actigraphy can misclassify periods of inactivity or even not-worn as detected sleep.

<!-- <div class="sk-plotly-graph-div">
--8<-- "assets/sleep-detect-demo.html"
</div> -->

In this task, we look to leverage a light-weight model that can outperform actigraphy similarly using only data from an IMU. For more advanced sleep analysis, refer to the [Sleep Stage Classification](./stage.md). By leveraring Ambiq's ultra-low-power microcontroller along with an ultra-low-power IMU, an efficient AI enabled actigraphy or fitness band will be able to run for weeks off a single charge.

<figure markdown>
  ![Wrist-based Sleep Classification](../assets/tasks/detect/ambiq-watch.webp){ width="540" }
  <figcaption>Wrist-based Sleep Detection</figcaption>
</figure>

---

## <span class="sk-h2-span">Model Zoo</span>

The following table provides the latest performance and accuracy results for pre-trained models. Additional result details can be found in [Model Zoo → Detect](../zoo/detect.md).


--8<-- "assets/zoo/detect/detect-model-zoo-table.md"

---

## <span class="sk-h2-span">Target Classes</span>

Below outlines the classes available for sleep detect classification. When training a model, the number of classes, mapping, and names must be provided.

--8<-- "assets/tasks/detect/detect-default-class-table.md"

!!! example "Class Mapping"

    Below is an example of a class mapping for a 2-class sleep detect model. The class map keys are the original class labels and the values are the new class labels. Any class not included will be skipped.

    ```json
    {
        "num_classes": 2,
        "class_names": ["AWAKE", "SLEEP"],
        "class_map": {
            "0": 0,  // Map AWAKE to AWAKE
            "1": 1   // Map SLEEP to SLEEP
        }
    }
    ```

---

## <span class="sk-h2-span">References</span>

* [AI-Driven sleep staging from actigraphy and heart rate](https://doi.org/10.1371/journal.pone.0285703)
