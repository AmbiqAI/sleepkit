# Sleep Detection

## <span class="sk-h2-span">Overview</span>

The objective of sleep detection is to identify periods of sustained sleep over the course of several days or weeks. The de facto standard for long-term, ambulatory sleep detection is actigraphy, which is a method of monitoring gross motor activity using an accelerometer. However, actigraphy is not a reliable method for sleep detection as it employs very simple heuristics to determine sleep. Often actigraphy can misclassify periods of inactivity or even not-worn as detected sleep.

<!-- <div class="sk-plotly-graph-div">
--8<-- "assets/sleep-detect-demo.html"
</div> -->

In this task, we look to leverage a light-weight model that can outperform actigraphy similarly using only data from an IMU. For more advanced sleep analysis, refer to the [Sleep Stage Classification](../stages/overview.md). By leveraring Ambiq's ultra-low-power microcontroller along with an ultra-low-power IMU, an efficient AI enabled actigraphy or fitness band will be able to run for weeks off a single charge.

<figure markdown>
  ![Wrist-based Sleep Classification](../assets/ambiq-watch.webp){ width="540" }
  <figcaption>Wrist-based Sleep Detection</figcaption>
</figure>
