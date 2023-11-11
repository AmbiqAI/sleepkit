# Sleep Stage Demo

A [jupyter notebook](https://github.com/AmbiqAI/sleepkit/tree/main/notebooks/sleep-stage-demo.ipynb) is provided to showcase the capabilities of the sleep stage classifier models. Based on the provided configuration parameters, the demo performs the following actions:

1. Loads the dataset (e.g. `fs001`)
1. Loads the trained model (e.g. `sleep-stage-4`)
1. Loads random test subject's data
1. Perform inference either on PC or EVB
1. Plot results

---

## <span class="sk-h2-span">Usage</span>

1. Load notebook in Jupyter
1. Update configuration parameters at top
1. If EVB is selected, connect EVB to PC via two USB-C cables
1. Run all cells

---

## <span class="sk-h2-span">Outputs</span>

<div class="sk-plotly-graph-div">
--8<-- "assets/sleep-stage-demo-example.html"
</div>

<div class="sk-plotly-graph-div">
--8<-- "assets/demo-sleep-cycle-pie.html"
</div>

---
