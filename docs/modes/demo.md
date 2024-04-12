# Task-Level Model Demo

## <span class="sk-h2-span">Introduction </span>

Each task in SleepKit has a corresponding demo mode that allows you to run a task-level demonstration using the specified backend inference engine (e.g. PC or EVB). This is useful to showcase the model's performance in real-time and to verify its accuracy in a real-world scenario. Similar to other modes, the demo can be invoked either via CLI or within `sleepkit` python package. At a high level, the demo mode performs the following actions based on the provided configuration parameters:

1. Load the configuration file (e.g. `stage-class-4`)
1. Load the desired dataset features (e.g. `cmidss`)
1. Load the trained model (e.g. `model.keras`)
1. Load random test subject's data
1. Perform inference via backend engine (e.g. PC or EVB)
1. Generate report

---

## <span class="sk-h2-span">Inference Backends</span>

SleepKit includes two built-in backend inference engines: PC and EVB. Additional backends can be easily added to the SleepKit framework by creating a new backend class and registering it to the backend factory.

### PC Backend

The PC backend is used to run the task-level demo on the local machine. This is useful for quick testing and debugging of the model.

1. Create / modify configuration file (e.g. `stage-class-4.json`)
1. Ensure "pc" is selected as the backend in configuration file.
1. Run demo `sleepkit --mode demo --task stage --config ./configs/stage-class-4.json`
1. HTML report will be saved to `${job_dir}/report.html`

### EVB Backend

The EVB backend is used to run the task-level demo on an Ambiq EVB. This is useful to showcase the model's performance in real-time and to verify its accuracy in a real-world scenario.

1. Create / modify configuration file (e.g. `stage-class-2.json`)
1. Ensure "evb" is selected as the backend in configuration file.
1. Plug EVB into PC via two USB-C cables.
1. Build and flash firmware to EVB `cd evb && make && make deploy`
1. Run demo `sleepkit --mode demo --task beat --config ./configs/stage-class-4.json`
1. HTML report will be saved to `${job_dir}/report.html`

### Bring-Your-Own-Backend

Similar to datasets, tasks, and models, the demo mode can be customized to use your own backend inference engine. SleepKit includes a backend factory (`BackendFactory`) that is used to create and run the backend engine.

#### How it Works

1. **Create a Backend**: Define a new backend by creating a new Python file. The file should contain a class that inherits from the `DemoBackend` base class and implements the required methods.

    ```python
    import sleepkit as sk

    class CustomBackend(sk.SKBackend):
        def __init__(self, config):
            super().__init__(config)

        def run(self, model, data):
            pass
    ```

2. **Register the Backend**: Register the new backend with the `BackendFactory` by calling the `register` method. This method takes the backend name and the backend class as arguments.

    ```python
    import sleepkit as sk
    sk.BackendFactory.register("custom", CustomBackend)
    ```

3. **Use the Backend**: The new backend can now be used by setting the `backend` flag in the demo configuration settings.

    ```python
    import sleepkit as sk
    task = sk.TaskFactory.get("detect")
    task.demo(sk.SKDemoParams(
        ...,
        backend="custom"
    ))
    ```
    _OR_ by creating the backend directly:

    ```python
    import sleepkit as sk
    backend = sk.BackendFactory.create("custom", config)
    ```

---

## <span class="sk-h2-span">Usage </span>

The following is an example of a task-level demo report for the sleep staging task.

=== "CLI"

    ```bash
    sleepkit -m export -t stage -c ./configs/stage-class-4.json
    ```

=== "Python"

    ```python
    from pathlib import Path
    import sleepkit as sk

    task = sk.TaskFactory.get("stage")
    task.export(sk.SKDemoParams(
        ...
    ))
    ```

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/stage/sleep-stage-demo.html"
</div>

---

## <span class="sk-h2-span">Arguments </span>

The following table lists the parameters that can be used to configure the demo mode. For argument `model_file`, the supported formats include `.keras` and `.tflite`.

--8<-- "assets/modes/demo-params.md"
