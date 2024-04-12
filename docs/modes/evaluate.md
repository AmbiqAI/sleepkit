# Model Evaluation

## <span class="sk-h2-span">Introduction </span>

Evaluate mode is used to test the performance of the model on the reserved test set for the specified task. Similar to training, the routine can be customized via CLI configuration file or by setting the parameters directly in the code. The evaluation process involves testing the model's performance on the test data to measure its accuracy, precision, recall, and F1 score. A number of results and metrics will be generated and saved to the `job_dir`.

---

## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will evaluate the sleep detect model using the reference configuration:

    === "CLI"

        ```bash
        sleepkit --mode evaluate --task detect --config ./configs/detect-class-2.json
        ```

    === "Python"

        ```python
        from pathlib import Path
        import sleepkit as sk

        task = sk.TaskFactory.get("detect")
        task.evaluate(sk.HKTestParams(
            ...
        ))
        ```

---

## <span class="sk-h2-span">Arguments </span>

The following tables lists the arguments that can be used with the `evaluate` command.

--8<-- "assets/modes/evaluate-params.md"
