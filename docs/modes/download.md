# Download Datasets

The `download` command is used to download all datasets specified. Please refer to [Datasets](../datasets/index.md) for details on the available datasets. Additional datasets can be added by creating a new dataset class and registering it with __SleepKit__ dataset factory.

## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will download and prepare four datasets.

    === "CLI"

        ```bash
        sleepkit -m download -c ./configs/download-datasets.json
        # ^ No task is required
        ```

    === "Python"

        ```python
        from pathlib import Path
        import sleepkit as sk

        sk.datasets.download_datasets(sk.SKDownloadParams(
            ds_path=Path("./datasets"),
            datasets=["ysyw", "cmidss", "mesa", "synthetic"],
            progress=True
        ))
        ```


## <span class="sk-h2-span">Arguments </span>

The following table lists the arguments that can be used with the `download` command. All datasets will be saved in their own subdirectory within the `ds_path` directory.

--8<-- "assets/modes/download-params.md"
