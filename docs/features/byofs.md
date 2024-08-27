# Bring-Your-Own-Features (BYOFS)

The Bring-Your-Own-Features (BYOFS) allows users to add custom feature sets to SleepKit to be used with built-in or custom tasks.

## How it Works

1. **Create a Feature Set**: Define a new feature set class that subclasses `sk.FeatureSet` and implements all abstract methods.

    ```py linenums="1"
    import sleepkit as sk

    class CustomFeatureSet(sk.FeatureSet):
        @staticmethod
        def name() -> str:
            return "custom"

        @staticmethod
        def feature_names() -> list[str]:
            return ["feature1", "feature2", "feature3"]

        @staticmethod
        def generate_subject_features(subject_id: str, ds_name: str, params: sk.TaskParams):
            pass
    ```

2. **Register the Feature Set**: Register the new feature set with the `sk.FeatureFactory` by calling the `register` method. This method takes the feature set name and the feature set class as arguments.

    ```py linenums="1"
    import sleepkit as sk
    sk.FeatureFactory.register(CustomFeatureSet.name, CustomFeatureSet)
    ```

3. **Use the Feature Set**: The new feature set can now be used to generate feature sets.

    ```py linenums="1"
    import sleepkit as sk

    # Create a task params object
    params = sk.TaskParams(
        ...
        feature=sk.FeatureParams(
            name="custom",
            ...
        )
    )

    # Load a task
    task = sk.TaskFactory.get("stage")

    # Use the custom feature set
    task.feature(params)

    ```
