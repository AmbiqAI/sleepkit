# Bring-Your-Own-Feature-Set (BYOFS)

The Bring-Your-Own-Feature-Set (BYOFS) allows users to add custom feature set generators to the feature store.

## How it Works

1. **Create a Feature Set Generator**: Define a new feature set generator by creating a new Python file. The file should contain a class that inherits from the `SKFeatureSet` base class and implements the required methods.

    ```python
    import sleepkit as sk

    class CustomFeatureSet(sk.SKFeatureSet):
        @staticmethod
        def name() -> str:
            return "custom"

        @staticmethod
        def feature_names() -> list[str]:
            return ["feature1", "feature2", "feature3"]

        @staticmethod
        def generate_features(ds_subject: tuple[str, str], args: SKFeatureParams):
            pass
    ```

2. **Register the Feature Set Generator**: Register the new feature set generator with the feature store by calling the `register` method. This method takes the feature set name and the feature set class as arguments.

    ```python
    import sleepkit as sk
    sk.FeatureStore.register(CustomFeatureSet.name, CustomFeatureSet)
    ```
3. **Use the Feature Set Generator**: The new feature set generator can now be used with the `FeatureStore` to generate feature sets. Note:` generate_feature_set` will utilize a thread pool to generate features in parallel.

    ```python
    import sleepkit as sk

    # Generate feature set
    sk.generate_feature_set(args=sk.SKFeatureParams(
        ...
    ))
    ```
