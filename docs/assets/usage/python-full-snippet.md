```python
import sleepkit as sk

with open("feature-config.json", "r", encoding="utf-8") as file:
    feat_config = json.load(file)

# Generate feature set
sk.generate_feature_set(sk.SKFeatureParams.model_validate(feat_config))

with open("task-config.json", "r", encoding="utf-8") as file:
    task_config = json.load(file)

# Grab stage task
task = sk.TaskFactory.get("stage")

# Train model
task.train(sk.SKTrainParams.model_validate(task_config))

# Evaluate model
task.evaluate(sk.SKTestParams.model_validate(task_config))

# Export model
task.export(sk.SKExportParams.model_validate(task_config))

```
