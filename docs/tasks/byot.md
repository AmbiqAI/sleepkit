# Bring-Your-Own-Task (BYOT)

The Bring-Your-Own-Task (BYOT) feature allows users to create custom tasks for training, evaluating, and deploying sleep-related AI models. This feature is useful for creating custom workflows for a given application with minimal coding.


## <span class="sk-h2-span">How it Works</span>

1. **Create a Task**: Define a new task by creating a new Python file. The file should contain a class that inherits from the `sk.Task` base class and implements the required methods.

    ```py linenums="1"
    import sleepkit as sk

    class CustomTask(sk.Task):

        @staticmethod
        def train(params: sk.TaskParams) -> None:
            pass

        @staticmethod
        def evaluate(params: sk.TaskParams) -> None:
            pass

        @staticmethod
        def export(params: sk.TaskParams) -> None:
            pass

        @staticmethod
        def demo(params: sk.TaskParams) -> None:
            pass

    ```

2. **Register the Task**: Register the new task with the `sk.TaskFactory` by calling the `register` method. This method takes the task name and the task class as arguments.

    ```py linenums="1"
    ...

    sk.TaskFactory.register("custom", CustomTask)
    ```

3. **Use the Task**: The new task can now be used with the `sk.TaskFactory` to perform various operations such as training, evaluating, and deploying models.

    ```py linenums="1"
    ...

    params = sk.TaskParams(...)
    task = sk.TaskFactory.get("custom")
    task.train(params)

    ```
