from .task import SKTask

_tasks: dict[str, type[SKTask]] = {}


class TaskFactory:
    """Task factory enables registering, creating, and listing tasks. It is a singleton class."""

    @staticmethod
    def register(name: str, task: type[SKTask]) -> None:
        """Register a task

        Args:
            name (str): task name
            task (type[SKTask]): task
        """
        _tasks[name] = task

    @staticmethod
    def unregister(name: str) -> None:
        """Unregister a task

        Args:
            name (str): task name
        """
        del _tasks[name]

    @staticmethod
    def create(name: str, **kwargs) -> SKTask:
        """Create a task

        Args:
            name (str): task name

        Returns:
            SKTask: task
        """
        return _tasks[name](**kwargs)

    @staticmethod
    def list() -> list[str]:
        """List registered tasks

        Returns:
            list[str]: task names
        """
        return list(_tasks.keys())

    @staticmethod
    def get(name: str) -> type[SKTask]:
        """Get a task

        Args:
            name (str): task name

        Returns:
            Type[SKTask]: task
        """
        return _tasks[name]

    @staticmethod
    def has(name: str) -> bool:
        """Check if a task is registered

        Args:
            name (str): task name

        Returns:
            bool: True if registered
        """
        return name in _tasks
