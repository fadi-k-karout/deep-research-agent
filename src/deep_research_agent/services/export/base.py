from abc import ABC, abstractmethod


class ExportStrategy(ABC):
    @property
    @abstractmethod
    def requires_path(self) -> bool:
        """Returns True if this strategy saves a file to the local disk."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Returns the UI grouping category (e.g., 'local_file', 'ai_chat', 'webhook')."""

    @abstractmethod
    def export(self, markdown_content: str, output_path: str | None = None) -> str:
        """Executes the export operation and returns a target path or URL."""
