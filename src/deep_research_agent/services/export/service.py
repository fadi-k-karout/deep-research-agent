import asyncio
import os
from pathlib import Path

from deep_research_agent.services.export.base import ExportStrategy
from deep_research_agent.services.export.exporters import (
    AIChatExporter,
    HtmlExporter,
    MarkdownExporter,
    PdfExporter,
)


def _normalize_format(fmt: str) -> str:
    return fmt.lower().strip().lstrip(".")


def _safe_basename(base_filename: str) -> str:
    """Reduce *base_filename* to a single path component inside the target dir."""
    name = Path(base_filename).name.strip().lstrip(".")
    if not name:
        raise ValueError(f"Invalid base_filename: {base_filename!r}")
    return name


class ExportService:
    def __init__(self):
        self._exporters: dict[str, ExportStrategy] = {
            "md": MarkdownExporter(),
            "pdf": PdfExporter(),
            "html": HtmlExporter(),
        }

        for app in AIChatExporter.get_available_platforms():
            self._exporters[app] = AIChatExporter(app)

    async def export_report(
        self,
        markdown_content: str,
        base_filename: str,
        formats: str | list[str],
        output_directory: str = ".",
    ) -> list[str]:
        if isinstance(formats, str):
            formats = [formats]

        normalized = [_normalize_format(fmt) for fmt in formats]

        unknown = [fmt for fmt in normalized if fmt not in self._exporters]
        if unknown:
            supported = ", ".join(sorted(self._exporters))
            raise ValueError(
                f"Unsupported export format(s): {', '.join(unknown)}. "
                f"Supported formats: {supported}"
            )

        safe_basename = _safe_basename(base_filename)
        saved_paths = []

        for fmt in normalized:
            exporter = self._exporters[fmt]

            # The service asks the strategy directly if it needs a path!
            path_argument = (
                os.path.join(output_directory, f"{safe_basename}.{fmt}")
                if exporter.requires_path
                else None
            )

            result_str = await asyncio.to_thread(
                exporter.export, markdown_content, path_argument
            )
            saved_paths.append(result_str)

        return saved_paths

    def get_available_formats(self) -> dict[str, list[str]]:
        """Dynamically groups formats for your Textual TUI using the strategy's self-declared category."""
        categories = {}
        for fmt, exporter in self._exporters.items():
            cat = exporter.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(fmt)
        return categories
