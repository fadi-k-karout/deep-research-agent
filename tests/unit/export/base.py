import tempfile
import unittest
from abc import ABC, abstractmethod
from pathlib import Path

from deep_research_agent.services.export.base import ExportStrategy


class BaseFileExporterUnitTest(unittest.TestCase, ABC):
    """Shared contract for every strategy that writes an artifact to disk."""

    @property
    @abstractmethod
    def exporter_class(self) -> type[ExportStrategy]:
        """The concrete exporter class under test."""

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension the exporter writes."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_path = str(Path(self.temp_dir.name) / f"report.{self.extension}")
        self.exporter = self.exporter_class()

    def artifact_size(self) -> int:
        return Path(self.output_path).stat().st_size

    def artifact_bytes(self) -> bytes:
        return Path(self.output_path).read_bytes()

    def test_requires_path_reports_true(self):
        self.assertTrue(self.exporter.requires_path)

    def test_category_is_local_file(self):
        self.assertEqual(self.exporter.category, "local_file")

    def test_export_writes_a_non_empty_file_at_the_requested_path(self):
        self.exporter.export("# Report\n\nBody", self.output_path)

        self.assertTrue(Path(self.output_path).is_file())
        self.assertGreater(self.artifact_size(), 0)

    def test_export_returns_the_requested_path(self):
        result = self.exporter.export("# Report\n\nBody", self.output_path)

        self.assertEqual(result, self.output_path)

    def test_export_with_explicit_none_output_path_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.exporter.export("# Report", None)

        self.assertIn("output_path is required", str(ctx.exception))

    def test_export_without_output_path_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.exporter.export("# Report")

        self.assertIn("output_path is required", str(ctx.exception))
        self.assertIn(self.exporter_class.__name__, str(ctx.exception))
