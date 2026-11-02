import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from deep_research_agent.services.export.service import ExportService

CONTENT = "# Report\n\nBody with **bold** text and Ünicode ✨"


class _SlowExporter:
    requires_path = False
    category = "test_slow"

    def __init__(self, delay: float):
        self.delay = delay

    def export(self, markdown_content: str, output_path: str | None = None) -> str:
        time.sleep(self.delay)
        return "slow-result"


class TestExportService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.sandbox_root = Path(temp_dir.name)
        self.output_dir = str(self.sandbox_root / "output")
        os.makedirs(self.output_dir)

        self.service = ExportService()

        patcher = patch(
            "deep_research_agent.services.export.exporters.webbrowser.open",
            return_value=True,
        )
        self.browser_open = patcher.start()
        self.addCleanup(patcher.stop)

    def artifacts(self) -> list[str]:
        return sorted(p.name for p in Path(self.output_dir).iterdir())

    async def export(self, formats, base_filename="report", output_dir=None):
        return await self.service.export_report(
            CONTENT,
            base_filename,
            formats,
            self.output_dir if output_dir is None else output_dir,
        )

    def test_get_available_formats_groups_strategies_by_declared_category(self):
        self.assertEqual(
            self.service.get_available_formats(),
            {
                "local_file": ["md", "pdf", "html"],
                "ai_chat": ["chatgpt", "claude"],
            },
        )

    def test_get_available_formats_agrees_with_each_exporter_category(self):
        for category, formats in self.service.get_available_formats().items():
            for fmt in formats:
                with self.subTest(format=fmt):
                    self.assertEqual(self.service._exporters[fmt].category, category)

    async def test_export_report_writes_every_requested_local_file_format(self):
        returned = await self.export(["md", "html", "pdf"])

        self.assertEqual(
            [Path(p).name for p in returned],
            ["report.md", "report.html", "report.pdf"],
        )
        for path in returned:
            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 0)

    async def test_export_report_returns_results_in_requested_order(self):
        returned = await self.export(["html", "md", "claude"])

        self.assertEqual(
            [Path(p).name for p in returned[:2]],
            ["report.html", "report.md"],
        )
        self.assertTrue(returned[2].startswith("https://claude.ai"))

    async def test_export_report_accepts_a_single_format_as_a_string(self):
        returned = await self.export("md")

        self.assertEqual([Path(p).name for p in returned], ["report.md"])

    async def test_export_report_normalizes_format_case_and_leading_dot(self):
        for index, fmt in enumerate(("md", "MD", ".md", ".MD", ".html")):
            with self.subTest(format=fmt):
                returned = await self.export(fmt, base_filename=f"report{index}")

                self.assertEqual(len(returned), 1)
                self.assertTrue(Path(returned[0]).is_file())

    async def test_export_report_trims_surrounding_whitespace_from_format(self):
        for fmt in (" md", "md ", " .md ", "\tmd\n"):
            with self.subTest(format=fmt):
                returned = await self.export(fmt)

                self.assertEqual(len(returned), 1)
                self.assertTrue(Path(returned[0]).is_file())

    async def test_export_report_returns_empty_list_for_empty_format_list(self):
        self.assertEqual(await self.export([]), [])
        self.assertEqual(self.artifacts(), [])

    async def test_export_report_returns_url_and_writes_no_file_for_ai_chat(self):
        returned = await self.export("chatgpt")

        self.assertEqual(len(returned), 1)
        self.assertTrue(returned[0].startswith("https://chat.openai.com"))
        self.browser_open.assert_called_once_with(url=returned[0], new=2)
        self.assertEqual(self.artifacts(), [])

    async def test_export_report_rejects_unknown_format(self):
        with self.assertRaises(ValueError) as ctx:
            await self.export("docx")

        message = str(ctx.exception)
        self.assertIn("docx", message)
        self.assertIn("md", message)

    async def test_export_report_rejects_unknown_format_within_a_batch(self):
        with self.assertRaises(ValueError):
            await self.export(["md", "docx"])

    async def test_export_report_writes_nothing_when_any_format_is_unknown(self):
        with self.assertRaises(ValueError):
            await self.export(["md", "html", "docx"])

        self.assertEqual(self.artifacts(), [])

    async def test_export_report_rejects_base_filename_without_a_usable_name(self):
        for base_filename in ("", "   ", ".", "..", "../", "/", "a/../.."):
            with self.subTest(base_filename=base_filename):
                with self.assertRaises(ValueError) as ctx:
                    await self.export("md", base_filename=base_filename)

                self.assertIn("Invalid base_filename", str(ctx.exception))

    async def test_export_report_confines_artifact_to_the_output_directory(self):
        escaping_filenames = (
            "../escaped",
            str(self.sandbox_root / "absolute_escaped"),
        )

        for base_filename in escaping_filenames:
            with self.subTest(base_filename=base_filename):
                returned = await self.export("md", base_filename=base_filename)

                self.assertEqual(len(returned), 1)
                self.assertTrue(
                    Path(returned[0])
                    .resolve()
                    .is_relative_to(Path(self.output_dir).resolve()),
                    f"export escaped the output directory: {returned[0]}",
                )

    async def test_export_report_keeps_the_final_path_component_of_base_filename(self):
        cases = {
            "../escaped": "escaped.md",
            "nested/dir/report": "report.md",
            "./report": "report.md",
        }

        for base_filename, expected_name in cases.items():
            with self.subTest(base_filename=base_filename):
                returned = await self.export("md", base_filename=base_filename)

                self.assertEqual([Path(p).name for p in returned], [expected_name])

    async def test_export_report_propagates_error_for_missing_output_directory(self):
        missing = str(self.sandbox_root / "does-not-exist")

        with self.assertRaises(FileNotFoundError):
            await self.export("md", output_dir=missing)

    async def test_export_report_leaves_the_event_loop_responsive(self):
        ticks: list[int] = []

        async def ticker():
            for _ in range(5):
                await asyncio.sleep(0.01)
                ticks.append(1)

        self.service._exporters["slow"] = _SlowExporter(0.2)
        tick_task = asyncio.create_task(ticker())

        returned = await self.export("slow")

        self.assertEqual(returned, ["slow-result"])
        self.assertTrue(ticks, "event loop was blocked while the export was in flight")
        tick_task.cancel()


if __name__ == "__main__":
    unittest.main()
