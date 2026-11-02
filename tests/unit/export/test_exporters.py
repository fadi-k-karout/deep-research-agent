import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

from deep_research_agent.services.export.exporters import (
    AIChatExporter,
    HtmlExporter,
    MarkdownExporter,
    PdfExporter,
)
from tests.unit.export.base import BaseFileExporterUnitTest

UNICODE_MARKUP_SAMPLE = "# Ünicode ✨\n\n[bold]raw[/] & <tag>text</tag>"


class TestMarkdownExporter(BaseFileExporterUnitTest):
    exporter_class = MarkdownExporter
    extension = "md"

    def test_export_writes_content_verbatim(self):
        content = "# Title\n\nSome **bold** text and a list:\n\n- one\n- two\n"

        self.exporter.export(content, self.output_path)

        self.assertEqual(Path(self.output_path).read_text(encoding="utf-8"), content)

    def test_export_preserves_unicode_and_raw_markup(self):
        self.exporter.export(UNICODE_MARKUP_SAMPLE, self.output_path)

        self.assertEqual(
            Path(self.output_path).read_text(encoding="utf-8"),
            UNICODE_MARKUP_SAMPLE,
        )

    def test_export_truncates_rather_than_appends_on_reuse(self):
        self.exporter.export("# First\n", self.output_path)
        first_size = self.artifact_size()

        self.exporter.export("# Second\n", self.output_path)

        self.assertLess(self.artifact_size(), first_size + 1024)
        self.assertNotIn("First", Path(self.output_path).read_text(encoding="utf-8"))


class TestPdfExporter(BaseFileExporterUnitTest):
    exporter_class = PdfExporter
    extension = "pdf"

    def test_export_produces_a_pdf_document(self):
        self.exporter.export("# Report\n\nBody", self.output_path)

        self.assertTrue(
            self.artifact_bytes().startswith(b"%PDF"),
            "expected the exported artifact to be a PDF document",
        )


class TestHtmlExporter(BaseFileExporterUnitTest):
    exporter_class = HtmlExporter
    extension = "html"

    def read_html(self) -> str:
        return Path(self.output_path).read_text(encoding="utf-8")

    def test_export_wraps_rendered_markdown_in_a_standalone_document(self):
        self.exporter.export("# Report\n\nSome **bold** text", self.output_path)

        html = self.read_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<h1>Report</h1>", html)
        self.assertIn("<strong>bold</strong>", html)
        self.assertIn("</html>", html)

    def test_export_does_not_leak_unrendered_markdown(self):
        self.exporter.export("# Report\n\n- item", self.output_path)

        self.assertNotIn("# Report", self.read_html())

    def test_export_preserves_unicode_characters(self):
        self.exporter.export(UNICODE_MARKUP_SAMPLE, self.output_path)

        self.assertIn("Ünicode ✨", self.read_html())

    def test_export_truncates_rather_than_appends_on_reuse(self):
        self.exporter.export("# First\n", self.output_path)
        first_size = self.artifact_size()

        self.exporter.export("# Second\n", self.output_path)

        self.assertLess(self.artifact_size(), first_size + 1024)
        self.assertNotIn("First", self.read_html())


class TestAIChatExporter(unittest.TestCase):
    def setUp(self):
        patcher = patch("deep_research_agent.services.export.exporters.webbrowser.open")
        self.browser_open = patcher.start()
        self.addCleanup(patcher.stop)
        self.browser_open.return_value = True

    def test_requires_path_reports_false(self):
        self.assertFalse(AIChatExporter("chatgpt").requires_path)

    def test_category_is_ai_chat(self):
        self.assertEqual(AIChatExporter("chatgpt").category, "ai_chat")

    def test_get_available_platforms_lists_every_supported_platform(self):
        self.assertEqual(
            AIChatExporter.get_available_platforms(), ["chatgpt", "claude"]
        )

    def test_platform_is_normalized_to_lowercase_and_trimmed(self):
        for raw in ("ChatGPT", "CHATGPT", " chatgpt ", "claude", " Claude\n"):
            with self.subTest(platform=raw):
                self.assertEqual(AIChatExporter(raw).platform, raw.lower().strip())

    def test_unsupported_platform_is_rejected(self):
        for raw in ("gemini", "", "chat gpt"):
            with self.subTest(platform=raw):
                with self.assertRaises(ValueError) as ctx:
                    AIChatExporter(raw)
                self.assertIn("Unsupported platform", str(ctx.exception))

    def test_export_returns_url_and_opens_browser_with_it(self):
        for platform, host in (
            ("chatgpt", "https://chat.openai.com"),
            ("claude", "https://claude.ai"),
        ):
            with self.subTest(platform=platform):
                self.browser_open.reset_mock()

                url = AIChatExporter(platform).export("# Report")

                self.assertTrue(url.startswith(host))
                self.browser_open.assert_called_once_with(url=url, new=2)

    def test_export_percent_encodes_the_prompt_payload(self):
        url = AIChatExporter("chatgpt").export("secret #1 & more")

        self.assertNotIn(" ", url)
        self.assertIn(urllib.parse.quote("secret #1 & more"), url)
        self.assertIn("please%20read%20the%20following%20report", url.lower())

    def test_export_ignores_output_path_and_writes_no_file(self):
        target = str(Path(tempfile.gettempdir()) / "should-never-be-written.md")

        url = AIChatExporter("chatgpt").export("# Report", target)

        self.assertTrue(url.startswith("https://"))
        self.assertFalse(Path(target).exists())

    def test_export_surfaces_failure_to_open_browser(self):
        self.browser_open.return_value = False

        with self.assertRaises(RuntimeError) as ctx:
            AIChatExporter("chatgpt").export("# Report")

        self.assertIn("browser", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
