import urllib.parse
import webbrowser
from typing import ClassVar

from markdown_it import MarkdownIt
from markdown_pdf import MarkdownPdf, Section

from deep_research_agent.services.export.base import ExportStrategy


class MarkdownExporter(ExportStrategy):
    @property
    def requires_path(self) -> bool:
        return True

    @property
    def category(self) -> str:
        return "local_file"

    def export(self, markdown_content: str, output_path: str | None = None) -> str:
        if output_path is None:
            raise ValueError("output_path is required for MarkdownExporter")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return output_path


class PdfExporter(ExportStrategy):
    @property
    def requires_path(self) -> bool:
        return True

    @property
    def category(self) -> str:
        return "local_file"

    def export(self, markdown_content: str, output_path: str | None = None) -> str:
        if output_path is None:
            raise ValueError("output_path is required for PdfExporter")
        pdf = MarkdownPdf(mode="gfm-like")
        pdf.add_section(Section(markdown_content))
        pdf.save(output_path)
        return output_path


class HtmlExporter(ExportStrategy):
    @property
    def requires_path(self) -> bool:
        return True

    @property
    def category(self) -> str:
        return "local_file"

    def export(self, markdown_content: str, output_path: str | None = None) -> str:
        if output_path is None:
            raise ValueError("output_path is required for HtmlExporter")
        md = MarkdownIt(config="gfm-like")
        compiled_html = md.render(markdown_content)
        standalone_html_template = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Embedded Report</title>
            <!-- Tailwind Play CDN with the official Typography plugin enabled -->
            <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
        </head>
        <body class="bg-slate-50 min-h-screen py-12 px-4 sm:px-6 lg:px-8">

            <!-- Container wrapper limiting document width and centering layout -->
            <main class="max-w-6xl mx-auto bg-white p-8 sm:p-12 rounded-xl shadow-sm border border-slate-200">

                <!-- The `.prose` class safely captures all standard headers, tables, and lists -->
                <article class="prose max-w-none prose-slate prose-table:border-collapse prose-th:bg-slate-100 prose-th:p-3 prose-td:p-3 prose-tr:border-b">
                    {compiled_html}
                </article>

            </main>

        </body>
        </html>
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(standalone_html_template)
        return output_path


class AIChatExporter(ExportStrategy):
    @property
    def requires_path(self) -> bool:
        return False

    @property
    def category(self) -> str:
        return "ai_chat"

    #: Deep links that pre-fill the prompt composer; ``{prompt}`` is replaced
    #: with a percent-encoded payload, so it must sit in the query string.
    PLATFORMS: ClassVar[dict[str, str]] = {
        "chatgpt": "https://chat.openai.com/?q={prompt}",
        "claude": "https://claude.ai/new?q={prompt}",
    }

    #: Browsers stop navigating past roughly 32k URL characters (Chrome is the
    #: strictest common case), and the report travels inside the query string,
    #: so refuse up front rather than opening a tab that silently fails.
    MAX_URL_LENGTH: ClassVar[int] = 20_000

    @classmethod
    def get_available_platforms(cls) -> list[str]:
        return list(cls.PLATFORMS.keys())

    def __init__(self, platform: str):
        self.platform = platform.lower().strip()
        if self.platform not in self.PLATFORMS:
            raise ValueError(f"Unsupported platform: {self.platform}")

    def export(self, markdown_content: str, output_path: str | None = None) -> str:
        prompt_payload = f"please read the following report:\n\n {markdown_content}"
        encoded_prompt = urllib.parse.quote(prompt_payload)
        target_url = self.PLATFORMS[self.platform].format(prompt=encoded_prompt)
        if len(target_url) > self.MAX_URL_LENGTH:
            raise ValueError(
                f"This report is too large to hand to {self.platform} as a link "
                f"({len(target_url):,} characters, limit {self.MAX_URL_LENGTH:,}). "
                "Export it to Markdown and paste the report into the chat instead."
            )
        if not webbrowser.open(url=target_url, new=2):
            raise RuntimeError(
                f"Could not open a web browser for {self.platform}: {target_url}"
            )
        return target_url
