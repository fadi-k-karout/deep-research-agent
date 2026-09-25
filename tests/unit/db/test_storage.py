import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from deep_research_agent.db.storage import ResearchAgentStorage


class TestResearchAgentStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "reports.db")
        self.storage = ResearchAgentStorage(self.db_path)

    def tearDown(self):
        self.storage.close()
        self.temp_dir.cleanup()

    def test_get_all_reports_returns_empty_list_for_new_database(self):
        self.assertEqual(self.storage.get_all_reports(), [])

    def test_save_then_get_all_reports_round_trip(self):
        report_id = self.storage.save_report("Research x", "# Report\n\nBody")

        reports = self.storage.get_all_reports()

        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual(report.id, report_id)
        self.assertEqual(report.topic, "Research x")
        self.assertEqual(report.content, "# Report\n\nBody")
        self.assertEqual(
            datetime.fromisoformat(report.created_at).tzinfo,
            UTC,
        )

    def test_get_all_reports_orders_newest_first(self):
        timestamps = [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        ]
        with patch("deep_research_agent.db.storage.datetime") as mock_datetime:
            mock_datetime.now.side_effect = timestamps
            self.storage.save_report("older", "older body")
            self.storage.save_report("newer", "newer body")

        reports = self.storage.get_all_reports()

        self.assertEqual([report.topic for report in reports], ["newer", "older"])

    def test_close_allows_reopening_for_queries(self):
        self.storage.save_report("Research x", "Body")
        self.storage.close()
        self.storage.close()

        reports = self.storage.get_all_reports()

        self.assertEqual([report.topic for report in reports], ["Research x"])

    def test_report_content_preserves_markup_and_unicode(self):
        content = "[bold]Ünicode ✨[/]"

        self.storage.save_report("Topic [raw]", content)

        report = self.storage.get_all_reports()[0]
        self.assertEqual(report.topic, "Topic [raw]")
        self.assertEqual(report.content, content)


if __name__ == "__main__":
    unittest.main()
