import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class Report:
    id: int
    topic: str
    content: str
    created_at: str


class ResearchAgentStorage:
    def __init__(self, db_path: str = "deep_research_agent.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        try:
            self._init_db()
        except Exception:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """Internal helper to reuse the connection across queries."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            # This allows fetching rows as dictionaries: row["content"] instead of row[2]
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Creates the reports table if it doesn't already exist."""
        query = """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
        conn = self._get_connection()
        with conn:  # Context manager automatically commits the transaction
            conn.execute(query)

    def save_report(self, topic: str, markdown_content: str) -> int:
        """Saves a new research report and returns its unique ID.
        Args:
           topic (str): The topic of the report.
           markdown_content (str): The report content in markdown format.
        Raises:
           ValueError: If the report could not be saved.
        Returns:
           int: The unique ID of the saved report.
        """
        query = """
        INSERT INTO reports (topic, content, created_at)
        VALUES (?, ?, ?);
        """
        timestamp = datetime.now(UTC).isoformat()
        conn = sqlite3.connect(self.db_path)
        # Use a call-local connection (instead of the cached one) so this
        # can safely run in a worker thread off the event loop.
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                cursor = conn.execute(query, (topic, markdown_content, timestamp))
                if cursor.lastrowid is None:
                    raise ValueError("Failed to save report")
                return cursor.lastrowid
        finally:
            conn.close()

    def get_all_reports(self) -> list[Report]:
        """Fetches all reports.
        Returns:
           list[Report]: A list of all reports.
        """
        query = """
        SELECT id, topic, content, created_at
        FROM reports
        ORDER BY created_at DESC
        """
        conn = self._get_connection()
        cursor = conn.execute(query)
        rows = cursor.fetchall()

        return [Report(**dict(row)) for row in rows]

    def close(self) -> None:
        """Safely closes the connection when the agent run is finished."""
        if self._conn:
            self._conn.close()
            self._conn = None
