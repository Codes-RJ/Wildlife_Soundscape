"""Migrations operations behind the EventDatabase facade."""

from __future__ import annotations
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
DEFAULT_ANALYTICS_SAMPLE_RATE = 48000
DEFAULT_ANALYTICS_BUCKET_SECONDS = 3600
UTC_SUFFIX = "Z"
UINT8_MAX = 255
UINT32_MAX = 4294967295
UINT64_MAX = 18446744073709551615
SQLITE_INT64_MAX = 9223372036854775807


def _ensure_column(
    conn: sqlite3.Connection,
    *,
    table: str,
    column: str,
    definition: str,
) -> None:
    """
    Add one hard-coded internal column when absent.
    """

    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()

    existing_columns = {str(row[1]) for row in rows}

    if column in existing_columns:
        return

    conn.execute(
        f"""
        ALTER TABLE {table}
        ADD COLUMN {column} {definition}
        """
    )


def _ensure_nullable_second_confidence(
    conn: sqlite3.Connection,
) -> None:
    """
    Upgrade old classifications tables where second_confidence was
    incorrectly declared NOT NULL.

    SQLite cannot directly DROP a NOT NULL constraint, therefore the
    table is rebuilt while preserving existing rows.
    """

    table_info = conn.execute(
        """
            PRAGMA table_info(
                classifications
            )
            """
    ).fetchall()

    second_column = next(
        (row for row in table_info if str(row[1]) == "second_confidence"),
        None,
    )

    # Fresh/current schema.
    if second_column is None or int(second_column[3]) == 0:
        return

    # ==============================================================
    # OLD TABLE REQUIRES REBUILD
    # ==============================================================

    conn.execute(
        """
        CREATE TABLE
            classifications_migrated (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                event_id INTEGER NOT NULL UNIQUE,

                label TEXT NOT NULL,

                confidence REAL NOT NULL,

                second_label TEXT,

                second_confidence REAL,

                margin REAL NOT NULL,

                scores_json TEXT NOT NULL,

                reasons_json TEXT NOT NULL,

                classifier_name TEXT NOT NULL,

                classifier_version TEXT NOT NULL,

                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(event_id)
                    REFERENCES events(id)
                    ON DELETE CASCADE
            )
        """
    )

    conn.execute(
        """
        INSERT INTO classifications_migrated(

            id,

            event_id,

            label,

            confidence,

            second_label,

            second_confidence,

            margin,

            scores_json,

            reasons_json,

            classifier_name,

            classifier_version,

            created_at
        )

        SELECT

            id,

            event_id,

            label,

            confidence,

            second_label,

            second_confidence,

            margin,

            scores_json,

            reasons_json,

            classifier_name,

            classifier_version,

            created_at

        FROM classifications
        """
    )

    conn.execute(
        """
        DROP TABLE classifications
        """
    )

    conn.execute(
        """
        ALTER TABLE classifications_migrated
        RENAME TO classifications
        """
    )
