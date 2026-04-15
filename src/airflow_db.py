"""
Database utilities for Airflow data pipeline tasks.
Handles PostgreSQL connections, table management, and data operations.
"""

import psycopg2
from psycopg2.extras import execute_values
import logging
import os
from typing import List, Dict, Tuple, Optional, Sequence

logger = logging.getLogger(__name__)


class PM25Database:
    """PostgreSQL helper for PM2.5 hourly ingestion."""
    
    def __init__(self, host: str = "postgres", port: int = 5432, 
                 database: str = "pm25", user: str = "postgres", 
                 password: str = "postgres"):
        """Initialize DB connection parameters."""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
    
    def connect(self):
        """Establish connection to PostgreSQL."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                options="-c timezone=Asia/Bangkok",
            )
            self.conn.autocommit = False  # Use transactions
            logger.info(f"Connected to {self.database}@{self.host}")
            return self.conn
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")
    
    def ensure_table(self) -> bool:
        """Create pm25_raw_hourly table if not exists."""
        if not self.conn:
            self.connect()
        
        create_sql = """
        CREATE TABLE IF NOT EXISTS pm25_raw_hourly (
            id SERIAL PRIMARY KEY,
            station_id INT NOT NULL,
            station_name TEXT,
            station_name_en TEXT,
            timestamp TIMESTAMPTZ NOT NULL,
            pm25 FLOAT,
            pm10 FLOAT,
            temp FLOAT,
            rh FLOAT,
            ws FLOAT,
            wd FLOAT,
            ingestion_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(station_id, timestamp),
            CHECK (pm25 >= 0 AND pm25 <= 500),
            CHECK (pm10 >= 0),
            CHECK (rh >= 0 AND rh <= 100),
            CHECK (ws >= 0)
        );
        """
        
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_station_timestamp 
            ON pm25_raw_hourly(station_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON pm25_raw_hourly(timestamp DESC);
        """

        alter_sql = """
        ALTER TABLE pm25_raw_hourly
        ADD COLUMN IF NOT EXISTS station_name TEXT;

        ALTER TABLE pm25_raw_hourly
        ADD COLUMN IF NOT EXISTS station_name_en TEXT;

        ALTER TABLE pm25_raw_hourly
        ALTER COLUMN ingestion_time
        SET DEFAULT CURRENT_TIMESTAMP;
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(create_sql)
            cur.execute(index_sql)
            cur.execute(alter_sql)
            self.conn.commit()
            logger.info("pm25_raw_hourly table and indexes ensured")
            cur.close()
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to create table: {e}")
            raise
    
    def _prepare_record_values(self, records: List[Dict]) -> List[Tuple]:
        return [
            (
                record.get("station_id"),
                record.get("station_name"),
                record.get("station_name_en"),
                record.get("timestamp"),
                record.get("pm25"),
                record.get("pm10"),
                record.get("temp"),
                record.get("rh"),
                record.get("ws"),
                record.get("wd"),
            )
            for record in records
        ]

    def insert_records(self, records: List[Dict], commit: bool = True) -> Tuple[int, int]:
        """
        Insert records idempotently (duplicates ignored).
        
        Args:
            records: List of dicts with keys: station_id, timestamp, pm25, pm10, temp, rh, ws, wd
        
        Returns:
            (inserted_count, duplicate_count)
        """
        if not self.conn:
            self.connect()
        
        if not records:
            logger.warning("No records to insert")
            return 0, 0
        
        values = self._prepare_record_values(records)
        
        insert_sql = """
        INSERT INTO pm25_raw_hourly
        (station_id, station_name, station_name_en, timestamp, pm25, pm10, temp, rh, ws, wd)
        VALUES %s
        ON CONFLICT (station_id, timestamp) DO NOTHING
        RETURNING 1;
        """
        
        try:
            cur = self.conn.cursor()
            inserted_rows = execute_values(cur, insert_sql, values, fetch=True)
            if commit:
                self.conn.commit()

            inserted = len(inserted_rows)
            duplicates = len(records) - inserted
            
            logger.info(f"Inserted {inserted} records, {duplicates} duplicates skipped")
            cur.close()
            
            return inserted, duplicates
        except psycopg2.Error as e:
            if commit:
                self.conn.rollback()
            logger.error(f"Insert failed: {e}")
            raise

    def delete_records_in_range(
        self,
        start_timestamp,
        end_timestamp,
        station_ids: Optional[Sequence[int]] = None,
        commit: bool = True,
    ) -> int:
        """Delete records in a half-open timestamp range [start, end)."""
        if not self.conn:
            self.connect()

        delete_sql = """
        DELETE FROM pm25_raw_hourly
        WHERE timestamp >= %s
          AND timestamp < %s
        """
        params: List[object] = [start_timestamp, end_timestamp]

        if station_ids:
            delete_sql += " AND station_id = ANY(%s)"
            params.append(list(station_ids))

        try:
            cur = self.conn.cursor()
            cur.execute(delete_sql, tuple(params))
            deleted_count = cur.rowcount
            if commit:
                self.conn.commit()
            cur.close()
            logger.info(
                "Deleted %s rows from pm25_raw_hourly for range [%s, %s)",
                deleted_count,
                start_timestamp,
                end_timestamp,
            )
            return deleted_count
        except psycopg2.Error as e:
            if commit:
                self.conn.rollback()
            logger.error(f"Delete failed: {e}")
            raise

    def replace_records_for_range(
        self,
        records: List[Dict],
        start_timestamp,
        end_timestamp,
        station_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[int, int, int]:
        """
        Atomically replace a station/date range in pm25_raw_hourly.

        Returns:
            (deleted_count, inserted_count, duplicate_count)
        """
        if not self.conn:
            self.connect()

        try:
            deleted_count = self.delete_records_in_range(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                station_ids=station_ids,
                commit=False,
            )
            inserted_count, duplicate_count = self.insert_records(records, commit=False)
            self.conn.commit()
            return deleted_count, inserted_count, duplicate_count
        except psycopg2.Error:
            self.conn.rollback()
            raise
    
    def _is_duplicate(self, record: Dict) -> bool:
        """Check if record already exists (station_id, timestamp)."""
        if not self.conn:
            return False
        
        check_sql = """
        SELECT 1 FROM pm25_raw_hourly 
        WHERE station_id = %s AND timestamp = %s
        LIMIT 1;
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(check_sql, (record.get("station_id"), record.get("timestamp")))
            result = cur.fetchone()
            cur.close()
            return result is not None
        except psycopg2.Error:
            return False
    
    def get_latest_by_station(self, station_id: int, limit: int = 7) -> List[Dict]:
        """Get latest records for a station (for drift detection)."""
        if not self.conn:
            self.connect()
        
        query_sql = """
        SELECT station_id, timestamp, pm25, pm10, temp, rh, ws, wd
        FROM pm25_raw_hourly
        WHERE station_id = %s
        ORDER BY timestamp DESC
        LIMIT %s;
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(query_sql, (station_id, limit))
            rows = cur.fetchall()
            cur.close()
            
            # Convert to list of dicts
            columns = ["station_id", "timestamp", "pm25", "pm10", "temp", "rh", "ws", "wd"]
            return [dict(zip(columns, row)) for row in rows]
        except psycopg2.Error as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_row_count(self, station_id: Optional[int] = None) -> int:
        """Get total rows or rows for a specific station."""
        if not self.conn:
            self.connect()
        
        if station_id:
            query_sql = "SELECT COUNT(*) FROM pm25_raw_hourly WHERE station_id = %s;"
            params = (station_id,)
        else:
            query_sql = "SELECT COUNT(*) FROM pm25_raw_hourly;"
            params = ()
        
        try:
            cur = self.conn.cursor()
            if params:
                cur.execute(query_sql, params)
            else:
                cur.execute(query_sql)
            result = cur.fetchone()[0]
            cur.close()
            return result
        except psycopg2.Error as e:
            logger.error(f"Count query failed: {e}")
            return 0


def get_db_connection(config_path: str = None) -> PM25Database:
    """Factory function to get a database connection."""
    # In production, would load credentials from config_path
    db = PM25Database(
        host=os.getenv("PM25_DB_HOST", "postgres"),
        port=int(os.getenv("PM25_DB_PORT", "5432")),
        database=os.getenv("PM25_DB_NAME", "pm25"),
        user=os.getenv("PM25_DB_USER", "postgres"),
        password=os.getenv("PM25_DB_PASSWORD", "postgres"),
    )
    db.connect()
    return db
