"""
Monitoring utilities for PM2.5 hourly ingestion pipeline.
Detects data quality issues, sensor drift, and generates alerts.
"""

import logging
from typing import Dict, List
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitors data quality and detects anomalies in PM2.5 ingestion."""

    def __init__(self, db):
        """Initialize with database connection."""
        self.db = db

    def check_recent_data(self, station_id: int, hours: int = 24) -> Dict:
        """
        Check data quality metrics for recent records.
        
        Returns:
            Dict with keys: records_count, null_rate, outliers, mean_pm25, std_pm25, alert_level
        """
        # Query recent data
        query = """
        SELECT pm25, pm10, temp, rh, ws
        FROM pm25_raw_hourly
        WHERE station_id = %s 
          AND timestamp > NOW() - INTERVAL '%d hours'
        ORDER BY timestamp DESC;
        """ % (station_id, hours)

        try:
            cur = self.db.conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            cur.close()

            if not rows:
                logger.warning(f"No recent data for station {station_id}")
                return {
                    "station_id": station_id,
                    "records_count": 0,
                    "null_rate": 1.0,
                    "outliers": 0,
                    "mean_pm25": None,
                    "std_pm25": None,
                    "alert_level": "NO_DATA",
                }

            # Extract PM2.5 values
            pm25_values = [row[0] for row in rows if row[0] is not None]
            null_count = len(rows) - len(pm25_values)

            # Calculate statistics
            mean_pm25 = statistics.mean(pm25_values) if pm25_values else None
            std_pm25 = statistics.stdev(pm25_values) if len(pm25_values) > 1 else 0

            # Detect outliers (values > mean + 2*std)
            outliers = sum(1 for v in pm25_values if mean_pm25 and v > mean_pm25 + 2*std_pm25)

            # Determine alert level
            null_rate = null_count / len(rows)
            alert_level = "OK"

            if null_rate > 0.5:
                alert_level = "HIGH_NULL_RATE"
            elif outliers > len(pm25_values) * 0.1:
                alert_level = "HIGH_OUTLIERS"
            elif mean_pm25 and mean_pm25 > 400:
                alert_level = "EXTREME_VALUES"

            return {
                "station_id": station_id,
                "records_count": len(rows),
                "null_rate": null_rate,
                "outliers": outliers,
                "mean_pm25": round(mean_pm25, 2) if mean_pm25 else None,
                "std_pm25": round(std_pm25, 2),
                "alert_level": alert_level,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return {
                "station_id": station_id,
                "error": str(e),
                "alert_level": "CHECK_FAILED",
            }

    def detect_sensor_drift(self, station_id: int,
                           current_window_hours: int = 1,
                           baseline_window_hours: int = 7*24) -> Dict:
        """
        Detect sensor drift by comparing current values to baseline.
        
        Compares last hour's mean PM2.5 to 7-day rolling mean.
        
        Returns:
            Dict with keys: current_mean, baseline_mean, drift_percentage, alert
        """
        try:
            cur = self.db.conn.cursor()

            # Current window (last hour)
            current_query = """
            SELECT AVG(pm25)
            FROM pm25_raw_hourly
            WHERE station_id = %s 
              AND timestamp > NOW() - INTERVAL '%d hours'
            """
            cur.execute(current_query % (station_id, current_window_hours))
            current_mean = cur.fetchone()[0]

            # Baseline window (7 days, excluding last hour)
            baseline_query = """
            SELECT AVG(pm25)
            FROM pm25_raw_hourly
            WHERE station_id = %s 
              AND timestamp > NOW() - INTERVAL '%d hours'
              AND timestamp <= NOW() - INTERVAL '%d hours'
            """
            cur.execute(baseline_query % (station_id, baseline_window_hours + current_window_hours, current_window_hours))
            baseline_mean = cur.fetchone()[0]

            cur.close()

            if not current_mean or not baseline_mean or baseline_mean == 0:
                logger.warning(f"Insufficient data for drift detection (station {station_id})")
                return {
                    "station_id": station_id,
                    "current_mean": current_mean,
                    "baseline_mean": baseline_mean,
                    "drift_percentage": None,
                    "alert": "INSUFFICIENT_DATA",
                }

            # Calculate drift percentage
            drift_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

            # Determine alert
            alert = "OK"
            if abs(drift_pct) > 50:
                alert = "SEVERE_DRIFT"
            elif abs(drift_pct) > 25:
                alert = "MODERATE_DRIFT"

            return {
                "station_id": station_id,
                "current_mean": round(current_mean, 2),
                "baseline_mean": round(baseline_mean, 2),
                "drift_percentage": round(drift_pct, 1),
                "alert": alert,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {
                "station_id": station_id,
                "error": str(e),
                "alert": "DETECTION_FAILED",
            }

    def check_api_health(self, min_records_per_hour: int = 2) -> Dict:
        """
        Check if recent hourly ingestions are healthy.
        
        Returns:
            Dict with keys: last_hour_records, expected_records, health_status
        """
        try:
            cur = self.db.conn.cursor()

            query = """
            SELECT COUNT(DISTINCT station_id) as station_count,
                   COUNT(*) as total_records
            FROM pm25_raw_hourly
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            """
            cur.execute(query)
            result = cur.fetchone()
            cur.close()

            station_count, total_records = result if result else (0, 0)
            expected_records = station_count * min_records_per_hour

            health_status = "HEALTHY"
            if total_records == 0:
                health_status = "NO_DATA"
            elif total_records < expected_records * 0.5:
                health_status = "DEGRADED"
            elif total_records < expected_records:
                health_status = "PARTIAL"

            return {
                "last_hour_records": total_records,
                "expected_records": expected_records,
                "station_count": station_count,
                "health_status": health_status,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking API health: {e}")
            return {
                "error": str(e),
                "health_status": "CHECK_FAILED",
            }


def generate_monitoring_report(db, station_ids: List[int] = None) -> Dict:
    """Generate comprehensive monitoring report for all stations."""
    monitor = DataQualityMonitor(db)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "api_health": monitor.check_api_health(),
        "stations": {},
    }

    # Default to stations 145 and 10
    if not station_ids:
        station_ids = [145, 10]

    for station_id in station_ids:
        report["stations"][station_id] = {
            "data_quality": monitor.check_recent_data(station_id, hours=24),
            "sensor_drift": monitor.detect_sensor_drift(station_id),
        }

    return report
