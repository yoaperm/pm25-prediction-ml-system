#!/usr/bin/env python3
"""
Test script for PM2.5 Hourly Ingestion DAG.
Sets up the database and runs basic tests locally.

Usage:
    python scripts/test_hourly_dag.py --setup     # Initialize DB
    python scripts/test_hourly_dag.py --verify    # Check DAG syntax
    python scripts/test_hourly_dag.py --mock      # Run mock ingestion
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_db():
    """Initialize PostgreSQL database and create tables."""
    logger.info("Setting up PM25 database...")
    
    try:
        import psycopg2
        from airflow_db import PM25Database
        
        # Create pm25 database if it doesn't exist
        try:
            conn = psycopg2.connect(
                host="postgres",
                port=5432,
                database="postgres",
                user="postgres",
                password="postgres"
            )
            cur = conn.cursor()
            
            # Check if pm25 database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'pm25';")
            if not cur.fetchone():
                cur.execute("CREATE DATABASE pm25;")
                logger.info("✅ Created pm25 database")
            else:
                logger.info("✅ pm25 database already exists")
            
            conn.commit()
            cur.close()
            conn.close()
        except psycopg2.Error as e:
            logger.warning(f"Could not create pm25 database: {e} (may already exist)")
        
        # Connect to pm25 database and create tables
        db = PM25Database(
            host="postgres",
            database="pm25",
            user="postgres",
            password="postgres"
        )
        db.connect()
        db.ensure_table()
        db.close()
        
        logger.info("✅ Database setup complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def verify_dag():
    """Verify DAG syntax is valid."""
    logger.info("Verifying DAG syntax...")
    
    try:
        import py_compile
        
        dag_path = Path(__file__).parent.parent / "dags" / "pm25_hourly_ingest_dag.py"
        py_compile.compile(str(dag_path), doraise=True)
        
        logger.info(f"✅ DAG syntax valid: {dag_path}")
        return True
        
    except py_compile.PyCompileError as e:
        logger.error(f"❌ DAG syntax error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def run_mock_ingestion():
    """Run a mock ingestion cycle to test the full pipeline."""
    logger.info("Running mock ingestion...")
    
    try:
        from airflow_db import get_db_connection
        from datetime import datetime, timedelta
        import json
        
        # Create mock records
        now = datetime.utcnow()
        mock_records = [
            {
                "station_id": 145,
                "timestamp": now.isoformat() + "Z",
                "pm25": 35.2,
                "pm10": 55.4,
                "temp": 28.5,
                "rh": 72.0,
                "ws": 1.2,
                "wd": 180.0,
            },
            {
                "station_id": 10,
                "timestamp": now.isoformat() + "Z",
                "pm25": 42.1,
                "pm10": 62.3,
                "temp": 29.0,
                "rh": 70.0,
                "ws": 1.5,
                "wd": 175.0,
            },
        ]
        
        # Initialize DB and insert
        db = get_db_connection()
        stored, duplicates = db.insert_records(mock_records)
        
        # Verify data
        count = db.get_row_count()
        logger.info(f"✅ Mock ingestion complete:")
        logger.info(f"   - Records inserted: {stored}")
        logger.info(f"   - Duplicates skipped: {duplicates}")
        logger.info(f"   - Total rows in table: {count}")
        
        # Check latest data
        for station_id in [145, 10]:
            latest = db.get_latest_by_station(station_id, limit=1)
            if latest:
                logger.info(f"   - Station {station_id}: PM2.5={latest[0]['pm25']} at {latest[0]['timestamp']}")
        
        db.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test PM2.5 Hourly Ingestion DAG")
    parser.add_argument("--setup", action="store_true", help="Initialize database")
    parser.add_argument("--verify", action="store_true", help="Verify DAG syntax")
    parser.add_argument("--mock", action="store_true", help="Run mock ingestion")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if not any([args.setup, args.verify, args.mock, args.all]):
        parser.print_help()
        sys.exit(1)
    
    results = {}
    
    if args.setup or args.all:
        results["setup"] = setup_db()
    
    if args.verify or args.all:
        results["verify"] = verify_dag()
    
    if args.mock or args.all:
        # Setup must run before mock
        if "setup" not in results or results["setup"]:
            results["mock"] = run_mock_ingestion()
        else:
            logger.warning("Skipping mock ingestion (setup failed)")
            results["mock"] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Test Summary:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name.capitalize():20s} {status}")
    
    all_passed = all(results.values())
    logger.info("="*50)
    
    if all_passed:
        logger.info("✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
