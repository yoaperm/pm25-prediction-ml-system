#!/usr/bin/env python3
"""
insert_predictions_to_db.py
============================
Insert model predictions into PostgreSQL pm25_predicted_hourly table.

Usage:
    python scripts/insert_predictions_to_db.py results/predictions_sarima_56.csv --station 56 --model sarima
    python scripts/insert_predictions_to_db.py results/predictions_transformer_59.csv --station 59 --model transformer
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import Error


def insert_predictions(csv_path, station_id, model_name, db_host="localhost", db_port=5432, 
                       db_name="pm25", db_user="postgres", db_pass="postgres"):
    """Insert predictions from CSV into PostgreSQL."""
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return False
    
    try:
        # Connect
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_pass
        )
        cur = conn.cursor()
        print(f"✅ Connected to {db_host}:{db_port}/{db_name}")
        
    except Error as e:
        print(f"❌ Connection error: {e}")
        return False
    
    try:
        # Load CSV
        print(f"📖 Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ["timestamp", "predicted_pm25"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"❌ Error: CSV missing columns: {missing_cols}")
            print(f"   Available: {list(df.columns)}")
            return False
        
        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        print(f"   Rows to insert: {len(df)}")
        print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"   Station: {station_id}, Model: {model_name}")
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name='pm25_predicted_hourly'
            )
        """)
        if not cur.fetchone()[0]:
            print("❌ Error: Table pm25_predicted_hourly does not exist")
            return False
        
        # Insert or update
        inserted = 0
        updated = 0
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0 and idx > 0:
                print(f"   Processing {idx}/{len(df)}...")
            
            try:
                cur.execute("""
                    INSERT INTO pm25_predicted_hourly 
                    (station_id, timestamp, model, predicted_pm25)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (station_id, timestamp, model) DO UPDATE
                    SET predicted_pm25 = EXCLUDED.predicted_pm25
                """, (
                    station_id,
                    row["timestamp"],
                    model_name,
                    float(row["predicted_pm25"])
                ))
                
                # Check if insert or update
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1
                    
            except Error as e:
                print(f"   ⚠️  Row {idx} error: {e}")
                continue
        
        conn.commit()
        print(f"\n✅ Success!")
        print(f"   Inserted: {inserted} new rows")
        print(f"   Updated: {updated} existing rows")
        print(f"   Total: {inserted + updated} rows affected")
        
        # Verify
        cur.execute("""
            SELECT COUNT(*) FROM pm25_predicted_hourly 
            WHERE station_id=%s AND model=%s
        """, (station_id, model_name))
        total = cur.fetchone()[0]
        print(f"   DB total for {model_name} station {station_id}: {total} rows")
        
        cur.close()
        conn.close()
        return True
        
    except Error as e:
        print(f"❌ Database error: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Insert model predictions into PostgreSQL pm25_predicted_hourly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Insert SARIMA predictions for station 56
  python scripts/insert_predictions_to_db.py results/predictions_sarima_56.csv --station 56 --model sarima

  # Insert Transformer predictions for station 59
  python scripts/insert_predictions_to_db.py results/predictions_transformer_59.csv --station 59 --model transformer

  # Custom database connection
  python scripts/insert_predictions_to_db.py pred.csv --station 56 --model lstm \\
    --db-host db.example.com --db-user myuser --db-pass mypass
        """)
    
    parser.add_argument("csv", type=str, help="Path to predictions CSV file")
    parser.add_argument("--station", type=int, required=True,
                        help="Station ID (56, 57, 58, 59, 61)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["sarima", "lstm", "transformer", "xgboost", "ridge_regression"],
                        help="Model name")
    parser.add_argument("--db-host", type=str, default="localhost",
                        help="Database host (default: localhost)")
    parser.add_argument("--db-port", type=int, default=5432,
                        help="Database port (default: 5432)")
    parser.add_argument("--db-name", type=str, default="pm25",
                        help="Database name (default: pm25)")
    parser.add_argument("--db-user", type=str, default="postgres",
                        help="Database user (default: postgres)")
    parser.add_argument("--db-pass", type=str, default="postgres",
                        help="Database password (default: postgres)")
    
    args = parser.parse_args()
    
    # Validate station ID
    valid_stations = [56, 57, 58, 59, 61]
    if args.station not in valid_stations:
        print(f"❌ Error: Station {args.station} not in valid list: {valid_stations}")
        sys.exit(1)
    
    success = insert_predictions(
        args.csv,
        args.station,
        args.model,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_pass=args.db_pass
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
