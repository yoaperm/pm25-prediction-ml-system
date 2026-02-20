"""
Data Loader Module
==================
Loads PM2.5 data from Excel files and station metadata.
"""

import pandas as pd
import yaml
import os


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_station_data(
    file_path: str,
    sheet_name: str,
    station_id: str,
) -> pd.DataFrame:
    """
    Load PM2.5 data for a specific station from Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.
    sheet_name : str
        Name of the data sheet.
    station_id : str
        Station ID column to extract (e.g., '10T').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'pm25'].
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Clean: keep only rows where Date is a valid datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Extract station column
    if station_id not in df.columns:
        raise ValueError(f"Station '{station_id}' not found in {file_path}")

    result = df[["Date", station_id]].copy()
    result.columns = ["date", "pm25"]

    # Convert pm25 to numeric (handles 'n/a' or other strings)
    result["pm25"] = pd.to_numeric(result["pm25"], errors="coerce")

    # Sort by date
    result = result.sort_values("date").reset_index(drop=True)

    return result


def load_metadata(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load station metadata from Excel file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['index', 'station_id', 'station_name', 'detail'].
    """
    meta = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Find the header row (contains 'ลำดับ')
    header_idx = None
    for i in range(min(10, len(meta))):
        if "ลำดับ" in str(meta.iloc[i].values):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find metadata header row")

    # Extract data rows after header (skip blank rows)
    data_rows = meta.iloc[header_idx + 1 :].copy()
    data_rows.columns = meta.iloc[header_idx].values

    # Keep relevant columns and drop NaN rows
    cols = ["ลำดับ", "รหัสสถานี", "ชื่อสถานี", "รายละเอียดจุดติดตั้งสถานี"]
    available_cols = [c for c in cols if c in data_rows.columns]
    data_rows = data_rows[available_cols].dropna(subset=["รหัสสถานี"])

    data_rows.columns = ["index", "station_id", "station_name", "detail"][
        : len(available_cols)
    ]
    data_rows = data_rows.reset_index(drop=True)

    return data_rows


def load_train_test_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train (2024) and test (2025) data for the configured station.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) each with columns ['date', 'pm25'].
    """
    station_id = config["station"]["id"]

    train_df = load_station_data(
        file_path=config["data"]["train_file"],
        sheet_name=config["data"]["train_sheet"],
        station_id=station_id,
    )

    test_df = load_station_data(
        file_path=config["data"]["test_file"],
        sheet_name=config["data"]["test_sheet"],
        station_id=station_id,
    )

    return train_df, test_df


if __name__ == "__main__":
    config = load_config()
    train_df, test_df = load_train_test_data(config)
    print(f"Train data: {train_df.shape}, date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test data:  {test_df.shape}, date range: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"\nTrain missing: {train_df['pm25'].isnull().sum()}")
    print(f"Test missing:  {test_df['pm25'].isnull().sum()}")
    print(f"\nTrain sample:\n{train_df.head()}")
