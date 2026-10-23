"""
Data Handler Module for SCADA Time-Series CSV Uploads in AquaPINN.

Provides robust CSV parsing, column mapping aliasing, missing-value time-series interpolation,
physical data validation, and sample SCADA dataset generation.
"""

from __future__ import annotations
import io
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Expected canonical column definitions and known aliases
COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "datetime", "date", "time", "t", "timestamp_utc"],
    "flow_rate": ["flow", "q", "flow_rate", "q_in", "influent_flow", "inflow", "flow_m3d"],
    "influent_cod": ["cod", "influent_cod", "s_s", "ss", "cod_in", "readily_cod", "bods"],
    "influent_nh4": ["nh4", "nh4_n", "s_nh", "snh", "ammonium", "ammonia", "nh4n", "nitrogen_nh4"],
    "temperature": ["temp", "temperature", "temp_c", "t_c", "temperature_c", "temp_degc"],
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe column names and map known SCADA aliases to canonical names."""
    df_clean = df.copy()
    column_mapping = {}
    for col in df_clean.columns:
        norm_col = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        matched = False
        for canonical, aliases in COLUMN_ALIASES.items():
            if norm_col in aliases:
                column_mapping[col] = canonical
                matched = True
                break
        if not matched:
            column_mapping[col] = norm_col

    df_clean = df_clean.rename(columns=column_mapping)
    return df_clean


def validate_scada_data(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Validate uploaded SCADA dataframe against required schema and physical limits.
    
    Returns:
        Tuple of (is_valid, error_messages, warning_messages)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df.empty:
        errors.append("Uploaded dataset is empty.")
        return False, errors, warnings

    required_canonical = ["timestamp", "flow_rate", "influent_cod", "influent_nh4", "temperature"]
    missing_cols = [col for col in required_canonical if col not in df.columns]

    if missing_cols:
        errors.append(f"Missing required columns (or recognizable aliases): {missing_cols}. "
                      f"Expected columns include: timestamp, flow_rate (Q), influent_cod, influent_nh4, temperature.")
        return False, errors, warnings

    # Validate timestamp parsing
    try:
        pd.to_datetime(df["timestamp"])
    except Exception as e:
        errors.append(f"Failed to parse 'timestamp' column as datetime: {e}")

    # Physical range validations
    if "flow_rate" in df.columns:
        if (df["flow_rate"] < 0).any():
            errors.append("Flow rate ($Q$) contains negative values, which is physically impossible.")
        elif (df["flow_rate"] == 0).all():
            warnings.append("Flow rate ($Q$) is zero across all timestamps.")

    if "influent_cod" in df.columns:
        if (df["influent_cod"] < 0).any():
            errors.append("Influent COD contains negative values.")
        elif (df["influent_cod"] > 5000).any():
            warnings.append("Some Influent COD values exceed 5000 g/m³, which is exceptionally high.")

    if "influent_nh4" in df.columns:
        if (df["influent_nh4"] < 0).any():
            errors.append("Influent Ammonium ($S_{NH}$) contains negative values.")
        elif (df["influent_nh4"] > 500).any():
            warnings.append("Some Influent Ammonium values exceed 500 g/m³, which is unusually high.")

    if "temperature" in df.columns:
        if ((df["temperature"] < 0) | (df["temperature"] > 50)).any():
            warnings.append("Temperature values fall outside typical biological wastewater range (0°C - 50°C).")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing values (NaNs) in SCADA time-series data using time-weighted interpolation.
    """
    df_processed = df.copy()

    df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])
    df_processed = df_processed.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = ["flow_rate", "influent_cod", "influent_nh4", "temperature"]
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

    df_indexed = df_processed.set_index("timestamp")

    try:
        df_indexed[numeric_cols] = df_indexed[numeric_cols].interpolate(method="time")
    except Exception:
        df_indexed[numeric_cols] = df_indexed[numeric_cols].interpolate(method="linear")

    df_indexed[numeric_cols] = df_indexed[numeric_cols].ffill().bfill()

    df_interpolated = df_indexed.reset_index()

    t0 = df_interpolated["timestamp"].min()
    df_interpolated["time_days"] = (df_interpolated["timestamp"] - t0).dt.total_seconds() / 86400.0

    return df_interpolated


def process_scada_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """
    Load, standardize, validate, and interpolate an uploaded SCADA CSV file.
    
    Args:
        uploaded_file: Streamlit UploadedFile or file path/buffer.
        
    Returns:
        Tuple of (processed_dataframe or None, error_messages, warning_messages)
    """
    try:
        if isinstance(uploaded_file, bytes):
            df = pd.read_csv(io.BytesIO(uploaded_file))
        elif isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, [f"Failed to read CSV file: {e}"], []

    df_standardized = standardize_columns(df)
    is_valid, errors, warnings = validate_scada_data(df_standardized)

    if not is_valid:
        return None, errors, warnings

    df_cleaned = interpolate_missing_values(df_standardized)
    return df_cleaned, errors, warnings


def generate_sample_scada_csv(num_days: int = 7, points_per_day: int = 24) -> str:
    """
    Generate a realistic synthetic SCADA time-series CSV string for testing and demonstration.
    """
    total_points = num_days * points_per_day
    timestamps = pd.date_range(start="2026-06-01 00:00:00", periods=total_points, freq=f"{24 // points_per_day}h" if points_per_day <= 24 else "1h")
    
    t_hours = np.arange(total_points) * (24.0 / points_per_day)
    
    flow_base = 1000.0 + 300.0 * np.sin(2.0 * np.pi * (t_hours - 6.0) / 24.0) + 150.0 * np.sin(2.0 * np.pi * (t_hours - 18.0) / 24.0)
    flow_noise = np.random.normal(0, 30.0, total_points)
    flow_rate = np.clip(flow_base + flow_noise, 300.0, 3000.0)

    cod_base = 200.0 + 80.0 * np.sin(2.0 * np.pi * (t_hours - 8.0) / 24.0)
    cod_noise = np.random.normal(0, 15.0, total_points)
    influent_cod = np.clip(cod_base + cod_noise, 80.0, 600.0)

    nh4_base = 40.0 + 12.0 * np.sin(2.0 * np.pi * (t_hours - 7.0) / 24.0)
    nh4_noise = np.random.normal(0, 3.0, total_points)
    influent_nh4 = np.clip(nh4_base + nh4_noise, 10.0, 100.0)

    temp_base = 20.0 + 3.0 * np.sin(2.0 * np.pi * (t_hours - 14.0) / 24.0)
    temp_noise = np.random.normal(0, 0.5, total_points)
    temperature = np.clip(temp_base + temp_noise, 10.0, 35.0)

    nan_indices_flow = np.random.choice(total_points, size=max(1, total_points // 50), replace=False)
    flow_rate_with_nans = flow_rate.copy()
    flow_rate_with_nans[nan_indices_flow] = np.nan

    nan_indices_cod = np.random.choice(total_points, size=max(1, total_points // 50), replace=False)
    cod_with_nans = influent_cod.copy()
    cod_with_nans[nan_indices_cod] = np.nan

    sample_df = pd.DataFrame({
        "timestamp": timestamps,
        "flow_rate": flow_rate_with_nans,
        "influent_cod": cod_with_nans,
        "influent_nh4": influent_nh4,
        "temperature": temperature,
    })

    return sample_df.to_csv(index=False)


def load_bsm1_benchmark(weather_condition: str = "Dry Weather", num_days: int = 7, points_per_day: int = 96) -> pd.DataFrame:
    """
    Generate IWA BSM1-style influent benchmark profiles for Dry Weather, Rain Event, and Storm Event.
    
    Args:
        weather_condition: One of "Dry Weather", "Rain Event", "Storm Event".
        num_days: Duration in days (default 7).
        points_per_day: Temporal resolution (default 96 points/day = 15-minute intervals).
    
    Returns:
        pd.DataFrame with columns: timestamp, time_days, flow_rate, influent_cod, influent_nh4, temperature
    """
    total_points = num_days * points_per_day
    timestamps = pd.date_range(start="2026-06-01 00:00:00", periods=total_points, freq="15min")
    t_hours = np.arange(total_points) * (24.0 / points_per_day)
    t_days = t_hours / 24.0

    condition = weather_condition.strip().title()
    if "Rain" in condition:
        cond_type = "rain"
    elif "Storm" in condition:
        cond_type = "storm"
    else:
        cond_type = "dry"

    q_base = 1000.0
    q_diurnal = 300.0 * np.sin(2.0 * np.pi * (t_hours - 6.0) / 24.0) + 150.0 * np.sin(2.0 * np.pi * (t_hours - 18.0) / 24.0)
    
    cod_base = 150.0
    cod_diurnal = 40.0 * np.sin(2.0 * np.pi * (t_hours - 8.0) / 24.0)

    nh4_base = 31.5
    nh4_diurnal = 8.0 * np.sin(2.0 * np.pi * (t_hours - 7.0) / 24.0)

    temp_base = 15.0 + 2.0 * np.sin(2.0 * np.pi * t_days / 365.0)

    if cond_type == "rain":
        rain_mask = (t_days >= 2.0) & (t_days <= 3.5)
        rain_pulse = 700.0 * np.sin(np.pi * (t_days[rain_mask] - 2.0) / 1.5)
        q_diurnal_mod = q_diurnal.copy()
        q_diurnal_mod[rain_mask] += rain_pulse
        
        flow_rate = q_base + q_diurnal_mod
        influent_cod = (cod_base + cod_diurnal) * np.where(rain_mask, 0.65, 1.0)
        influent_nh4 = (nh4_base + nh4_diurnal) * np.where(rain_mask, 0.60, 1.0)
        temperature = temp_base - np.where(rain_mask, 1.5, 0.0)

    elif cond_type == "storm":
        storm_mask = (t_days >= 2.5) & (t_days <= 3.0)
        storm_surge = 1500.0 * np.sin(np.pi * (t_days[storm_mask] - 2.5) / 0.5)
        q_diurnal_mod = q_diurnal.copy()
        q_diurnal_mod[storm_mask] += storm_surge

        flow_rate = q_base + q_diurnal_mod
        first_flush = (t_days >= 2.5) & (t_days < 2.65)
        cod_mod = (cod_base + cod_diurnal) * np.where(storm_mask, 0.5, 1.0)
        cod_mod[first_flush] *= 1.8

        nh4_mod = (nh4_base + nh4_diurnal) * np.where(storm_mask, 0.45, 1.0)
        temperature = temp_base - np.where(storm_mask, 2.5, 0.0)
        
        influent_cod = cod_mod
        influent_nh4 = nh4_mod

    else:
        flow_rate = q_base + q_diurnal
        influent_cod = cod_base + cod_diurnal
        influent_nh4 = nh4_base + nh4_diurnal
        temperature = temp_base

    flow_rate = np.clip(flow_rate + np.random.normal(0, 15.0, total_points), 200.0, 4000.0)
    influent_cod = np.clip(influent_cod + np.random.normal(0, 5.0, total_points), 50.0, 500.0)
    influent_nh4 = np.clip(influent_nh4 + np.random.normal(0, 1.5, total_points), 5.0, 100.0)
    temperature = np.clip(temperature + np.random.normal(0, 0.2, total_points), 10.0, 30.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "time_days": t_days,
        "flow_rate": flow_rate,
        "influent_cod": influent_cod,
        "influent_nh4": influent_nh4,
        "temperature": temperature,
    })
    return df

