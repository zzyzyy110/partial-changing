import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats


warnings.simplefilter("ignore")

CALCE_STANDARD_CAPACITY = 1160.0
SOH_JUMP_THRESHOLD = 0.05

INPUT_DATA_DIR = "path/to/calce_raw_data"
OUTPUT_DATA_DIR = "path/to/calce_extracted_features"

FILE_NAMES = [
    "CS2_35_detailed.csv",
    "CS2_36_detailed.csv",
    "CS2_37_detailed.csv",
    "CS2_38_detailed.csv",
]


def calculate_soh_calce(df: pd.DataFrame) -> Dict[int, float]:
    soh_data = {}

    for cycle, group in df.groupby("cycle_index"):
        max_capacity = group["cumulative_capacity(mA.h)"].max()
        soh_value = max_capacity / CALCE_STANDARD_CAPACITY
        soh_data[cycle] = soh_value

    return soh_data


def extract_soh_data(df: pd.DataFrame) -> Dict[int, float]:
    soh_data = {}

    for cycle, group in df.groupby("cycle_index"):
        soh_values = group["SOH"].dropna().unique()

        if len(soh_values) > 0:
            soh_data[cycle] = soh_values[0]
        else:
            print(
                f"Warning: Cycle {cycle} has no valid SOH value and will be excluded."
            )

    return soh_data


def detect_abnormal_soh_cycles(
    soh_data: Dict[int, float],
    sigma: float = 3.0,
) -> List[int]:
    if not soh_data:
        print("Warning: No valid SOH data are available for abnormal-cycle detection.")
        return []

    sorted_cycles = sorted(soh_data.keys())
    sorted_soh = np.array([soh_data[cycle] for cycle in sorted_cycles])

    abnormal_cycles = []

    for i in range(1, len(sorted_cycles)):
        current_cycle = sorted_cycles[i]
        current_soh = sorted_soh[i]
        previous_soh = sorted_soh[i - 1]
        soh_difference = abs(current_soh - previous_soh)

        if soh_difference > SOH_JUMP_THRESHOLD:
            abnormal_cycles.append(current_cycle)
            print(
                f"Cycle {current_cycle} flagged as abnormal: "
                f"previous SOH={previous_soh:.5f}, "
                f"current SOH={current_soh:.5f}, "
                f"difference={soh_difference:.5f}"
            )

    print("SOH abnormal-cycle detection summary:")
    print(
        f"Threshold: absolute SOH difference between adjacent cycles "
        f"> {SOH_JUMP_THRESHOLD:.5f}"
    )
    print(f"Total valid cycles: {len(sorted_cycles)}")
    print(f"Detected abnormal cycles: {len(abnormal_cycles)}")
    print(
        f"Abnormal cycle indices: "
        f"{sorted(abnormal_cycles) if abnormal_cycles else 'None'}"
    )

    return abnormal_cycles


def calculate_voltage_features(
    voltage: pd.Series,
) -> Tuple[float, float, float, float, float, float]:
    return (
        voltage.mean(),
        voltage.max(),
        voltage.min(),
        voltage.var(),
        scipy.stats.kurtosis(voltage),
        scipy.stats.skew(voltage),
    )


def calculate_capacity_distribution_features(
    group: pd.DataFrame,
    v_start: float,
    v_end: float,
) -> Tuple[float, float, float, float]:
    q_charge = (
        group["cumulative_capacity(mA.h)"].max()
        - group["cumulative_capacity(mA.h)"].min()
    )

    voltage_bins = np.arange(v_start, v_end + 0.001, 0.01)
    interval_vars = []

    for i in range(len(voltage_bins) - 1):
        interval_start = voltage_bins[i]
        interval_end = voltage_bins[i + 1]

        interval_data = group[
            (group["voltage(V)"] >= interval_start)
            & (group["voltage(V)"] <= interval_end)
        ]

        if len(interval_data) < 2:
            interval_vars.append(0)
        else:
            interval_capacities = interval_data["cumulative_capacity(mA.h)"]
            interval_vars.append(interval_capacities.var())

    capacity_array = np.array(interval_vars)

    return (
        q_charge,
        capacity_array.var(),
        scipy.stats.kurtosis(capacity_array, nan_policy="omit"),
        scipy.stats.skew(capacity_array, nan_policy="omit"),
    )


def calculate_voltage_rate_features(
    time: np.ndarray,
    voltage: np.ndarray,
) -> Tuple[float, float]:
    voltage_rate = np.diff(voltage) / np.diff(time)

    if len(voltage_rate) > 0:
        return np.mean(voltage_rate), np.var(voltage_rate)

    return 0.0, 0.0


def calculate_energy_features(
    time: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
) -> float:
    time_hours = time / 3600.0
    power = voltage * current
    return np.trapz(power, time_hours)


def process_voltage_segment(
    segment_data: pd.DataFrame,
    soh_data: Dict[int, float],
    v_start: float,
    v_end: float,
) -> pd.DataFrame:
    features = []
    segment_groups = segment_data.groupby("cycle_index")

    print(
        f"Voltage segment {v_start:.2f}-{v_end:.2f} V contains "
        f"{len(segment_groups)} cycles after abnormal-cycle removal."
    )

    for cycle, group in segment_groups:
        if len(group) < 3:
            print(
                f"Cycle {cycle} has insufficient data points "
                f"({len(group)}); skipped."
            )
            continue

        group = group.sort_values("relative_time(s)")

        voltage = group["voltage(V)"]
        time = group["relative_time(s)"].values
        current = group["current(A)"].values

        voltage_features = calculate_voltage_features(voltage)

        q_charge, q_var, q_kurtosis, q_skewness = (
            calculate_capacity_distribution_features(group, v_start, v_end)
        )

        voltage_rate_mean, voltage_rate_var = calculate_voltage_rate_features(
            time,
            voltage.values,
        )

        time_duration = (
            group["relative_time(s)"].max()
            - group["relative_time(s)"].min()
        )

        charging_energy = calculate_energy_features(
            time,
            voltage.values,
            current,
        )

        soh_value = soh_data.get(cycle, np.nan)

        if pd.isna(soh_value):
            print(f"Cycle {cycle} has no valid SOH value; skipped.")
            continue

        feature_line = [
            cycle,
            *voltage_features,
            q_charge,
            q_var,
            q_kurtosis,
            q_skewness,
            voltage_rate_mean,
            voltage_rate_var,
            time_duration,
            25.0,
            1,
            charging_energy,
            round(voltage.min(), 2),
            soh_value,
        ]

        features.append(feature_line)

    columns = [
        "cycle number",
        "voltage_mean",
        "voltage_max",
        "voltage_min",
        "voltage_var",
        "voltage_kurtosis",
        "voltage_skewness",
        "charge_capacity",
        "capacity_var",
        "capacity_kurtosis",
        "capacity_skewness",
        "voltage_rate_mean",
        "voltage_rate_var",
        "charge_time",
        "temperature",
        "rate",
        "charging_energy",
        "start_voltage",
        "SOH",
    ]

    print(
        f"Extracted {len(features)} feature records from "
        f"{v_start:.2f}-{v_end:.2f} V."
    )

    return pd.DataFrame(features, columns=columns)


def process_calce_battery_file(
    file_path: str,
    output_dir: str,
    voltage_segments: List[Tuple[float, float]],
    sigma: float = 3.0,
) -> None:
    file_name = os.path.basename(file_path)

    print("=" * 60)
    print(f"Processing CALCE battery file: {file_name}")
    print("=" * 60)

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print("File loaded successfully with UTF-8-SIG encoding.")
        print(f"Columns: {list(df.columns)}")
        print(f"Original row count: {len(df)}")
        print("First five rows:")
        print(df.head())
    except Exception as exc:
        print(f"Error: Failed to read {file_name}: {exc}")
        return

    required_columns = [
        "relative_time(s)",
        "voltage(V)",
        "current(A)",
        "cumulative_capacity(mA.h)",
        "cycle_index",
        "stage",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        print(
            f"Error: {file_name} is missing required columns: "
            f"{', '.join(missing_columns)}"
        )
        print(f"Available columns: {list(df.columns)}")

        df.columns = [column.strip() for column in df.columns]
        print(f"Columns after whitespace stripping: {list(df.columns)}")

        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            return

        print("All required columns are available after whitespace stripping.")

    df_charge = df[df["stage"].str.lower() == "charge"].copy()

    if df_charge.empty:
        df_charge = df[
            df["stage"].str.contains("charge", case=False, na=False)
        ].copy()

        if df_charge.empty:
            print(
                f"Error: {file_name} contains no charging-stage data; skipped."
            )
            return

    original_cycles = sorted(df_charge["cycle_index"].unique())

    print(
        f"Charging-stage cycles: {original_cycles[0]}-"
        f"{original_cycles[-1]} "
        f"({len(original_cycles)} cycles)"
    )

    print(
        f"Calculating SOH using nominal capacity "
        f"{CALCE_STANDARD_CAPACITY} mAh."
    )

    soh_dict = calculate_soh_calce(df_charge)
    df_charge["SOH"] = df_charge["cycle_index"].map(soh_dict)

    raw_soh_data = extract_soh_data(df_charge)

    if not raw_soh_data:
        print("Error: No valid SOH data are available; file processing stopped.")
        return

    abnormal_cycles = detect_abnormal_soh_cycles(
        raw_soh_data,
        sigma=sigma,
    )

    df_filtered = df_charge[
        ~df_charge["cycle_index"].isin(abnormal_cycles)
    ].reset_index(drop=True)

    filtered_rows = len(df_filtered)

    filtered_cycles = (
        sorted(df_filtered["cycle_index"].unique())
        if not df_filtered.empty
        else []
    )

    print("Abnormal-cycle removal summary:")
    print(
        f"Before removal: {len(original_cycles)} cycles, "
        f"{len(df_charge)} rows"
    )
    print(
        f"After removal: {len(filtered_cycles)} cycles, "
        f"{filtered_rows} rows"
    )

    if filtered_cycles:
        print(
            f"Remaining cycle range: "
            f"{filtered_cycles[0]}-{filtered_cycles[-1]}"
        )
    else:
        print(
            "Warning: No data remain after abnormal-cycle removal; "
            "file processing stopped."
        )
        return

    filtered_soh_data = extract_soh_data(df_filtered)

    print(
        f"Extracted SOH values for {len(filtered_soh_data)} "
        f"filtered cycles."
    )

    voltage_min = df_filtered["voltage(V)"].min()
    voltage_max = df_filtered["voltage(V)"].max()

    print(
        f"Filtered voltage range: "
        f"{voltage_min:.2f}-{voltage_max:.2f} V"
    )
    print("Filtered-data statistics:")
    print(
        f"Mean voltage: "
        f"{df_filtered['voltage(V)'].mean():.3f} V"
    )
    print(
        f"Mean current: "
        f"{df_filtered['current(A)'].mean():.3f} A"
    )
    print(
        f"Maximum capacity: "
        f"{df_filtered['cumulative_capacity(mA.h)'].max():.1f} mAh"
    )
    print(f"Mean SOH: {df_filtered['SOH'].mean():.3f}")

    output_files = []

    for v_start, v_end in voltage_segments:
        if v_end < voltage_min or v_start > voltage_max:
            print(
                f"Voltage segment {v_start:.2f}-{v_end:.2f} V "
                f"is outside the data range "
                f"{voltage_min:.2f}-{voltage_max:.2f} V; skipped."
            )
            continue

        v_start_str = f"{v_start:.2f}"
        v_end_str = f"{v_end:.2f}"

        print(
            f"Processing voltage segment: "
            f"{v_start_str}-{v_end_str} V"
        )

        segment_mask = (
            (df_filtered["voltage(V)"] >= v_start)
            & (df_filtered["voltage(V)"] <= v_end)
        )

        segment_data = df_filtered[segment_mask]

        if segment_data.empty:
            print(
                f"No data found for voltage segment "
                f"{v_start_str}-{v_end_str} V; skipped."
            )
            continue

        print(
            f"Voltage segment {v_start_str}-{v_end_str} V contains "
            f"{len(segment_data)} rows after abnormal-cycle removal."
        )

        features_df = process_voltage_segment(
            segment_data,
            filtered_soh_data,
            v_start,
            v_end,
        )

        if features_df.empty:
            print(
                f"No valid features extracted from "
                f"{v_start_str}-{v_end_str} V; output skipped."
            )
            continue

        battery_id = (
            file_name
            .replace("_detailed.csv", "")
            .replace(".csv", "")
        )

        output_file = (
            f"CALCE-battery_{battery_id}_"
            f"{v_start_str}-{v_end_str}V.csv"
        )

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)

        try:
            features_df.to_csv(output_path, index=False)
            print(
                f"Feature file saved successfully: {output_file}"
            )
            output_files.append(output_file)
        except Exception as exc:
            print(
                f"Error: Failed to save features for "
                f"{v_start_str}-{v_end_str} V: {exc}"
            )

    print(
        f"Completed {file_name}. "
        f"Generated {len(output_files)} feature files."
    )

    if output_files:
        print(f"Generated files: {output_files}")


def main() -> None:
    sigma = 3.0

    start_voltages = np.arange(3.60, 4.01, 0.05)
    voltage_segments = [
        (start_voltage, start_voltage + 0.10)
        for start_voltage in start_voltages
    ]

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("CALCE Battery Feature Extraction")
    print("=" * 60)
    print("Configuration:")
    print(f"Input directory: {INPUT_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DATA_DIR}")
    print(f"Number of input files: {len(FILE_NAMES)}")
    print(
        f"Number of voltage segments: {len(voltage_segments)} "
        f"({[f'{start:.2f}-{end:.2f}V' for start, end in voltage_segments]})"
    )
    print(
        f"SOH nominal capacity: "
        f"{CALCE_STANDARD_CAPACITY} mAh"
    )
    print(
        f"Abnormal-cycle criterion: absolute SOH difference between "
        f"adjacent cycles > {SOH_JUMP_THRESHOLD:.5f}"
    )
    print("=" * 60)

    for file_name in FILE_NAMES:
        file_path = os.path.join(INPUT_DATA_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"Input file not found: {file_name}; skipped.")
            continue

        process_calce_battery_file(
            file_path=file_path,
            output_dir=OUTPUT_DATA_DIR,
            voltage_segments=voltage_segments,
            sigma=sigma,
        )

    print("=" * 60)
    print("All CALCE battery files have been processed.")
    print(f"Extracted feature data directory: {OUTPUT_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()