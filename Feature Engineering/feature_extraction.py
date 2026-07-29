import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats

warnings.simplefilter("ignore")

INPUT_DATA_DIR = "path/to/interpolated_battery_data"
OUTPUT_DATA_DIR = "path/to/extracted_features"

FILE_NAMES = [
    "interp_CY25-1_1-#1.csv",
    "interp_CY25-1_1-#2.csv",
    "interp_CY25-1_1-#3.csv",
    "interp_CY25-1_1-#4.csv",
    "interp_CY25-1_1-#5.csv",
    "interp_CY25-1_1-#6.csv",
    "interp_CY25-1_1-#7.csv",
    "interp_CY25-1_1-#8.csv",
    "interp_CY25-1_1-#9.csv",
]

FIXED_CURRENT_A = 1.75
TEMPERATURE_C = 25.0
CHARGE_RATE_C = 1.0

START_VOLTAGES = np.arange(3.60, 4.01, 0.05)
VOLTAGE_SEGMENTS = [(start, start + 0.10) for start in START_VOLTAGES]

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)


def extract_soh_data(cc_mode: pd.DataFrame) -> Dict[int, float]:
    soh_data = {}

    for cycle, group in cc_mode.groupby("cycle number"):
        soh_values = group["SOH"].dropna().unique()

        if len(soh_values) > 0:
            soh_data[cycle] = soh_values[0]
        else:
            print(
                f"Warning: cycle {cycle} has no valid SOH value."
            )

    return soh_data


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
        group["Q charge/mA.h"].max()
        - group["Q charge/mA.h"].min()
    )

    voltage_bins = np.arange(
        v_start,
        v_end + 0.001,
        0.01,
    )

    interval_variances = []

    for i in range(len(voltage_bins) - 1):
        interval_start = voltage_bins[i]
        interval_end = voltage_bins[i + 1]

        interval_data = group[
            (group["voltage"] >= interval_start)
            & (group["voltage"] <= interval_end)
        ]

        if len(interval_data) < 2:
            interval_variances.append(0.0)
        else:
            interval_capacities = interval_data[
                "Q charge/mA.h"
            ]
            interval_variances.append(
                interval_capacities.var()
            )

    capacity_array = np.array(
        interval_variances,
        dtype=float,
    )

    return (
        q_charge,
        capacity_array.var(),
        scipy.stats.kurtosis(
            capacity_array,
            nan_policy="omit",
        ),
        scipy.stats.skew(
            capacity_array,
            nan_policy="omit",
        ),
    )


def calculate_voltage_rate_features(
    time: np.ndarray,
    voltage: np.ndarray,
) -> Tuple[float, float]:
    voltage_rate = (
        np.diff(voltage)
        / np.diff(time)
    )

    if len(voltage_rate) > 0:
        return (
            float(np.mean(voltage_rate)),
            float(np.var(voltage_rate)),
        )

    return 0.0, 0.0


def calculate_energy_features(
    time: np.ndarray,
    voltage: np.ndarray,
) -> float:
    time_hours = time / 3600.0
    power = voltage * FIXED_CURRENT_A
    return float(
        np.trapz(
            power,
            time_hours,
        )
    )


def process_voltage_segment(
    segment_data: pd.DataFrame,
    soh_data: Dict[int, float],
    v_start: float,
    v_end: float,
) -> pd.DataFrame:
    features = []
    segment_groups = segment_data.groupby(
        "cycle number"
    )

    print(
        f"Voltage segment {v_start:.2f}-{v_end:.2f} V contains "
        f"{len(segment_groups)} cycles."
    )

    for cycle, group in segment_groups:
        if len(group) < 3:
            print(
                f"Cycle {cycle} contains only {len(group)} data points "
                "and is skipped."
            )
            continue

        group = group.sort_values(
            "time/s"
        )

        voltage = group["voltage"]
        time = group[
            "time/s"
        ].to_numpy(dtype=float)

        voltage_features = (
            calculate_voltage_features(
                voltage
            )
        )

        (
            q_charge,
            q_variance,
            q_kurtosis,
            q_skewness,
        ) = calculate_capacity_distribution_features(
            group,
            v_start,
            v_end,
        )

        (
            voltage_rate_mean,
            voltage_rate_variance,
        ) = calculate_voltage_rate_features(
            time,
            voltage.to_numpy(dtype=float),
        )

        time_duration = (
            group["time/s"].max()
            - group["time/s"].min()
        )

        charging_energy = (
            calculate_energy_features(
                time,
                voltage.to_numpy(dtype=float),
            )
        )

        soh_value = soh_data.get(
            cycle,
            np.nan,
        )

        if pd.isna(soh_value):
            print(
                f"Cycle {cycle} has no valid SOH value and is skipped."
            )
            continue

        feature_row = [
            cycle,
            *voltage_features,
            q_charge,
            q_variance,
            q_kurtosis,
            q_skewness,
            voltage_rate_mean,
            voltage_rate_variance,
            time_duration,
            TEMPERATURE_C,
            CHARGE_RATE_C,
            charging_energy,
            round(
                voltage.min(),
                2,
            ),
            soh_value,
        ]

        features.append(
            feature_row
        )

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
        f"Extracted {len(features)} feature rows from "
        f"{v_start:.2f}-{v_end:.2f} V."
    )

    return pd.DataFrame(
        features,
        columns=columns,
    )


def process_battery_file(
    file_path: str,
    voltage_segments: List[Tuple[float, float]],
) -> None:
    file_name = os.path.basename(
        file_path
    )

    print(
        f"Processing battery file: {file_name}"
    )

    try:
        df = pd.read_csv(
            file_path
        )
        print(
            f"Loaded {len(df)} rows."
        )

    except Exception as error:
        print(
            f"Error reading {file_name}: {error}"
        )
        return

    cc_mode = df

    required_columns = [
        "cycle number",
        "SOH",
        "voltage",
        "time/s",
        "Q charge/mA.h",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in cc_mode.columns
    ]

    if missing_columns:
        print(
            f"Error: {file_name} is missing required columns: "
            f"{', '.join(missing_columns)}"
        )
        return

    voltage_min = cc_mode[
        "voltage"
    ].min()

    voltage_max = cc_mode[
        "voltage"
    ].max()

    print(
        f"Voltage range: {voltage_min:.2f}-{voltage_max:.2f} V"
    )

    print(
        f"Cycle range: "
        f"{cc_mode['cycle number'].min()}-"
        f"{cc_mode['cycle number'].max()}"
    )

    soh_data = extract_soh_data(
        cc_mode
    )

    print(
        f"Extracted SOH values for "
        f"{len(soh_data)} cycles."
    )

    for v_start, v_end in voltage_segments:
        if (
            v_end < voltage_min
            or v_start > voltage_max
        ):
            print(
                f"Voltage segment {v_start:.2f}-{v_end:.2f} V "
                "is outside the available voltage range and is skipped."
            )
            continue

        v_start_str = f"{v_start:.2f}"
        v_end_str = f"{v_end:.2f}"

        print(
            f"Processing voltage segment: "
            f"{v_start_str}-{v_end_str} V"
        )

        segment_mask = (
            (cc_mode["voltage"] >= v_start)
            & (cc_mode["voltage"] <= v_end)
        )

        segment_data = cc_mode[
            segment_mask
        ]

        if segment_data.empty:
            print(
                f"No data found in voltage segment "
                f"{v_start_str}-{v_end_str} V."
            )
            continue

        print(
            f"Voltage segment {v_start_str}-{v_end_str} V contains "
            f"{len(segment_data)} rows."
        )

        features_df = (
            process_voltage_segment(
                segment_data,
                soh_data,
                v_start,
                v_end,
            )
        )

        if features_df.empty:
            print(
                f"No valid features were extracted from "
                f"{v_start_str}-{v_end_str} V."
            )
            continue

        battery_id = (
            file_name
            .split("#")[1]
            .split(".")[0]
        )

        output_file = (
            f"C-battery_CY25-1_1-#{battery_id}_"
            f"{v_start_str}-{v_end_str}V.csv"
        )

        output_path = os.path.join(
            OUTPUT_DATA_DIR,
            output_file,
        )

        try:
            features_df.to_csv(
                output_path,
                index=False,
            )

            print(
                f"Saved features: {output_file}"
            )

        except Exception as error:
            print(
                f"Error saving {output_file}: {error}"
            )

    print(
        f"Completed all voltage segments for "
        f"{file_name}."
    )


def main() -> None:
    print("=" * 60)
    print("Battery Feature Extraction")
    print("=" * 60)
    print(f"Input directory: {INPUT_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DATA_DIR}")
    print(f"Number of input files: {len(FILE_NAMES)}")
    print(f"Number of voltage segments: {len(VOLTAGE_SEGMENTS)}")
    print("=" * 60)

    for file_name in FILE_NAMES:
        file_path = os.path.join(
            INPUT_DATA_DIR,
            file_name,
        )

        if not os.path.exists(
            file_path
        ):
            print(
                f"File not found: {file_name}. Skipping."
            )
            continue

        process_battery_file(
            file_path,
            VOLTAGE_SEGMENTS,
        )

    print("=" * 60)
    print("All battery files have been processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()