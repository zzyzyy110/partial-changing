import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

INPUT_DATA_DIR = "path/to/battery_raw_data"
OUTPUT_DATA_DIR = "path/to/interpolated_battery_data"

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

FILE_NAMES = [
    "CY35-05_1-#1.csv",
    "CY35-05_1-#2.csv",
    "CY35-05_1-#3.csv",
]

TARGET_VOLTAGE = np.round(np.arange(3.60, 4.101, 0.01), 5)
REFERENCE_CAPACITY_MAH = 3500.0

for file_name in FILE_NAMES:
    file_path = os.path.join(INPUT_DATA_DIR, file_name)
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")

    cycle_max_capacities = {}
    for cycle_num, group in df.groupby("cycle number"):
        cycle_max_capacities[cycle_num] = group["Q charge/mA.h"].max()

    cc_charge = df[
        (df["control/V"] == 0) &
        (df["control/mA"] > 0)
    ].copy()

    all_cycles = []

    for cycle_num, group in cc_charge.groupby("cycle number"):
        group = group.sort_values("Ecell/V")
        group = group.drop_duplicates(subset="Ecell/V", keep="last")
        group["Ecell/V"] = group["Ecell/V"].round(5)

        if len(group) < 2:
            continue

        if group["Ecell/V"].min() > 3.60 or group["Ecell/V"].max() < 4.10:
            continue

        cycle_max_capacity = cycle_max_capacities.get(cycle_num, 0.0)
        soh = (
            cycle_max_capacity / REFERENCE_CAPACITY_MAH
            if REFERENCE_CAPACITY_MAH > 0
            else 0.0
        )

        time_interp = interp1d(
            group["Ecell/V"],
            group["time/s"],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        qcharge_interp = interp1d(
            group["Ecell/V"],
            group["Q charge/mA.h"],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        interp_time = time_interp(TARGET_VOLTAGE)
        interp_qcharge = qcharge_interp(TARGET_VOLTAGE)

        interp_df = pd.DataFrame(
            {
                "cycle number": cycle_num,
                "voltage": TARGET_VOLTAGE,
                "time/s": interp_time,
                "Q charge/mA.h": interp_qcharge,
                "source": "interpolation",
                "SOH": soh,
            }
        )

        original_data = group[
            (group["Ecell/V"] >= 3.60) &
            (group["Ecell/V"] <= 4.10)
        ].copy()

        original_df = pd.DataFrame(
            {
                "cycle number": cycle_num,
                "voltage": original_data["Ecell/V"],
                "time/s": original_data["time/s"],
                "Q charge/mA.h": original_data["Q charge/mA.h"],
                "source": "original",
                "SOH": soh,
            }
        )

        combined_df = pd.concat(
            [interp_df, original_df],
            ignore_index=True,
        )

        combined_df = combined_df.sort_values(
            ["voltage", "source"]
        )

        combined_df = combined_df.drop_duplicates(
            subset=["voltage"],
            keep="last",
        )

        combined_df = combined_df.sort_values("voltage")
        all_cycles.append(combined_df)

    if all_cycles:
        result_df = pd.concat(all_cycles, ignore_index=True)

        output_name = "interp_" + file_name
        output_path = os.path.join(OUTPUT_DATA_DIR, output_name)

        result_df.to_csv(
            output_path,
            index=False,
            float_format="%.5f",
        )

        print(f"Completed: {file_name} -> {output_name}")
    else:
        print(f"No valid data found: {file_name}")