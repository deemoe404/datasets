#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wind_datasets.registry import get_dataset_spec, list_dataset_ids  # noqa: E402
from wind_datasets.source_schema import build_source_schema_inventory  # noqa: E402


def _greenbyte_rule(source_asset: str, source_table: str, column: str) -> tuple[str, str, str, str, str]:
    del source_asset
    turbine_core = {
        "Date and time": ("keep", "", "timestamp", "all", "Timestamp key."),
        "Power (kW)": ("keep", "", "target_kw", "all", "Forecast target history."),
        "Power, Minimum (kW)": ("keep", "", "power_stats_history", "power_stats_history", "Target-derived history."),
        "Power, Maximum (kW)": ("keep", "", "power_stats_history", "power_stats_history", "Target-derived history."),
        "Power, Standard deviation (kW)": ("keep", "", "power_stats_history", "power_stats_history", "Target-derived history."),
        "Wind speed (m/s)": ("keep", "", "Wind speed (m/s)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Wind direction (°)": ("keep", "", "Wind direction (°)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Nacelle position (°)": ("keep", "", "Nacelle position (°)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Generator RPM (RPM)": ("keep", "", "Generator RPM (RPM)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Rotor speed (RPM)": ("keep", "", "Rotor speed (RPM)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Ambient temperature (converter) (°C)": ("keep", "", "Ambient temperature (converter) (°C)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Nacelle temperature (°C)": ("keep", "", "Nacelle temperature (°C)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Power factor (cosphi)": ("keep", "", "Power factor (cosphi)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Reactive power (kvar)": ("keep", "", "Reactive power (kvar)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Blade angle (pitch position) A (°)": ("keep", "", "Blade angle (pitch position) A (°)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Blade angle (pitch position) B (°)": ("keep", "", "Blade angle (pitch position) B (°)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
        "Blade angle (pitch position) C (°)": ("keep", "", "Blade angle (pitch position) C (°)", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage1|static_calendar_stage2", "Stage-1 covariate."),
    }
    pmu = {
        "Date and time": ("keep", "", "timestamp", "all", "Timestamp key."),
        "GMS Power (kW)": ("keep", "", "farm_pmu__gms_power_kw", "staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage2", "Stage-2 operations covariate."),
        "GMS Reactive power (kvar)": ("keep", "", "farm_pmu__gms_reactive_power_kvar", "staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage2", "Stage-2 operations covariate."),
        "GMS Current (A)": ("keep", "", "farm_pmu__gms_current_a", "staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime|static_calendar_stage2", "Stage-2 operations covariate."),
    }
    status = {
        "Timestamp start": ("keep", "", "event_start", "staged_past_covariates.stage3_regime", "Required to build regime event features."),
        "Timestamp end": ("keep", "", "event_end", "staged_past_covariates.stage3_regime", "Required to build regime event features."),
        "Status": ("keep", "", "status", "staged_past_covariates.stage3_regime", "Required to build regime event features."),
        "Code": ("keep", "", "code", "staged_past_covariates.stage3_regime", "Required to build regime event features."),
        "Service contract category": ("keep", "", "service_contract_category", "staged_past_covariates.stage3_regime", "Required to build regime event features."),
        "IEC category": ("keep", "", "iec_category", "staged_past_covariates.stage3_regime", "Required to build regime event features."),
    }
    wt_static = {
        "Title": ("keep", "", "turbine_static", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Turbine id mapping."),
        "Identity": ("keep", "", "turbine_static", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Source turbine key."),
        "Latitude": ("keep", "", "latitude", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static spatial feature."),
        "Longitude": ("keep", "", "longitude", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static spatial feature."),
        "Elevation (m)": ("keep", "", "elevation_m", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine metadata."),
        "Rated power (kW)": ("keep", "", "rated_power_kw", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine metadata."),
        "Hub Height (m)": ("keep", "", "hub_height_m", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine metadata."),
        "Rotor Diameter (m)": ("keep", "", "rotor_diameter_m", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine metadata."),
        "Manufacturer": ("keep", "", "manufacturer", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine metadata."),
        "Model": ("keep", "", "model", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine metadata."),
        "Country": ("keep", "", "country", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine metadata."),
        "Commercial Operations Date": ("keep", "", "commercial_operation_date", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine metadata."),
    }
    table_rules = {
        "Turbine_Data": turbine_core,
        "Device_Data_PMU": pmu,
        "Status": status,
        "WT_static": wt_static,
        "Device_Data_Grid_Meter": {"Date and time": ("keep", "", "", "", "Timestamp retained so dropped payloads still align cleanly.")},
        "static": {},
        "dataSignalMapping": {},
    }
    decision, mask_rule, canonical, protocols, notes = table_rules.get(source_table, {}).get(
        column,
        ("drop", "", "", "", "Not required by the supported farm task bundles."),
    )
    reason = "Retained by the supported farm-level feature protocols." if decision != "drop" else "Dropped before silver to avoid rebuilding unused raw columns."
    return decision, mask_rule, canonical, protocols, reason + " " + notes, ""


def _hill_rule(source_asset: str, source_table: str, column: str) -> tuple[str, str, str, str, str]:
    del source_asset
    keep_sets = {
        "Hill_of_Towie_turbine_metadata": {
            "Turbine Name",
            "Station ID",
            "Latitude",
            "Longitude",
            "Manufacturer",
            "Model",
            "Rated power (kW)",
            "Hub Height (m)",
            "Rotor Diameter (m)",
            "Country",
            "Commercial Operations Date",
        },
        "tblSCTurbine": {
            "TimeStamp",
            "StationId",
            "wtc_ActPower_mean",
            "wtc_ActPower_min",
            "wtc_ActPower_max",
            "wtc_ActPower_stddev",
            "wtc_ActPower_endvalue",
            "wtc_SecAnemo_mean",
            "wtc_PriAnemo_mean",
            "wtc_YawPos_mean",
            "wtc_GenRpm_mean",
            "wtc_MainSRpm_mean",
            "wtc_PitchRef_BladeA_mean",
            "wtc_PitchRef_BladeB_mean",
            "wtc_PitchRef_BladeC_mean",
            "wtc_TwrHumid_mean",
        },
        "tblSCTurGrid": {
            "TimeStamp",
            "StationId",
            "wtc_ActPower_mean",
            "wtc_ActPower_min",
            "wtc_ActPower_max",
            "wtc_ActPower_stddev",
            "wtc_ActPower_endvalue",
        },
        "tblSCTurFlag": {"TimeStamp", "StationId"},
        "tblGrid": {"TimeStamp", "ActivePower", "ReActivePower", "PowerFactor"},
        "tblGridScientific": {"TimeStamp"},
        "tblSCTurCount": {"TimeStamp", "StationId"},
        "tblSCTurDigiIn": {"TimeStamp", "StationId"},
        "tblSCTurDigiOut": {"TimeStamp", "StationId"},
        "tblSCTurIntern": {"TimeStamp", "StationId"},
        "tblSCTurPress": {"TimeStamp", "StationId", "wtc_HydPress_mean"},
        "tblSCTurTemp": {"TimeStamp", "StationId", "wtc_AmbieTmp_mean", "wtc_NacelTmp_mean", "wtc_GeOilTmp_mean"},
        "tblAlarmLog": {"TimeOn", "TimeOff", "StationNr", "Alarmcode"},
        "Hill_of_Towie_AeroUp_install_dates": {"Turbine", "First date of AeroUp works", "Last date of AeroUp works"},
        "ShutdownDuration": {"TimeStamp_StartFormat", "TurbineName", "ShutdownDuration"},
    }
    if column in keep_sets.get(source_table, set()):
        reason = "Retained by the supported farm-level feature protocols or core row alignment."
        return "keep", "", "", "", reason, ""
    return "drop", "", "", "", "Dropped before silver to avoid rebuilding unused raw columns.", ""


def _sdwpf_rule(source_asset: str, source_table: str, column: str) -> tuple[str, str, str, str, str]:
    del source_asset
    rules = {
        "sdwpf_main": {
            "TurbID": ("keep", "", "turbine_id", "all", "Structural turbine key."),
            "Day": ("keep", "", "day", "all", "Required to derive timestamps."),
            "Tmstamp": ("keep", "", "clock", "all", "Required to derive timestamps."),
            "Patv": ("keep", "", "target_kw", "all", "Forecast target."),
            "Wspd": ("keep", "", "Wspd", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-1 covariate."),
            "Wdir": ("keep+mask", "sdwpf_wdir_out_of_range", "Wdir", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-1 covariate with documented abnormal ranges."),
            "Etmp": ("keep", "", "Etmp", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-1 covariate."),
            "Itmp": ("keep", "", "Itmp", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-1 covariate."),
            "Ndir": ("keep+mask", "sdwpf_ndir_out_of_range", "Ndir", "staged_past_covariates.stage1_core|staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-1 covariate with documented abnormal ranges."),
            "Pab1": ("keep+mask", "sdwpf_pitch_gt_89", "Pab1", "staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-2 covariate with documented abnormal ranges."),
            "Pab2": ("keep+mask", "sdwpf_pitch_gt_89", "Pab2", "staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-2 covariate with documented abnormal ranges."),
            "Pab3": ("keep+mask", "sdwpf_pitch_gt_89", "Pab3", "staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-2 covariate with documented abnormal ranges."),
            "Prtv": ("keep", "", "Prtv", "staged_past_covariates.stage2_ops|staged_past_covariates.stage3_regime", "Stage-2 covariate."),
        },
        "sdwpf_location": {
            "TurbID": ("keep", "", "turbine_id", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static turbine mapping."),
            "x": ("keep", "", "coord_x", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static projected x coordinate."),
            "y": ("keep", "", "coord_y", "static_calendar|static_calendar_stage1|static_calendar_stage2", "Static projected y coordinate."),
        },
    }
    decision, mask_rule, canonical, protocols, note = rules.get(source_table, {}).get(
        column,
        ("drop", "", "", "", "Not required by the supported farm task bundles."),
    )
    reason = "Retained by the supported farm-level feature protocols." if decision != "drop" else "Dropped before silver to avoid rebuilding unused raw columns."
    return decision, mask_rule, canonical, protocols, reason + " " + note, ""


def _rule_for(dataset_id: str, source_asset: str, source_table: str, column: str) -> tuple[str, str, str, str, str, str]:
    if dataset_id in {"kelmarsh", "penmanshiel"}:
        return _greenbyte_rule(source_asset, source_table, column)
    if dataset_id == "hill_of_towie":
        return _hill_rule(source_asset, source_table, column)
    if dataset_id == "sdwpf_kddcup":
        return _sdwpf_rule(source_asset, source_table, column)
    raise ValueError(f"Unsupported dataset_id {dataset_id!r}.")


def main() -> int:
    output_dir = REPO_ROOT / "src" / "wind_datasets" / "data" / "source_column_policy"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_id in list_dataset_ids():
        spec = get_dataset_spec(dataset_id)
        inventory = list(build_source_schema_inventory(spec))
        if dataset_id == "hill_of_towie":
            inventory.append(
                {
                    "source_asset": "tblSCTurbine",
                    "source_table_or_file": "tblSCTurbine",
                    "source_column": "dup_label",
                }
            )
        inventory = sorted(
            inventory,
            key=lambda row: (str(row["source_asset"]), str(row["source_table_or_file"]), str(row["source_column"])),
        )
        output_path = output_dir / f"{dataset_id}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "source_asset",
                    "source_table_or_file",
                    "source_column",
                    "decision",
                    "mask_rule",
                    "canonical_outputs",
                    "required_for_protocols",
                    "reason",
                    "notes",
                ]
            )
            for row in inventory:
                decision, mask_rule, canonical_outputs, required_for_protocols, reason, notes = _rule_for(
                    dataset_id,
                    str(row["source_asset"]),
                    str(row["source_table_or_file"]),
                    str(row["source_column"]),
                )
                writer.writerow(
                    [
                        row["source_asset"],
                        row["source_table_or_file"],
                        row["source_column"],
                        decision,
                        mask_rule,
                        canonical_outputs,
                        required_for_protocols,
                        reason,
                        notes,
                    ]
                )
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
