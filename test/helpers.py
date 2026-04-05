from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import polars as pl

from wind_datasets.models import DatasetSpec


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def build_greenbyte_fixture(root: Path, dataset_name: str, turbine_name: str) -> DatasetSpec:
    static_name = f"{dataset_name}_WT_static.csv"
    scada_dir = root / f"{dataset_name}_SCADA_2024_0001"
    _write_text(
        root / static_name,
        f"""
        Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),Hub Height (m),Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,Commercial Operations Date
        {dataset_name},{turbine_name},{turbine_name},ID-1,Senvion,MM82,2050,78.5,92,52.4,-0.94,140,UK,2016-01-01
        """,
    )
    _write_text(
        scada_dir / f"Turbine_Data_{turbine_name.replace(' ', '_')}_sample.csv",
        f"""
        # This file was exported by Greenbyte.
        #
        # Turbine: {turbine_name}
        # Time zone: UTC
        # Date and time,Power (kW),Wind speed (m/s),Wind direction (°),Nacelle position (°),Generator RPM (RPM),Rotor speed (RPM),Ambient temperature (converter) (°C),Nacelle temperature (°C),Grid frequency (Hz),Blade angle (pitch position) A (°),Blade angle (pitch position) B (°),Blade angle (pitch position) C (°)
        2024-01-01 00:00:00,100,8.0,180,174,1500,12,5,10,50,1,1,
        2024-01-01 00:00:00,,8.2,,,,,,,,,,0.9
        2024-01-01 00:10:00,110,8.4,181,175,1510,12.1,5.1,10.2,50,1,1,1
        2024-01-01 00:20:00,120,8.8,182,176,1520,12.2,5.2,10.4,50,1,1,1
        2024-01-01 00:30:00,NaN,9.1,183,177,1530,12.3,5.3,10.6,50,1,1,1
        2024-01-01 00:50:00,140,9.5,184,178,1540,12.4,5.4,10.8,50,1,1,1
        2024-01-01 01:00:00,150,9.8,185,179,1550,12.5,5.5,11.0,50,1,1,1
        2024-01-01 01:10:00,160,10.0,186,180,1560,12.6,5.6,11.2,50,1,1,1
        2024-01-01 01:20:00,170,10.2,187,181,1570,12.7,5.7,11.4,50,1,1,1
        """,
    )
    _write_text(
        scada_dir / f"Status_{turbine_name.replace(' ', '_')}_sample.csv",
        f"""
        # Status export
        #
        # Turbine: {turbine_name}
        Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category,Global contract category,Custom contract category
        2024-01-01 00:00:00,2024-01-01 00:05:00,00:05:00,Stop,710,Sample stop,,Operating states,Technical Standby,,
        2024-01-01 00:05:00,-,-,Informational,0,System OK,,System OK,Full Performance,,
        """,
    )
    return DatasetSpec(
        dataset_id=dataset_name.lower(),
        source_root=root,
        resolution_minutes=10,
        turbine_ids=(turbine_name,),
        target_column="Power (kW)",
        target_unit="kW",
        timezone_policy="utc_documented",
        timestamp_convention="source_utc_naive",
        default_feature_groups=("continuous_main",),
        handler="greenbyte",
    )


def build_hill_fixture(root: Path) -> DatasetSpec:
    _write_text(
        root / "Hill_of_Towie_turbine_metadata.csv",
        """
        Wind Farm,Turbine Name,Latitude,Longitude,Station ID,Manufacturer,Model,Rated power (kW),Hub Height (m),Rotor Diameter (m),Country,Commercial Operations Date
        Hill of Towie,T01,57.5,-3.0,1001,Siemens,SWT,2300,59,82,UK,2012-05-12
        Hill of Towie,T02,57.6,-3.1,1002,Siemens,SWT,2300,59,82,UK,2012-05-12
        """,
    )
    _write_text(
        root / "2024" / "tblSCTurbine_2024_01.csv",
        """
        TimeStamp,StationId,wtc_PrWindSp_mean
        2024-01-01 00:00:00,1001,7.0
        2024-01-01 00:00:00,1002,6.5
        2024-01-01 00:10:00,1001,7.2
        2024-01-01 00:10:00,1002,6.7
        2024-01-01 00:20:00,1001,7.4
        2024-01-01 00:20:00,1002,6.8
        2024-01-01 00:30:00,1001,7.6
        2024-01-01 00:30:00,1002,7.0
        2024-01-01 00:40:00,1001,7.8
        2024-01-01 00:40:00,1002,7.2
        2024-01-01 00:50:00,1001,8.0
        2024-01-01 00:50:00,1002,7.4
        2024-01-01 01:00:00,1001,8.1
        2024-01-01 01:00:00,1001,8.1
        2024-01-01 01:00:00,1002,7.5
        """,
    )
    _write_text(
        root / "2024" / "tblSCTurGrid_2024_01.csv",
        """
        TimeStamp,StationId,wtc_ActPower_mean
        2024-01-01 00:00:00,1001,1000
        2024-01-01 00:00:00,1002,950
        2024-01-01 00:10:00,1001,1010
        2024-01-01 00:10:00,1002,960
        2024-01-01 00:20:00,1001,1020
        2024-01-01 00:30:00,1001,1030
        2024-01-01 00:30:00,1002,980
        2024-01-01 00:40:00,1001,1040
        2024-01-01 00:40:00,1002,990
        2024-01-01 00:50:00,1001,1050
        2024-01-01 00:50:00,1002,1000
        2024-01-01 01:00:00,1001,1060
        2024-01-01 01:00:00,1001,1060
        2024-01-01 01:00:00,1002,1010
        """,
    )
    _write_text(
        root / "2024" / "tblSCTurFlag_2024_01.csv",
        """
        TimeStamp,StationId,wtc_ScInOper_timeon
        2024-01-01 00:00:00,1001,600
        2024-01-01 00:00:00,1002,600
        2024-01-01 00:10:00,1001,600
        2024-01-01 00:10:00,1002,600
        2024-01-01 00:20:00,1001,600
        2024-01-01 00:20:00,1002,600
        2024-01-01 00:30:00,1001,600
        2024-01-01 00:30:00,1002,600
        2024-01-01 00:40:00,1001,600
        2024-01-01 00:40:00,1002,600
        2024-01-01 00:50:00,1001,600
        2024-01-01 00:50:00,1002,600
        2024-01-01 01:00:00,1001,600
        2024-01-01 01:00:00,1001,600
        2024-01-01 01:00:00,1002,600
        """,
    )
    return DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=root,
        resolution_minutes=10,
        turbine_ids=("T01", "T02"),
        target_column="wtc_ActPower_mean",
        target_unit="kW",
        timezone_policy="unknown_unverified",
        timestamp_convention="source_local_or_naive",
        default_feature_groups=("tblSCTurbine", "tblSCTurGrid", "tblSCTurFlag"),
        handler="hill_of_towie",
    )


def build_sdwpf_fixture(root: Path) -> DatasetSpec:
    (root / "sdwpf_full").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "TurbID": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "Tmstamp": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:10:00",
                "2024-01-01 00:20:00",
                "2024-01-01 00:40:00",
                "2024-01-01 00:50:00",
                "2024-01-01 01:00:00",
                "2024-01-01 00:00:00",
                "2024-01-01 00:10:00",
                "2024-01-01 00:20:00",
                "2024-01-01 00:40:00",
                "2024-01-01 00:50:00",
                "2024-01-01 01:00:00",
            ],
            "Wspd": [6.0, 2.0, 6.5, 7.0, 7.2, 7.4, 5.8, 6.0, 6.1, 6.4, 6.7, 6.9],
            "Wdir": [180, 180, 180, 180, 180, 200, 170, 170, 170, 170, 170, 170],
            "Ndir": [0, 0, 0, 0, 800, 0, 5, 5, 5, 5, 5, 5],
            "Pab1": [10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12],
            "Pab2": [10, 10, 10, 95, 10, 10, 12, 12, 12, 12, 12, 12],
            "Pab3": [10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12],
            "Patv": [500, -5, 0, 560, 580, 590, 450, 460, 470, 490, 510, 520],
        }
    ).write_parquet(root / "sdwpf_full" / "sdwpf_2001_2112_full.parquet")
    _write_text(
        root / "sdwpf_full" / "sdwpf_turb_location_elevation.csv",
        """
        TurbID,x,y,Ele
        1,0,0,100
        2,1,1,101
        """,
    )
    return DatasetSpec(
        dataset_id="sdwpf_full",
        source_root=root / "sdwpf_full",
        resolution_minutes=10,
        turbine_ids=("1", "2"),
        target_column="Patv",
        target_unit="kW",
        timezone_policy="unknown_unverified",
        timestamp_convention="source_naive",
        default_feature_groups=("main",),
        handler="sdwpf_full",
        default_quality_profile="official_v1",
    )
