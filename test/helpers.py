from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import polars as pl

from wind_datasets.models import DatasetSpec, OfficialRelease


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def build_greenbyte_fixture(
    root: Path,
    dataset_name: str,
    turbine_name: str,
    file_end: str = "2025-01-01",
) -> DatasetSpec:
    static_name = f"{dataset_name}_WT_static.csv"
    scada_dir = root / f"{dataset_name}_SCADA_2024_0001"
    pmu_dir = root / f"{dataset_name}_PMU_2024_0002"
    grid_dir = root / f"{dataset_name}_Grid_Meter_2024_0003"
    turbine_file_stem = f"Turbine_Data_{turbine_name.replace(' ', '_')}_2024-01-01_-_{file_end}_0001"
    status_file_stem = f"Status_{turbine_name.replace(' ', '_')}_2024-01-01_-_{file_end}_0001"
    _write_text(
        root / static_name,
        f"""
        Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),Hub Height (m),Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,Commercial Operations Date
        {dataset_name},{turbine_name},{turbine_name},ID-1,Senvion,MM82,2050,78.5,92,52.4,-0.94,140,UK,2016-01-01
        """,
    )
    _write_text(
        scada_dir / f"{turbine_file_stem}.csv",
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
        scada_dir / f"{status_file_stem}.csv",
        f"""
        # Status export
        #
        # Turbine: {turbine_name}
        Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category,Global contract category,Custom contract category
        2024-01-01 00:00:00,2024-01-01 00:10:00,00:10:00,Stop,710,Sample stop,,Operating states,Technical Standby,,
        2024-01-01 00:10:00,2024-01-01 00:20:00,00:10:00,Informational,0,System OK,,System OK,Full Performance,,
        2024-01-01 00:20:00,2024-01-01 00:30:00,00:10:00,Warning,410,Sample warning,,Operating states,Partial Performance,,
        """,
    )
    _write_text(
        pmu_dir / f"Device_Data_{dataset_name}_PMU_2024-01-01_-_{file_end}_0002.csv",
        f"""
        # This file was exported by Greenbyte.
        #
        # Device: {dataset_name} PMU
        # Device type: Production Meter
        # Time zone: UTC
        # Date and time,GMS Power (kW),GMS Power setpoint (kW),GMS Grid frequency (Hz)
        2024-01-01 00:00:00,900,950,50.0
        2024-01-01 00:10:00,910,955,50.0
        2024-01-01 00:20:00,920,960,49.9
        2024-01-01 00:30:00,930,965,49.9
        """,
    )
    _write_text(
        grid_dir / f"Device_Data_{dataset_name}_Grid_Meter_2024-01-01_-_{file_end}_0003.csv",
        f"""
        # This file was exported by Greenbyte.
        #
        # Device: {dataset_name} Grid Meter
        # Device type: Grid Meter
        # Time zone: UTC
        # Date and time,Grid Meter Energy Export (kWh),Grid Meter Net Energy (kWh),Grid Meter Power factor (tanphi)
        2024-01-01 00:00:00,100,100,0.95
        2024-01-01 00:10:00,101,101,0.96
        2024-01-01 00:20:00,102,102,0.97
        2024-01-01 00:30:00,103,103,0.98
        """,
    )
    _write_text(
        grid_dir / f"Status_{dataset_name}_Grid_Meter_2024-01-01_-_{file_end}_0003.csv",
        f"""
        # Status export
        #
        # {dataset_name} Grid Meter production: NaN kWh
        Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category,Global contract category,Custom contract category
        2024-01-01 00:10:00,2024-01-01 00:20:00,00:10:00,Communication,9997,Data communication unavailable,,External conditions,System Warning,,
        """,
    )
    if dataset_name.lower() == "kelmarsh":
        releases = (
            OfficialRelease(
                release_id="legacy_2022",
                source_url="https://zenodo.org/records/5841834",
                published_date="2022-02-01",
                coverage_start="2016-01-03",
                coverage_end="2021-06-30",
            ),
            OfficialRelease(
                release_id="extended_2025",
                source_url="https://zenodo.org/records/16807551",
                published_date="2025-08-12",
                coverage_start="2016-01-03",
                coverage_end="2024-12-31",
            ),
        )
    else:
        releases = (
            OfficialRelease(
                release_id="legacy_2022",
                source_url="https://zenodo.org/records/5946808",
                published_date="2022-02-07",
                coverage_start="2016-06-02",
                coverage_end="2021-06-30",
            ),
            OfficialRelease(
                release_id="extended_2025",
                source_url="https://zenodo.org/records/16807304",
                published_date="2025-08-13",
                coverage_start="2016-06-02",
                coverage_end="2024-12-31",
            ),
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
        official_name=f"{dataset_name} wind farm data",
        official_releases=releases,
        default_expected_release_id="extended_2025",
        requires_pre_extracted_sources=True,
        official_assets=("turbine_static", "turbine_scada", "status_events", "signal_mapping"),
        default_ingested_assets=("turbine_static", "turbine_scada", "status_events"),
        default_excluded_assets=("pmu_meter", "grid_meter"),
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
        2024-01-01 01:00:00,1001,8.2
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
    _write_text(
        root / "2024" / "tblGrid_2024_01.csv",
        """
        TimeStamp,Station,WPSStatus,TimestampStation,CurrentL1,CurrentL2,CurrentL3,VoltageL1,VoltageL2,VoltageL3,ActivePower,ReActivePower,ActivePowerExport,ReActivePowerExport,ActivePowerImport,ReActivePowerImport,DigitalInputBlock1,DigitalInputBlock2,DigitalInputBlock3,DigitalInputBlock4,GridTHD,PowerFactor,Frequency
        2024-01-01 00:00:00,Hill,1,2024-01-01 00:00:00,10,10,10,400,401,402,2000,100,2000,100,0,0,1,0,0,0,2,0.98,50
        2024-01-01 00:10:00,Hill,1,2024-01-01 00:10:00,11,11,11,401,402,403,2010,101,2010,101,0,0,1,0,0,0,2,0.98,50
        2024-01-01 00:20:00,Hill,1,2024-01-01 00:20:00,12,12,12,402,403,404,2020,102,2020,102,0,0,1,0,0,0,2,0.98,50
        2024-01-01 00:30:00,Hill,1,2024-01-01 00:30:00,13,13,13,403,404,405,2030,103,2030,103,0,0,1,0,0,0,2,0.98,50
        2024-01-01 00:40:00,Hill,1,2024-01-01 00:40:00,14,14,14,404,405,406,2040,104,2040,104,0,0,1,0,0,0,2,0.98,50
        2024-01-01 00:50:00,Hill,1,2024-01-01 00:50:00,15,15,15,405,406,407,2050,105,2050,105,0,0,1,0,0,0,2,0.98,50
        2024-01-01 01:00:00,Hill,1,2024-01-01 01:00:00,16,16,16,406,407,408,2060,106,2060,106,0,0,1,0,0,0,2,0.98,50
        """,
    )
    _write_text(
        root / "2024" / "tblGridScientific_2024_01.csv",
        """
        TimeStamp,Station,WPSStatus,DataOk,ActiveEnergy,ReActiveEnergy,ActivePowerMin,ActivePowerMax,ActivePowerMean,ActivePowerSD,ReActivePowerMin,ReActivePowerMax,ReActivePowerMean,ReActivePowerSD,VoltageMin,VoltageMax,VoltageMean,VoltageSD,CurrentMin,CurrentMax,CurrentMean,CurrentSD,Energy
        2024-01-01 00:00:00,Hill,1,1,20,1,1900,2100,2000,10,90,110,100,5,399,402,400,1,9,11,10,0.5,20
        2024-01-01 00:10:00,Hill,1,1,21,1,1910,2110,2010,10,91,111,101,5,400,403,401,1,10,12,11,0.5,21
        2024-01-01 00:20:00,Hill,1,1,22,1,1920,2120,2020,10,92,112,102,5,401,404,402,1,11,13,12,0.5,22
        2024-01-01 00:30:00,Hill,1,1,23,1,1930,2130,2030,10,93,113,103,5,402,405,403,1,12,14,13,0.5,23
        2024-01-01 00:40:00,Hill,1,1,24,1,1940,2140,2040,10,94,114,104,5,403,406,404,1,13,15,14,0.5,24
        2024-01-01 00:50:00,Hill,1,1,25,1,1950,2150,2050,10,95,115,105,5,404,407,405,1,14,16,15,0.5,25
        2024-01-01 01:00:00,Hill,1,1,26,1,1960,2160,2060,10,96,116,106,5,405,408,406,1,15,17,16,0.5,26
        """,
    )
    for table_name, value_column, value_base in (
        ("tblSCTurCount", "wtc_TurRdHrT_endvalue", 100),
        ("tblSCTurDigiIn", "wtc_SmokeNac_counts", 1),
        ("tblSCTurDigiOut", "wtc_TurbinOK_counts", 5),
        ("tblSCTurIntern", "wtc_ValSuppV_mean", 230),
        ("tblSCTurPress", "wtc_BrakPres_mean", 20),
        ("tblSCTurTemp", "wtc_AmbieTmp_mean", 8),
    ):
        _write_text(
            root / "2024" / f"{table_name}_2024_01.csv",
            f"""
            TimeStamp,StationId,{value_column}
            2024-01-01 00:00:00,1001,{value_base}
            2024-01-01 00:00:00,1002,{value_base + 1}
            2024-01-01 00:10:00,1001,{value_base + 2}
            2024-01-01 00:10:00,1002,{value_base + 3}
            2024-01-01 00:20:00,1001,{value_base + 4}
            2024-01-01 00:20:00,1002,{value_base + 5}
            2024-01-01 00:30:00,1001,{value_base + 6}
            2024-01-01 00:30:00,1002,{value_base + 7}
            2024-01-01 00:40:00,1001,{value_base + 8}
            2024-01-01 00:40:00,1002,{value_base + 9}
            2024-01-01 00:50:00,1001,{value_base + 10}
            2024-01-01 00:50:00,1002,{value_base + 11}
            2024-01-01 01:00:00,1001,{value_base + 12}
            2024-01-01 01:00:00,1002,{value_base + 13}
            """,
        )
    _write_text(
        root / "2024" / "tblAlarmLog_2024_01.csv",
        """
        TimeOn,TimeOff,StationNr,Alarmcode
        2024-01-01 00:15:00,2024-01-01 00:35:00,1001,42
        2024-01-01 00:40:00,2024-01-01 00:50:00,1002,99
        """,
    )
    _write_text(
        root / "ShutdownDuration.csv",
        """
        TimeStamp_StartFormat,TurbineName,ShutdownDuration
        2024-01-01 00:20:00,T01,600
        2024-01-01 00:20:00,T02,0
        2024-01-01 00:30:00,T01,300
        2024-01-01 00:30:00,T02,120
        """,
    )
    _write_text(
        root / "Hill_of_Towie_AeroUp_install_dates.csv",
        """
        Turbine,First date of AeroUp works,Last date of AeroUp works
        T01,2023-12-31,2024-01-01
        T02,2024-01-01,2024-01-02
        """,
    )
    return DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=root,
        resolution_minutes=10,
        turbine_ids=("T01", "T02"),
        target_column="wtc_ActPower_mean",
        target_unit="kW",
        timezone_policy="utc_documented",
        timestamp_convention="source_utc_naive_interval_end",
        default_feature_groups=("tblSCTurbine", "tblSCTurGrid", "tblSCTurFlag"),
        handler="hill_of_towie",
        official_name="Hill of Towie wind farm open dataset",
        official_releases=(
            OfficialRelease(
                release_id="v1_2025",
                source_url="https://zenodo.org/records/14870023",
                published_date="2025-03-28",
                coverage_start="2016-01-01",
                coverage_end="2024-08-31",
                notes="UTC timestamps. 10-minute timestamps denote interval end.",
            ),
        ),
        default_expected_release_id="v1_2025",
        requires_pre_extracted_sources=True,
        official_assets=("turbine_metadata", "tblSCTurbine", "tblSCTurGrid", "tblSCTurFlag"),
        default_ingested_assets=("turbine_metadata", "tblSCTurbine", "tblSCTurGrid", "tblSCTurFlag"),
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
        timezone_policy="utc_plus_8_documented",
        timestamp_convention="source_local_naive_utc_plus_8",
        default_feature_groups=("main",),
        handler="sdwpf_full",
        default_quality_profile="official_v1",
        official_name="SDWPF_full",
        official_releases=(
            OfficialRelease(
                release_id="scientific_data_2024",
                source_url="https://www.nature.com/articles/s41597-024-03427-5",
                published_date="2024-06-24",
                coverage_start="2020-01-01",
                coverage_end="2021-12-31",
            ),
        ),
        default_expected_release_id="scientific_data_2024",
        official_assets=("main_parquet", "turbine_location_elevation"),
        default_ingested_assets=("main_parquet", "turbine_location_elevation"),
    )
