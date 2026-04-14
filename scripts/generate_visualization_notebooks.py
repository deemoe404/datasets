#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Sequence

from wind_datasets.registry import get_dataset_spec, list_dataset_ids
from wind_datasets.visualization import (
    ProtocolNotebookMetadata,
    list_supported_feature_protocol_ids_for_dataset,
    list_unsupported_feature_protocol_ids_for_dataset,
    load_protocol_notebook_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src"
DEFAULT_CACHE_ROOT = REPO_ROOT / "cache"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one visualization notebook per dataset.")
    parser.add_argument(
        "dataset_ids",
        nargs="*",
        default=list(list_dataset_ids()),
        help="Dataset ids to generate. Defaults to all registered datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated notebooks will be written.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="Cache directory used to read or lazily build task metadata.",
    )
    return parser.parse_args(argv)


def dataset_notebook_path(dataset_id: str, *, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(output_dir) / f"visualization_{dataset_id}.ipynb"


def collect_dataset_protocol_metadata(
    dataset_id: str,
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> tuple[tuple[ProtocolNotebookMetadata, ...], tuple[str, ...]]:
    protocol_metadata = tuple(
        load_protocol_notebook_metadata(
            dataset_id,
            feature_protocol_id,
            cache_root=cache_root,
        )
        for feature_protocol_id in list_supported_feature_protocol_ids_for_dataset(dataset_id)
    )
    omitted_protocol_ids = list_unsupported_feature_protocol_ids_for_dataset(dataset_id)
    return protocol_metadata, omitted_protocol_ids


def build_dataset_notebook_payload(
    dataset_id: str,
    *,
    official_name: str,
    task_id: str,
    protocol_metadata: Sequence[ProtocolNotebookMetadata],
    omitted_protocol_ids: Sequence[str] = (),
) -> dict[str, object]:
    cells: list[dict[str, object]] = [
        _markdown_cell(_intro_markdown(official_name, dataset_id, task_id, omitted_protocol_ids)),
        _code_cell(_setup_code(dataset_id, task_id)),
        _markdown_cell(
            textwrap.dedent(
                """
                ## Site Layout

                Top figure for the dataset-level turbine layout. Coordinates and neighbor edges come from the
                normalized `turbine_static` sidecar used elsewhere in the repository.
                """
            ).strip()
        ),
        _code_cell(_site_layout_code(dataset_id)),
    ]

    for metadata in protocol_metadata:
        cells.append(_markdown_cell(_protocol_markdown(metadata)))
        cells.append(_code_cell(_protocol_plot_code(metadata.feature_protocol_id)))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_dataset_notebook(
    dataset_id: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> Path:
    spec = get_dataset_spec(dataset_id)
    protocol_metadata, omitted_protocol_ids = collect_dataset_protocol_metadata(
        dataset_id,
        cache_root=cache_root,
    )
    task_id = protocol_metadata[0].task_id if protocol_metadata else "next_6h_from_24h"
    payload = build_dataset_notebook_payload(
        dataset_id,
        official_name=spec.official_name or dataset_id,
        task_id=task_id,
        protocol_metadata=protocol_metadata,
        omitted_protocol_ids=omitted_protocol_ids,
    )
    output_path = dataset_notebook_path(dataset_id, output_dir=output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for dataset_id in args.dataset_ids:
        write_dataset_notebook(
            dataset_id,
            output_dir=args.output_dir,
            cache_root=args.cache_root,
        )
    return 0


def _intro_markdown(
    official_name: str,
    dataset_id: str,
    task_id: str,
    omitted_protocol_ids: Sequence[str],
) -> str:
    omitted_line = ""
    if omitted_protocol_ids:
        omitted = ", ".join(f"`{feature_protocol_id}`" for feature_protocol_id in omitted_protocol_ids)
        omitted_line = f"\n- Unsupported protocol sections omitted for this dataset: {omitted}."
    return textwrap.dedent(
        f"""
        # {official_name} Visualization

        - Dataset: `{dataset_id}`
        - Task: `{task_id}`
        - The top figure shows the full-farm turbine layout and neighbor structure.
        - Each protocol section below renders a farm-level timestamp status tile for one supported `feature_protocol_id`.
        - Green means the farm timestamp is clean. Yellow means at least one turbine uses a protocol mask input at that
          timestamp; mask has the highest display priority and overrides issue colors. Red means there is a
          protocol-variable issue after excluding warmup-only `missing_past_covariates` flags, with no target issue and
          no mask hit. Black means at least one turbine has a target issue (`quality_flags != ""`) and there is no mask
          hit.
        - Running a section may lazily build missing cache artifacts on first use, so the first run can be slower.{omitted_line}
        """
    ).strip()


def _protocol_markdown(metadata: ProtocolNotebookMetadata) -> str:
    protocol_columns = ", ".join(
        f"`{column}`"
        for column in (*metadata.target_history_mask_columns, *metadata.past_covariates)
    ) or "_none_"
    target_history_masks = ", ".join(f"`{column}`" for column in metadata.target_history_mask_columns) or "_none_"
    past_covariate_masks = ", ".join(f"`{column}`" for column in metadata.past_covariate_mask_columns) or "_none_"
    derived_source_columns = ", ".join(f"`{column}`" for column in metadata.derived_source_columns) or "_none_"
    notes = "\n".join(f"- Note: {note}" for note in metadata.dataset_specific_notes)
    notes_block = f"\n{notes}" if notes else ""
    mask_badge = ""
    mask_lines = ""
    mask_block = ""
    if metadata.target_history_mask_columns or metadata.past_covariate_mask_columns:
        mask_badge = (
            ' <span style="background-color:#ffe066;color:#5f4b00;'
            'padding:0.1rem 0.45rem;border-radius:0.35rem;font-size:0.72em;'
            'font-weight:700;vertical-align:middle;">MASK</span>'
        )
        mask_line_items = ['- <span style="color:#8a6d00;font-weight:700;">Masked protocol inputs enabled.</span>']
        if metadata.target_history_mask_columns:
            mask_line_items.append(f"- Target history mask columns: {target_history_masks}")
        if metadata.past_covariate_mask_columns:
            mask_line_items.append(f"- Companion mask columns: {past_covariate_masks}")
        mask_line_items.append(f"- Mask polarity: `{metadata.mask_polarity or 'unspecified'}`")
        mask_lines = "\n".join(mask_line_items)
        mask_block = f"\n{mask_lines}"
    return textwrap.dedent(
        f"""
        ## {metadata.display_name}{mask_badge}

        - `task_id`: `{metadata.task_id}`
        - `feature_protocol_id`: `{metadata.feature_protocol_id}`
        - Summary: {metadata.summary}
        - Protocol-covered task columns: {protocol_columns}
        - Raw source columns for derived protocol covariates: {derived_source_columns}{mask_block}{notes_block}
        """
    ).strip()


def _setup_code(dataset_id: str, task_id: str) -> str:
    return textwrap.dedent(
        f"""
        import importlib
        from pathlib import Path
        import sys

        import matplotlib.pyplot as plt
        import polars as pl
        from IPython.display import display


        def _find_repo_root(start: Path) -> Path:
            for candidate in (start, *start.parents):
                if (candidate / "pyproject.toml").exists():
                    return candidate
            raise FileNotFoundError("Unable to locate the repository root from the notebook working directory.")


        REPO_ROOT = _find_repo_root(Path.cwd())
        SRC_ROOT = REPO_ROOT / "src"
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))
        importlib.invalidate_caches()
        for module_name in [name for name in tuple(sys.modules) if name == "wind_datasets" or name.startswith("wind_datasets.")]:
            sys.modules.pop(module_name, None)

        from wind_datasets.visualization import (
            build_site_layout,
            load_farm_status_tile,
            load_turbine_static_for_visualization,
            plot_farm_status_tile,
            plot_site_layout,
        )

        DATASET_ID = "{dataset_id}"
        TASK_ID = "{task_id}"
        CACHE_ROOT = REPO_ROOT / "cache"
        SITE_NEIGHBORS = 2
        SITE_EDGE_COLOR = "#aab2bd"
        SITE_NODE_COLOR = "#2b6f97"
        STATUS_CLEAN_COLOR = "#2ca25f"
        STATUS_MASK_COLOR = "#ffd54f"
        STATUS_FEATURE_ISSUE_COLOR = "#d73027"
        STATUS_TARGET_ISSUE_COLOR = "#111111"
        PADDING_COLOR = "#ffffff"
        DPI = 160

        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.titlesize"] = 11
        """
    ).strip()


def _site_layout_code(dataset_id: str) -> str:
    return textwrap.dedent(
        f"""
        site_static = load_turbine_static_for_visualization("{dataset_id}", cache_root=CACHE_ROOT)
        site_layout = build_site_layout(site_static, neighbor_k=SITE_NEIGHBORS)
        site_figure, site_axis = plot_site_layout(
            site_layout,
            edge_color=SITE_EDGE_COLOR,
            node_color=SITE_NODE_COLOR,
        )
        plt.show()
        """
    ).strip()


def _protocol_plot_code(feature_protocol_id: str) -> str:
    return textwrap.dedent(
        f"""
        status_tile = load_farm_status_tile(
            DATASET_ID,
            "{feature_protocol_id}",
            cache_root=CACHE_ROOT,
        )
        status_summary = (
            pl.DataFrame([status_tile.to_summary()])
            .with_columns(
                pl.col("start_timestamp").dt.strftime("%Y-%m-%d %H:%M:%S"),
                pl.col("end_timestamp").dt.strftime("%Y-%m-%d %H:%M:%S"),
                pl.col("any_issue_share").round(4),
                pl.col("mask_hit_share").round(4),
            )
        )
        display(status_summary)
        status_figure, status_axis = plot_farm_status_tile(
            status_tile,
            clean_color=STATUS_CLEAN_COLOR,
            mask_color=STATUS_MASK_COLOR,
            feature_issue_color=STATUS_FEATURE_ISSUE_COLOR,
            target_issue_color=STATUS_TARGET_ISSUE_COLOR,
            padding_color=PADDING_COLOR,
        )
        plt.show()
        """
    ).strip()


def _markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(source),
    }


def _code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(source),
    }


def _source_lines(source: str) -> list[str]:
    lines = source.splitlines()
    if not lines:
        return []
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


if __name__ == "__main__":
    raise SystemExit(main())
