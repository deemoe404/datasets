from __future__ import annotations

import argparse
import csv
import gzip
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

DIAGNOSTICS_DIR = Path(__file__).resolve().parent
FAMILY_DIR = DIAGNOSTICS_DIR.parent
EXPERIMENT_ROOT = FAMILY_DIR.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))

import formal_tuning as formal  # noqa: E402

FAMILY_ID = formal.FAMILY_ID
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "experiment"
    / "artifacts"
    / "scratch"
    / FAMILY_ID
    / "long_run_20260425_paper_grade"
)
DEFAULT_PUBLISH_ROOT = (
    REPO_ROOT
    / "experiment"
    / "artifacts"
    / "published"
    / FAMILY_ID
    / "20260425-paper-grade-long-run"
)
PYTHON = REPO_ROOT / ".conda" / "bin" / "python"
FORMAL_RUNNER = FAMILY_DIR / "run_world_model_official_baselines_v2_formal_tuning.py"
ROLLING = formal.ROLLING_EVAL_PROTOCOL
NON_OVERLAP = formal.NON_OVERLAP_EVAL_PROTOCOL
TRAINABLE_SEEDS = (3407, 42, 2026, 17, 926)


@dataclass(frozen=True)
class QueueItem:
    item_id: str
    phase: str
    description: str
    command: tuple[str, ...]
    expected_artifacts: tuple[str, ...] = ()
    priority: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "command": list(self.command),
            "expected_artifacts": list(self.expected_artifacts),
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _expected(output_path: Path, *, origin_errors: bool = False) -> tuple[str, ...]:
    paths = [output_path, output_path.with_suffix(".summary.json")]
    if origin_errors:
        paths.append(output_path.with_suffix(".origin_errors.csv"))
    return tuple(str(path) for path in paths)


def _formal_command(
    *,
    output_path: Path,
    variants: Sequence[str],
    split_names: Sequence[str] = ("val",),
    eval_protocols: Sequence[str] = (ROLLING,),
    seed: int = 3407,
    learning_rate: float = 1e-4,
    train_batch_size: int = 128,
    max_epochs: int = 3,
    residual_anchor_steps: int = 0,
    max_train_origins: int | None = 4096,
    max_eval_origins: int | None = 1024,
    max_checkpoint_origins: int | None = 512,
    checkpoint_eval_protocol: str = ROLLING,
    gate_origin_count: int = 64,
    origin_errors: bool = False,
    extra_args: Sequence[str] = (),
    run_label: str | None = None,
) -> tuple[str, ...]:
    command: list[str] = [
        str(PYTHON),
        str(FORMAL_RUNNER),
        "--dataset",
        "kelmarsh",
        "--output-path",
        str(output_path),
        "--seed",
        str(seed),
        "--learning-rate",
        str(learning_rate),
        "--train-batch-size",
        str(train_batch_size),
        "--max-epochs",
        str(max_epochs),
        "--checkpoint-eval-protocol",
        checkpoint_eval_protocol,
        "--gate-origin-count",
        str(gate_origin_count),
        "--residual-anchor-steps",
        str(residual_anchor_steps),
    ]
    for variant in variants:
        command.extend(["--variant", variant])
    for split_name in split_names:
        command.extend(["--split-name", split_name])
    for eval_protocol in eval_protocols:
        command.extend(["--eval-protocol", eval_protocol])
    if max_train_origins is not None:
        command.extend(["--max-train-origins", str(max_train_origins)])
    if max_eval_origins is not None:
        command.extend(["--max-eval-origins", str(max_eval_origins)])
    if max_checkpoint_origins is not None:
        command.extend(["--max-checkpoint-origins", str(max_checkpoint_origins)])
    if origin_errors:
        command.extend(["--origin-error-output-path", str(output_path.with_suffix(".origin_errors.csv"))])
    if run_label:
        command.extend(["--run-label", run_label])
    command.extend(extra_args)
    return tuple(command)


def _phase2_formal_item(
    run_root: Path,
    *,
    item_id: str,
    family: str,
    variant: str,
    learning_rate: float,
    max_epochs: int,
    extra_args: Sequence[str],
    residual_anchor_steps: int = 1,
    priority: int = 20,
) -> QueueItem:
    output_path = run_root / "phase2_validation_search" / f"{item_id}.csv"
    return QueueItem(
        item_id=item_id,
        phase="phase2_validation_only_repair_search",
        description=f"Validation-only bounded search for {variant}.",
        command=_formal_command(
            output_path=output_path,
            variants=(variant,),
            split_names=("val",),
            eval_protocols=(ROLLING, NON_OVERLAP),
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            residual_anchor_steps=residual_anchor_steps,
            origin_errors=True,
            extra_args=extra_args,
            run_label=item_id,
        ),
        expected_artifacts=_expected(output_path, origin_errors=True),
        priority=priority,
        metadata={
            "family": family,
            "variant": variant,
            "selected_by": "validation_only",
            "no_test_feedback": True,
            "phase_role": "candidate_search",
        },
    )


def build_queue(run_root: Path = DEFAULT_RUN_ROOT) -> list[QueueItem]:
    run_root = Path(run_root).resolve()
    items: list[QueueItem] = []
    items.append(
        QueueItem(
            item_id="phase0_snapshot",
            phase="phase0_hygiene_preflight",
            description="Snapshot git status, diff stat, artifact inventory, and GPU info into the run root.",
            command=(str(PYTHON), str(Path(__file__).resolve()), "snapshot", "--run-root", str(run_root)),
            expected_artifacts=(
                str(run_root / "preflight" / "git_status_short.txt"),
                str(run_root / "preflight" / "git_diff_stat.txt"),
                str(run_root / "preflight" / "gpu_info.txt"),
                str(run_root / "preflight" / "artifact_inventory.json"),
            ),
            priority=0,
        )
    )
    items.append(
        QueueItem(
            item_id="phase0_pytest_v2",
            phase="phase0_hygiene_preflight",
            description="Run the official baseline v2 unit test file before launching long jobs.",
            command=(str(PYTHON), "-m", "pytest", "-q", "test/test_world_model_official_baselines_v2.py"),
            priority=1,
        )
    )

    controls_output = run_root / "phase1_controls" / "controls_full_val_test.csv"
    items.append(
        QueueItem(
            item_id="phase1_controls_full_val_test",
            phase="phase1_complete_controls",
            description="Full val/test controls: last-value, seasonal, and Ridge residual B0.",
            command=_formal_command(
                output_path=controls_output,
                variants=(
                    formal.PERSISTENCE_VARIANT,
                    formal.SEASONAL_PERSISTENCE_VARIANT,
                    formal.RIDGE_RESIDUAL_VARIANT,
                ),
                split_names=("val", "test"),
                eval_protocols=(ROLLING, NON_OVERLAP),
                max_train_origins=None,
                max_eval_origins=None,
                max_checkpoint_origins=None,
                max_epochs=1,
                residual_anchor_steps=0,
                origin_errors=True,
                run_label="paper_grade_controls_full_val_test",
            ),
            expected_artifacts=_expected(controls_output, origin_errors=True),
            priority=10,
            metadata={"family": "controls", "selected_by": "validation_only", "no_test_feedback": True},
        )
    )

    chronos_nonoverlap = run_root / "phase1_controls" / "chronos2_nonoverlap_val_test.csv"
    items.append(
        QueueItem(
            item_id="phase1_chronos2_nonoverlap_val_test",
            phase="phase1_complete_controls",
            description="Chronos-2 full non-overlap val/test through the formal runner.",
            command=_formal_command(
                output_path=chronos_nonoverlap,
                variants=(formal.CHRONOS2_VARIANT,),
                split_names=("val", "test"),
                eval_protocols=(NON_OVERLAP,),
                max_train_origins=None,
                max_eval_origins=None,
                max_checkpoint_origins=None,
                max_epochs=1,
                residual_anchor_steps=0,
                origin_errors=True,
                run_label="paper_grade_chronos2_nonoverlap_val_test",
            ),
            expected_artifacts=_expected(chronos_nonoverlap, origin_errors=True),
            priority=11,
            metadata={"family": "chronos2", "selected_by": "validation_only", "no_test_feedback": True},
        )
    )
    _add_chronos_rolling_items(items, run_root)

    for hidden_dim in (64, 96):
        for gcn_depth in (2, 3):
            for dropout in (0.0, 0.1):
                for lr in (0.0005, 0.001):
                    suffix = f"h{hidden_dim}_gcn{gcn_depth}_dropout{_safe(dropout)}_lr{_safe(lr)}_anchor1"
                    items.append(
                        _phase2_formal_item(
                            run_root,
                            item_id=f"dgcrn_b2_residual_{suffix}",
                            family="dgcrn",
                            variant=formal.DGCRN_RESIDUAL_VARIANT,
                            learning_rate=lr,
                            max_epochs=4,
                            extra_args=(
                                "--dgcrn-hidden-dim",
                                str(hidden_dim),
                                "--dgcrn-gcn-depth",
                                str(gcn_depth),
                                "--dgcrn-dropout",
                                str(dropout),
                            ),
                        )
                    )
                    items.append(
                        _phase2_formal_item(
                            run_root,
                            item_id=f"dgcrn_b3_geometry_residual_{suffix}",
                            family="dgcrn",
                            variant=formal.DGCRN_GEOMETRY_RESIDUAL_VARIANT,
                            learning_rate=lr,
                            max_epochs=4,
                            extra_args=(
                                "--dgcrn-hidden-dim",
                                str(hidden_dim),
                                "--dgcrn-gcn-depth",
                                str(gcn_depth),
                                "--dgcrn-dropout",
                                str(dropout),
                            ),
                        )
                    )

    for variant, feature_mode in (
        (formal.ITRANSFORMER_TARGET_RESIDUAL_VARIANT, "target_only"),
        (formal.ITRANSFORMER_EXOG_RESIDUAL_VARIANT, "target_plus_exog"),
    ):
        for d_model in (64, 128):
            for e_layers in (1, 2):
                for lr in (0.0001, 0.0003):
                    item_id = f"itransformer_{feature_mode}_residual_d{d_model}_e{e_layers}_lr{_safe(lr)}_anchor1"
                    items.append(
                        _phase2_formal_item(
                            run_root,
                            item_id=item_id,
                            family="itransformer",
                            variant=variant,
                            learning_rate=lr,
                            max_epochs=4,
                            extra_args=(
                                "--itransformer-d-model",
                                str(d_model),
                                "--itransformer-e-layers",
                                str(e_layers),
                                "--itransformer-dropout",
                                "0.1",
                            ),
                        )
                    )

    for variant, feature_mode in (
        (formal.TIMEXER_TARGET_RESIDUAL_VARIANT, "target_only"),
        (formal.TIMEXER_FULL_RESIDUAL_VARIANT, "full_exog"),
    ):
        for d_model in (64, 128):
            for patch_len in (6, 12):
                for e_layers in (1, 2):
                    for lr in (0.0001, 0.0003):
                        item_id = (
                            f"timexer_{feature_mode}_residual_d{d_model}_patch{patch_len}"
                            f"_e{e_layers}_lr{_safe(lr)}_anchor1"
                        )
                        items.append(
                            _phase2_formal_item(
                                run_root,
                                item_id=item_id,
                                family="timexer",
                                variant=variant,
                                learning_rate=lr,
                                max_epochs=4,
                                extra_args=(
                                    "--timexer-d-model",
                                    str(d_model),
                                    "--timexer-patch-len",
                                    str(patch_len),
                                    "--timexer-e-layers",
                                    str(e_layers),
                                    "--timexer-dropout",
                                    "0.1",
                                ),
                            )
                        )

    items.append(
        _phase2_formal_item(
            run_root,
            item_id="tft_residual_b2_h32_lstm1_heads4_hc16_dropout0p1_lr0p0003_anchor1",
            family="tft",
            variant=formal.TFT_RESIDUAL_VARIANT,
            learning_rate=0.0003,
            max_epochs=8,
            extra_args=(
                "--tft-hidden-size",
                "32",
                "--tft-lstm-layers",
                "1",
                "--tft-attention-head-size",
                "4",
                "--tft-hidden-continuous-size",
                "16",
                "--tft-dropout",
                "0.1",
            ),
            priority=40,
        )
    )
    items.append(
        _phase2_formal_item(
            run_root,
            item_id="mtgnn_calendar_residual_b1_gcn2_sub3_node32_res32_skip64_end128_layers2_dropout0_lr0p003_anchor1",
            family="mtgnn",
            variant=formal.MTGNN_CALENDAR_RESIDUAL_VARIANT,
            learning_rate=0.003,
            max_epochs=8,
            extra_args=(
                "--mtgnn-gcn-depth",
                "2",
                "--mtgnn-subgraph-size",
                "3",
                "--mtgnn-node-dim",
                "32",
                "--mtgnn-residual-channels",
                "32",
                "--mtgnn-skip-channels",
                "64",
                "--mtgnn-end-channels",
                "128",
                "--mtgnn-layers",
                "2",
                "--mtgnn-dropout",
                "0.0",
            ),
            priority=50,
        )
    )
    items.append(
        QueueItem(
            item_id="phase3_select_validation_configs",
            phase="phase3_full_validation_selection",
            description="Select candidates from validation rows only; no test metrics are read.",
            command=(
                str(PYTHON),
                str(Path(__file__).resolve()),
                "select",
                "--run-root",
                str(run_root),
                "--queue-path",
                str(run_root / "queue.json"),
                "--selection-output",
                str(run_root / "phase3_selected_validation_configs.json"),
            ),
            expected_artifacts=(str(run_root / "phase3_selected_validation_configs.json"),),
            priority=70,
            metadata={"selected_by": "validation_only", "no_test_feedback": True},
        )
    )
    items.append(
        QueueItem(
            item_id="phase3_materialize_full_validation_queue",
            phase="phase3_full_validation_selection",
            description="Write a second queue that reruns selected configs on full validation with no origin caps.",
            command=(
                str(PYTHON),
                str(Path(__file__).resolve()),
                "materialize",
                "--run-root",
                str(run_root),
                "--selection-input",
                str(run_root / "phase3_selected_validation_configs.json"),
                "--queue-output",
                str(run_root / "phase3_full_validation_queue.json"),
                "--mode",
                "full-validation",
            ),
            expected_artifacts=(str(run_root / "phase3_full_validation_queue.json"),),
            priority=71,
            metadata={"selected_by": "validation_only", "no_test_feedback": True},
        )
    )
    items.append(
        QueueItem(
            item_id="phase4_materialize_test_multiseed_queue",
            phase="phase4_frozen_test_once_multiseed",
            description=(
                "Write the frozen test/multiseed queue from a validation-only selection JSON. "
                "By default this expects phase4_selected_full_validation_configs.json, generated by rerunning select on phase3 outputs."
            ),
            command=(
                str(PYTHON),
                str(Path(__file__).resolve()),
                "materialize",
                "--run-root",
                str(run_root),
                "--selection-input",
                str(run_root / "phase4_selected_full_validation_configs.json"),
                "--queue-output",
                str(run_root / "phase4_test_multiseed_queue.json"),
                "--mode",
                "test-multiseed",
            ),
            expected_artifacts=(str(run_root / "phase4_test_multiseed_queue.json"),),
            priority=72,
            metadata={"selected_by": "validation_only", "no_test_feedback": True},
        )
    )
    items.append(
        QueueItem(
            item_id="phase5_aggregate_status",
            phase="phase5_publish",
            description="Aggregate long-run driver status and artifact inventory for publication handoff.",
            command=(
                str(PYTHON),
                str(Path(__file__).resolve()),
                "aggregate",
                "--run-root",
                str(run_root),
                "--publish-root",
                str(DEFAULT_PUBLISH_ROOT),
            ),
            expected_artifacts=(str(run_root / "long_run_status_summary.json"),),
            priority=90,
            metadata={"selected_by": "validation_only", "no_test_feedback": True},
        )
    )
    return sorted(items, key=lambda item: (item.priority, item.item_id))


def _add_chronos_rolling_items(items: list[QueueItem], run_root: Path) -> None:
    script = DIAGNOSTICS_DIR / "chronos2_rolling_shards.py"
    output_dir = run_root / "phase1_controls" / "chronos2_rolling_shards"
    shard_size = 4096
    window_counts = {"val": 47139, "test": 94458}
    for split_name, total_count in window_counts.items():
        common = (
            str(PYTHON),
            str(script),
        )
        plan_expected = output_dir / f"chronos2_kelmarsh_{split_name}_{ROLLING}_plan.json"
        command = [
            *common,
            "plan",
            "--dataset-id",
            "kelmarsh",
            "--split",
            split_name,
            "--eval-protocol",
            ROLLING,
            "--output-dir",
            str(output_dir),
            "--shard-size",
            str(shard_size),
        ]
        if split_name == "test":
            command.append("--allow-test")
        items.append(
            QueueItem(
                item_id=f"phase1_chronos2_{split_name}_rolling_plan",
                phase="phase1_complete_controls",
                description=f"Write Chronos-2 {split_name} rolling shard plan.",
                command=tuple(command),
                expected_artifacts=(str(plan_expected),),
                priority=12,
                metadata={"family": "chronos2", "split_name": split_name, "eval_protocol": ROLLING},
            )
        )
        shard_count = (total_count + shard_size - 1) // shard_size
        for shard_index in range(shard_count):
            start = shard_index * shard_size
            stop = min(start + shard_size, total_count)
            stem = f"chronos2_kelmarsh_{split_name}_{ROLLING}_shard_{start:06d}_{stop:06d}"
            command = [
                *common,
                "eval-shard",
                "--dataset-id",
                "kelmarsh",
                "--split",
                split_name,
                "--eval-protocol",
                ROLLING,
                "--output-dir",
                str(output_dir),
                "--device",
                "cuda",
                "--batch-size",
                "32",
                "--shard-size",
                str(shard_size),
                "--shard-index",
                str(shard_index),
            ]
            if split_name == "test":
                command.append("--allow-test")
            items.append(
                QueueItem(
                    item_id=f"phase1_chronos2_{split_name}_rolling_shard_{shard_index:03d}",
                    phase="phase1_complete_controls",
                    description=f"Evaluate Chronos-2 {split_name} rolling shard [{start}, {stop}).",
                    command=tuple(command),
                    expected_artifacts=(
                        str(output_dir / f"{stem}.json"),
                        str(output_dir / f"{stem}.csv"),
                        str(output_dir / f"{stem}.abs_errors.npy"),
                    ),
                    priority=13 if split_name == "val" else 14,
                    metadata={
                        "family": "chronos2",
                        "split_name": split_name,
                        "eval_protocol": ROLLING,
                        "shard_start": start,
                        "shard_stop": stop,
                    },
                )
            )
        aggregate_stem = f"chronos2_kelmarsh_{split_name}_{ROLLING}_aggregate"
        command = [
            *common,
            "aggregate",
            "--dataset-id",
            "kelmarsh",
            "--split",
            split_name,
            "--eval-protocol",
            ROLLING,
            "--output-dir",
            str(output_dir),
        ]
        if split_name == "test":
            command.append("--allow-test")
        items.append(
            QueueItem(
                item_id=f"phase1_chronos2_{split_name}_rolling_aggregate",
                phase="phase1_complete_controls",
                description=f"Aggregate complete Chronos-2 {split_name} rolling shards.",
                command=tuple(command),
                expected_artifacts=(
                    str(output_dir / f"{aggregate_stem}.json"),
                    str(output_dir / f"{aggregate_stem}.csv"),
                ),
                priority=15,
                metadata={"family": "chronos2", "split_name": split_name, "eval_protocol": ROLLING},
            )
        )


def write_plan(run_root: Path, *, queue_path: Path | None = None) -> Path:
    run_root = Path(run_root).resolve()
    queue_path = queue_path or run_root / "queue.json"
    items = build_queue(run_root)
    payload = {
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_root": str(run_root),
        "family_id": FAMILY_ID,
        "target": {
            "dataset_id": "kelmarsh",
            "feature_protocol_id": formal.FEATURE_PROTOCOL_ID,
            "task_id": formal.TASK_ID,
            "history_steps": formal.HISTORY_STEPS,
            "forecast_steps": formal.FORECAST_STEPS,
        },
        "policy": {
            "selected_by": "validation_only",
            "no_test_feedback": True,
            "test_metrics_used_for_config_selection": False,
            "timeout_policy": "depth_first",
            "gate_c_failures": "appendix_diagnostic_only",
        },
        "queue": [item.to_json() for item in items],
    }
    _write_json(queue_path, payload)
    return queue_path


def snapshot(run_root: Path) -> None:
    run_root = Path(run_root).resolve()
    preflight_dir = run_root / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    _write_text_command(preflight_dir / "git_status_short.txt", ["git", "status", "--short"])
    _write_text_command(preflight_dir / "git_diff_stat.txt", ["git", "diff", "--stat"])
    _write_text_command(preflight_dir / "gpu_info.txt", ["nvidia-smi"])
    inventory_roots = [
        REPO_ROOT / "experiment" / "artifacts" / "scratch" / FAMILY_ID,
        REPO_ROOT / "experiment" / "artifacts" / "published" / FAMILY_ID,
        REPO_ROOT / "experiment" / "artifacts" / "runs" / FAMILY_ID,
    ]
    inventory: list[dict[str, Any]] = []
    for root in inventory_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            inventory.append(
                {
                    "path": str(path.resolve().relative_to(REPO_ROOT)),
                    "size_bytes": int(stat.st_size),
                    "mtime": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                }
            )
    _write_json(
        preflight_dir / "artifact_inventory.json",
        {
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "artifact_count": len(inventory),
            "artifacts": sorted(inventory, key=lambda row: row["path"]),
        },
    )


def _write_text_command(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        body = completed.stdout
        if completed.stderr:
            body += "\n[stderr]\n" + completed.stderr
        body += f"\n[exit_code] {completed.returncode}\n"
    except FileNotFoundError as exc:
        body = f"{command[0]} not available: {exc}\n[exit_code] 127\n"
    path.write_text(body, encoding="utf-8")


def run_queue(queue_path: Path, *, run_root: Path, force: bool = False, start_after: str | None = None) -> None:
    run_root = Path(run_root).resolve()
    if not queue_path.exists():
        write_plan(run_root, queue_path=queue_path)
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    status_dir = run_root / "status"
    log_dir = run_root / "logs"
    status_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    skipping = start_after is not None
    for item in payload["queue"]:
        item_id = item["item_id"]
        if skipping:
            if item_id == start_after:
                skipping = False
            continue
        expected = [Path(path) for path in item.get("expected_artifacts", [])]
        status_path = status_dir / f"{item_id}.exit.json"
        if not force and expected and all(path.exists() for path in expected):
            _write_json(
                status_path,
                _exit_record(item, returncode=0, status="skipped_existing_artifacts", started_at=None, ended_at=None),
            )
            continue
        started_at = datetime.now(tz=UTC).isoformat()
        started = time.perf_counter()
        log_path = log_dir / f"{item_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"$ {shlex.join(item['command'])}\n\n")
            log_handle.flush()
            completed = subprocess.run(
                item["command"],
                cwd=REPO_ROOT,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        ended_at = datetime.now(tz=UTC).isoformat()
        artifact_status = {str(path): path.exists() for path in expected}
        status = "completed" if completed.returncode == 0 and all(artifact_status.values()) else "failed"
        _write_json(
            status_path,
            {
                **_exit_record(
                    item,
                    returncode=int(completed.returncode),
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                ),
                "duration_seconds": time.perf_counter() - started,
                "log_path": str(log_path),
                "artifact_status": artifact_status,
            },
        )
        if completed.returncode != 0:
            raise SystemExit(f"{item_id} failed with exit code {completed.returncode}; see {log_path}")


def _exit_record(
    item: dict[str, Any],
    *,
    returncode: int,
    status: str,
    started_at: str | None,
    ended_at: str | None,
) -> dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "phase": item["phase"],
        "description": item["description"],
        "status": status,
        "returncode": returncode,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "command": item["command"],
        "metadata": item.get("metadata", {}),
    }


def select_validation_configs(run_root: Path, *, queue_path: Path, selection_output: Path) -> None:
    run_root = Path(run_root).resolve()
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    candidates: dict[str, list[dict[str, Any]]] = {}
    for item in payload["queue"]:
        metadata = item.get("metadata", {})
        if metadata.get("phase_role") != "candidate_search":
            continue
        family = metadata.get("family")
        if not family:
            continue
        csv_paths = [Path(path) for path in item.get("expected_artifacts", []) if str(path).endswith(".csv")]
        result_csvs = [path for path in csv_paths if path.name.endswith(".csv") and not path.name.endswith(".origin_errors.csv")]
        for result_csv in result_csvs:
            if not result_csv.exists():
                continue
            rows = _read_csv(result_csv)
            rolling = _metric_row(rows, split_name="val", eval_protocol=ROLLING)
            nonoverlap = _metric_row(rows, split_name="val", eval_protocol=NON_OVERLAP)
            if rolling is None:
                continue
            gate_c = _boolish(rolling.get("gate_c_passed"))
            if gate_c is False:
                continue
            candidates.setdefault(family, []).append(
                {
                    "item_id": item["item_id"],
                    "family": family,
                    "variant": metadata.get("variant"),
                    "result_csv": str(result_csv),
                    "val_rolling_rmse_pu": _float_or_inf(rolling.get("rmse_pu")),
                    "val_nonoverlap_mae_pu": _float_or_inf(nonoverlap.get("mae_pu") if nonoverlap else None),
                    "gate_b_passed": _boolish(rolling.get("gate_b_passed")),
                    "gate_c_passed": gate_c,
                    "command": item["command"],
                }
            )
    selected: dict[str, dict[str, Any]] = {}
    for family, family_candidates in candidates.items():
        selected[family] = min(
            family_candidates,
            key=lambda row: (row["val_rolling_rmse_pu"], row["val_nonoverlap_mae_pu"], row["item_id"]),
        )
    _write_json(
        selection_output,
        {
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "run_root": str(run_root),
            "selected_by": "validation_only",
            "no_test_feedback": True,
            "test_metrics_read": False,
            "selection_metric": "val rolling RMSE; tie-break val non-overlap MAE",
            "selected": selected,
            "candidate_counts": {family: len(rows) for family, rows in candidates.items()},
        },
    )


def materialize_selected_queue(
    run_root: Path,
    *,
    selection_input: Path,
    queue_output: Path,
    mode: str,
) -> None:
    run_root = Path(run_root).resolve()
    if not selection_input.exists():
        _write_json(
            queue_output,
            {
                "created_at_utc": datetime.now(tz=UTC).isoformat(),
                "run_root": str(run_root),
                "source_selection": str(selection_input),
                "mode": mode,
                "status": "blocked_missing_selection",
                "blocked_reason": "selection_input_not_found",
                "selected_by": "validation_only",
                "no_test_feedback": True,
                "queue": [],
            },
        )
        return
    selection = json.loads(selection_input.read_text(encoding="utf-8"))
    selected = selection.get("selected", {})
    items: list[QueueItem] = []
    if mode == "full-validation":
        for priority, family in enumerate(("dgcrn", "itransformer", "timexer", "tft", "mtgnn"), start=1):
            row = selected.get(family)
            if not row:
                continue
            max_epochs = 12 if family in {"itransformer", "timexer"} else 16
            output_path = run_root / "phase3_full_validation" / f"{row['item_id']}_full_validation.csv"
            command = _rewrite_formal_command(
                row["command"],
                output_path=output_path,
                split_names=("val",),
                eval_protocols=(ROLLING, NON_OVERLAP),
                seed=3407,
                max_epochs=max_epochs,
                origin_errors=True,
                run_label=f"{row['item_id']}_full_validation",
            )
            items.append(
                QueueItem(
                    item_id=f"phase3_full_validation_{row['item_id']}",
                    phase="phase3_full_validation_selection",
                    description=f"Full validation rerun for validation-selected {family} config.",
                    command=command,
                    expected_artifacts=_expected(output_path, origin_errors=True),
                    priority=priority,
                    metadata={
                        "family": family,
                        "source_selection_item_id": row["item_id"],
                        "selected_by": "validation_only",
                        "no_test_feedback": True,
                        "phase_role": "candidate_search",
                    },
                )
            )
    elif mode == "test-multiseed":
        for family_priority, family in enumerate(("dgcrn", "itransformer", "timexer", "tft", "mtgnn"), start=1):
            row = selected.get(family)
            if not row:
                continue
            max_epochs = 12 if family in {"itransformer", "timexer"} else 16
            for seed_index, seed in enumerate(TRAINABLE_SEEDS):
                output_path = run_root / "phase4_frozen_test_multiseed" / f"{row['item_id']}_seed{seed}_test.csv"
                command = _rewrite_formal_command(
                    row["command"],
                    output_path=output_path,
                    split_names=("test",),
                    eval_protocols=(ROLLING, NON_OVERLAP),
                    seed=seed,
                    max_epochs=max_epochs,
                    origin_errors=True,
                    run_label=f"{row['item_id']}_seed{seed}_test_once",
                )
                items.append(
                    QueueItem(
                        item_id=f"phase4_test_once_{row['item_id']}_seed{seed}",
                        phase="phase4_frozen_test_once_multiseed",
                        description=f"Frozen test-once run for validation-selected {family} config seed {seed}.",
                        command=command,
                        expected_artifacts=_expected(output_path, origin_errors=True),
                        priority=family_priority * 10 + seed_index,
                        metadata={
                            "family": family,
                            "source_selection_item_id": row["item_id"],
                            "seed": seed,
                            "selected_by": "validation_only",
                            "no_test_feedback": True,
                            "test_feedback_used_for_config_selection": False,
                        },
                    )
                )
    else:
        raise ValueError(f"Unsupported materialize mode {mode!r}.")
    _write_json(
        queue_output,
        {
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "run_root": str(run_root),
            "source_selection": str(selection_input),
            "mode": mode,
            "selected_by": "validation_only",
            "no_test_feedback": True,
            "queue": [item.to_json() for item in sorted(items, key=lambda item: (item.priority, item.item_id))],
        },
    )


def _rewrite_formal_command(
    command: Sequence[str],
    *,
    output_path: Path,
    split_names: Sequence[str],
    eval_protocols: Sequence[str],
    seed: int,
    max_epochs: int,
    origin_errors: bool,
    run_label: str,
) -> tuple[str, ...]:
    drop_options = {
        "--output-path",
        "--split-name",
        "--eval-protocol",
        "--max-train-origins",
        "--max-eval-origins",
        "--max-checkpoint-origins",
        "--max-epochs",
        "--seed",
        "--origin-error-output-path",
        "--run-label",
        "--no-record-run",
    }
    rewritten: list[str] = []
    index = 0
    command = list(command)
    while index < len(command):
        token = command[index]
        if token in drop_options:
            if token != "--no-record-run" and index + 1 < len(command):
                index += 2
            else:
                index += 1
            continue
        rewritten.append(token)
        index += 1
    rewritten.extend(["--output-path", str(output_path), "--seed", str(seed), "--max-epochs", str(max_epochs)])
    for split_name in split_names:
        rewritten.extend(["--split-name", split_name])
    for eval_protocol in eval_protocols:
        rewritten.extend(["--eval-protocol", eval_protocol])
    if origin_errors:
        rewritten.extend(["--origin-error-output-path", str(output_path.with_suffix(".origin_errors.csv"))])
    rewritten.extend(["--run-label", run_label])
    return tuple(rewritten)


def _read_csv(path: Path) -> list[dict[str, str]]:
    resolved = path if path.exists() else Path(f"{path}.gz")
    opener = gzip.open if resolved.suffix == ".gz" else Path.open
    with opener(resolved, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_row(rows: Iterable[dict[str, str]], *, split_name: str, eval_protocol: str) -> dict[str, str] | None:
    best: dict[str, str] | None = None
    for row in rows:
        if row.get("split_name") != split_name or row.get("eval_protocol") != eval_protocol:
            continue
        if row.get("metric_scope") != "overall" or row.get("trial_status") != "completed":
            continue
        if row.get("is_best_validation_trial", "True").lower() not in {"true", "1"}:
            continue
        if best is None or _float_or_inf(row.get("rmse_pu")) < _float_or_inf(best.get("rmse_pu")):
            best = row
    return best


def _boolish(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _float_or_inf(value: Any) -> float:
    if value is None or value == "":
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def aggregate_status(run_root: Path, *, publish_root: Path) -> None:
    run_root = Path(run_root).resolve()
    publish_root = Path(publish_root).resolve()
    status_dir = run_root / "status"
    statuses = []
    if status_dir.exists():
        for status_path in sorted(status_dir.glob("*.exit.json")):
            statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
    summary = {
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_root": str(run_root),
        "publish_root": str(publish_root),
        "status_count": len(statuses),
        "completed_count": sum(1 for row in statuses if row.get("status") in {"completed", "skipped_existing_artifacts"}),
        "failed_count": sum(1 for row in statuses if row.get("status") == "failed"),
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "test_metrics_used_for_config_selection": False,
        "statuses": statuses,
    }
    _write_json(run_root / "long_run_status_summary.json", summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable one-GPU driver for the official baselines v2 paper-grade long run.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Write the long-run queue JSON without executing it.")
    plan_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    plan_parser.add_argument("--queue-path", type=Path)

    snapshot_parser = subparsers.add_parser("snapshot", help="Snapshot repository/GPU/artifact state into the run root.")
    snapshot_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)

    run_parser = subparsers.add_parser("run", help="Execute queue items one at a time with logs and exit records.")
    run_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run_parser.add_argument("--queue-path", type=Path)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--start-after")

    select_parser = subparsers.add_parser("select", help="Select configs from validation CSVs only.")
    select_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    select_parser.add_argument("--queue-path", type=Path)
    select_parser.add_argument("--selection-output", type=Path)

    materialize_parser = subparsers.add_parser("materialize", help="Write full-validation or frozen-test queues from a selection JSON.")
    materialize_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    materialize_parser.add_argument("--selection-input", type=Path, required=True)
    materialize_parser.add_argument("--queue-output", type=Path, required=True)
    materialize_parser.add_argument("--mode", choices=("full-validation", "test-multiseed"), required=True)

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate driver status for publication handoff.")
    aggregate_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    aggregate_parser.add_argument("--publish-root", type=Path, default=DEFAULT_PUBLISH_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "plan":
        queue_path = write_plan(args.run_root, queue_path=args.queue_path)
        print(str(queue_path))
        return 0
    if args.command == "snapshot":
        snapshot(args.run_root)
        return 0
    if args.command == "run":
        queue_path = args.queue_path or args.run_root / "queue.json"
        run_queue(queue_path, run_root=args.run_root, force=args.force, start_after=args.start_after)
        return 0
    if args.command == "select":
        queue_path = args.queue_path or args.run_root / "queue.json"
        selection_output = args.selection_output or args.run_root / "phase3_selected_validation_configs.json"
        select_validation_configs(args.run_root, queue_path=queue_path, selection_output=selection_output)
        return 0
    if args.command == "materialize":
        materialize_selected_queue(
            args.run_root,
            selection_input=args.selection_input,
            queue_output=args.queue_output,
            mode=args.mode,
        )
        return 0
    if args.command == "aggregate":
        aggregate_status(args.run_root, publish_root=args.publish_root)
        return 0
    raise AssertionError(f"Unhandled command {args.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
