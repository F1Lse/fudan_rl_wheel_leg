"""Compare two spin evaluations made with the same scenario and seed.

This reads saved CSV/JSON only; Isaac Gym is not needed.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


MATCH_FIELDS = (
    "target_id", "target_sha256", "seed", "num_envs", "yaw_commands_radps",
    "height_m", "episode_seconds", "warmup_seconds", "yaw_ramp_seconds",
    "episodes_per_yaw",
    "tilt_metric_window", "wobble_metric_window", "axis_sway_frame",
    "randomization", "push_robots", "domain_randomization",
    "domain_ablation", "fixed_restitution", "observation_noise",
    "stable_thresholds", "trace_seconds",
    "robot_asset", "robot_asset_sha256", "code_sha256",
    "control_stiffness", "control_damping",
)


def load_result(path):
    with (path / "manifest.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    with (path / "episodes.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No episodes in {path}")
    return manifest, rows


def p90(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def metrics(rows, yaw):
    group = [row for row in rows if float(row["yaw_cmd_radps"]) == yaw]
    if not group:
        raise ValueError(f"Missing yaw {yaw:g} in evaluation")
    wobble = (
        p90(float(row["steady_wobble_rms_deg"]) for row in group)
        if "steady_wobble_rms_deg" in group[0] else None
    )
    axis_sway = (
        p90(float(row["steady_world_axis_sway_rms_deg"]) for row in group)
        if "steady_world_axis_sway_rms_deg" in group[0] else None
    )
    return {
        "mae": mean(float(row["yaw_mae_radps"]) for row in group),
        "drift": p90(float(row["max_drift_m"]) for row in group),
        "tilt": p90(float(row["max_tilt_deg"]) for row in group),
        "rate": p90(float(row["rms_roll_pitch_rate_radps"]) for row in group),
        "wobble": wobble,
        "axis_sway": axis_sway,
        "stable": mean(row["stable"].lower() == "true" for row in group),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--gate-yaws", default="", help="comma-separated yaw cells to gate")
    parser.add_argument("--max-mae", type=float, default=0.8)
    parser.add_argument("--max-drift", type=float, default=0.15)
    parser.add_argument("--max-tilt", type=float, default=15.0)
    parser.add_argument("--max-rate", type=float, default=0.5)
    parser.add_argument("--max-old-mae-increase", type=float, default=0.1)
    parser.add_argument("--max-old-axis-sway-increase", type=float)
    args = parser.parse_args()

    base_manifest, base_rows = load_result(args.baseline)
    cand_manifest, cand_rows = load_result(args.candidate)
    mismatches = [
        key for key in MATCH_FIELDS if base_manifest.get(key) != cand_manifest.get(key)
    ]
    if mismatches:
        raise ValueError("Evaluations are not paired; differing fields: " + ", ".join(mismatches))

    gate_yaws = (
        {float(value.strip()) for value in args.gate_yaws.split(",")}
        if args.gate_yaws else set()
    )
    unknown = gate_yaws - set(base_manifest["yaw_commands_radps"])
    if unknown:
        raise ValueError(f"Gate yaw(s) missing from evaluation: {sorted(unknown)}")

    print("| yaw | MAE baseline -> candidate | p90 drift baseline -> candidate | "
          "p90 tilt baseline -> candidate | p90 world-axis sway baseline -> candidate | "
          "p90 body-frame variation baseline -> candidate | "
          "p90 RP rate baseline -> candidate | "
          "stable baseline -> candidate | gate |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    any_failed = False
    for yaw in base_manifest["yaw_commands_radps"]:
        old = metrics(base_rows, yaw)
        new = metrics(cand_rows, yaw)
        passed = (
            new["mae"] <= args.max_mae
            and new["drift"] <= args.max_drift
            and new["tilt"] <= args.max_tilt
            and new["rate"] <= args.max_rate
            and new["mae"] - old["mae"] <= args.max_old_mae_increase
        )
        if args.max_old_axis_sway_increase is not None:
            if old["axis_sway"] is None or new["axis_sway"] is None:
                raise ValueError("World-axis sway comparison requires new evaluator outputs")
            passed = passed and (
                new["axis_sway"] - old["axis_sway"]
                <= args.max_old_axis_sway_increase
            )
        if yaw in gate_yaws and not passed:
            any_failed = True
        status = ("PASS" if passed else "FAIL") if yaw in gate_yaws else "n/a"
        wobble_text = (
            "n/a" if old["wobble"] is None or new["wobble"] is None else
            f"{old['wobble']:.2f} -> {new['wobble']:.2f} deg"
        )
        axis_sway_text = (
            "n/a" if old["axis_sway"] is None or new["axis_sway"] is None else
            f"{old['axis_sway']:.2f} -> {new['axis_sway']:.2f} deg"
        )
        print(
            f"| {yaw:+g} | {old['mae']:.2f} -> {new['mae']:.2f} | "
            f"{old['drift']:.3f} -> {new['drift']:.3f} m | "
            f"{old['tilt']:.1f} -> {new['tilt']:.1f} deg | "
            f"{axis_sway_text} | "
            f"{wobble_text} | "
            f"{old['rate']:.2f} -> {new['rate']:.2f} rad/s | "
            f"{old['stable']:.1%} -> {new['stable']:.1%} | {status} |"
        )
    if gate_yaws:
        print("GATE:", "FAIL" if any_failed else "PASS")
        raise SystemExit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
