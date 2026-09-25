"""Summarize recorded push transients from full spin evaluation traces."""

import argparse
import csv
import math
from pathlib import Path
import statistics


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def mean(values):
    return statistics.mean(values) if values else None


def rms(values):
    return math.sqrt(mean([value * value for value in values])) if values else None


def p90(values):
    values = sorted(value for value in values if value is not None)
    return values[math.ceil(0.9 * len(values)) - 1] if values else None


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def window(samples, start, end):
    return [row for row in samples if start <= row["time_s"] < end]


def window_metrics(samples):
    if not samples:
        return (None, None, None, None)
    return (
        mean([abs(row["actual_yaw_radps"] - row["cmd_yaw_radps"]) for row in samples]),
        mean([row["tilt_deg"] for row in samples]),
        rms([math.hypot(row["roll_rate_radps"], row["pitch_rate_radps"]) for row in samples]),
        max(row["torque_limit_fraction"] for row in samples),
    )


def xy_shift(samples, start, delay):
    if not samples or samples[-1]["time_s"] < start + delay:
        return None
    anchor = min(samples, key=lambda row: abs(row["time_s"] - start))
    later = min(samples, key=lambda row: abs(row["time_s"] - start - delay))
    return math.hypot(later["x_m"] - anchor["x_m"], later["y_m"] - anchor["y_m"])


def net_xy_travel(samples):
    if len(samples) < 2:
        return None
    return math.hypot(
        samples[-1]["x_m"] - samples[0]["x_m"],
        samples[-1]["y_m"] - samples[0]["y_m"],
    )


def analyze(directory):
    episodes = read_rows(directory / "episodes.csv")
    results = []
    for episode in episodes:
        path = directory / "traces" / f"episode_{int(episode['episode_id']):04d}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Full trace missing: {path}")
        samples = [
            {key: float(value) for key, value in row.items()}
            for row in read_rows(path)
        ]
        pushes = [row for row in samples if row["push_impulse_ns"] > 0]
        for push in pushes:
            time = push["time_s"]
            pre = window_metrics(window(samples, time - 1.0, time))
            early = window_metrics(window(samples, time, time + 1.0))
            late = window_metrics(window(samples, time + 3.0, time + 4.0))
            results.append({
                "episode_id": episode["episode_id"],
                "yaw_cmd_radps": episode["yaw_cmd_radps"],
                "push_time_s": time,
                "push_impulse_ns": push["push_impulse_ns"],
                "pre_yaw_mae_radps": pre[0],
                "early_yaw_mae_radps": early[0],
                "late_yaw_mae_radps": late[0],
                "pre_mean_tilt_deg": pre[1],
                "early_mean_tilt_deg": early[1],
                "late_mean_tilt_deg": late[1],
                "pre_rp_rate_rms_radps": pre[2],
                "early_rp_rate_rms_radps": early[2],
                "late_rp_rate_rms_radps": late[2],
                "pre_peak_torque_fraction": pre[3],
                "early_peak_torque_fraction": early[3],
                "late_peak_torque_fraction": late[3],
            "xy_shift_1s_m": xy_shift(samples, time, 1.0),
            "xy_shift_4s_m": xy_shift(samples, time, 4.0),
            "pre_xy_travel_1s_m": net_xy_travel(window(samples, time - 1.0, time)),
            "early_xy_travel_1s_m": net_xy_travel(window(samples, time, time + 1.0)),
            "late_xy_travel_1s_m": net_xy_travel(window(samples, time + 3.0, time + 4.0)),
            "complete": episode["timed_out"],
            })
    if not results:
        raise ValueError(f"No recorded push events in {directory}")
    with (directory / "push_diagnostics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    lines = ["# Push transient diagnostics", "",
             "Windows are 1 s before, 0–1 s after, and 3–4 s after the actual push event.",
             "The position shift is relative to the push instant, not the episode start.", "",
             "| yaw | pushes | median impulse Ns | pre/early/late yaw MAE rad/s | pre/early/late mean tilt ° | pre/early/late roll-pitch RMS rad/s | p90 XY shift at 1/4 s m | p90 pre/early/late 1 s XY travel m |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for yaw in sorted({float(row["yaw_cmd_radps"]) for row in results}):
        group = [row for row in results if float(row["yaw_cmd_radps"]) == yaw]
        triple = lambda names: "/".join(fmt(mean([row[name] for row in group if row[name] is not None])) for name in names)
        lines.append(
            f"| {yaw:+g} | {len(group)} | {fmt(statistics.median(row['push_impulse_ns'] for row in group))} | "
            f"{triple(('pre_yaw_mae_radps', 'early_yaw_mae_radps', 'late_yaw_mae_radps'))} | "
            f"{triple(('pre_mean_tilt_deg', 'early_mean_tilt_deg', 'late_mean_tilt_deg'))} | "
            f"{triple(('pre_rp_rate_rms_radps', 'early_rp_rate_rms_radps', 'late_rp_rate_rms_radps'))} | "
            f"{fmt(p90([row['xy_shift_1s_m'] for row in group]))}/{fmt(p90([row['xy_shift_4s_m'] for row in group]))} | "
            f"{fmt(p90([row['pre_xy_travel_1s_m'] for row in group]))}/"
            f"{fmt(p90([row['early_xy_travel_1s_m'] for row in group]))}/"
            f"{fmt(p90([row['late_xy_travel_1s_m'] for row in group]))} |"
        )
    lines.extend(("", "See `push_diagnostics.csv` for each event.", ""))
    (directory / "push_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
    return directory / "push_diagnostics.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    args = parser.parse_args()
    print(analyze(args.evaluation_dir))


if __name__ == "__main__":
    main()
