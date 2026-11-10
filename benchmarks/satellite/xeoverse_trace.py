# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export real-time orbital segments using an existing XeoVerse checkout.

Only the orbit and Sun geometry are physical inputs. The sunlight-only compute
policy and 30/50/100 ms mission slots are explicit experimental assumptions.
Run with the XeoVerse environment, which supplies numpy/scipy/sgp4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

EARTH_RADIUS_KM = 6378.137


def shadow_margin(position, sun):
    """Signed cylindrical-shadow limb distance; negative means eclipsed."""
    along = sum(a * b for a, b in zip(position, sun))
    perpendicular = math.sqrt(sum((a - along * b) ** 2 for a, b in zip(position, sun)))
    return perpendicular - EARTH_RADIUS_KM if along < 0 else math.inf


def segment_limits(start, duration, exit_s, entry_s):
    """Partition sunlight into assumed mission slots without scaling time."""
    cuts = sorted(
        {
            start,
            start + duration,
            *(
                t
                for t in (
                    exit_s,
                    exit_s + 10,
                    exit_s + 20,
                    entry_s - 20,
                    entry_s - 10,
                    entry_s,
                )
                if start < t < start + duration
            ),
        }
    )
    limits = []
    for left, right in zip(cuts, cuts[1:]):
        middle = (left + right) / 2
        if not exit_s <= middle < entry_s:
            continue
        distance = min(middle - exit_s, entry_s - middle)
        cap = 0.030 if distance < 10 else 0.05 if distance < 20 else 0.1
        limits.append(
            {"start_s": left - start, "end_s": right - start, "max_step_s": cap}
        )
    windows = []
    for limit in limits:
        if windows and windows[-1][1] == limit["start_s"]:
            windows[-1][1] = limit["end_s"]
        else:
            windows.append([limit["start_s"], limit["end_s"]])
    return windows, limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xeoverse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("Positive seed count required")
    sys.path.insert(0, str(args.xeoverse.resolve()))
    import numpy as np
    from experiment_utils import validate_trace, write_json
    from x_links.optical import sun_direction_teme
    from x_visualization.scenario import EPOCH, default_fleet

    args.output.mkdir(parents=True, exist_ok=False)
    fleet = default_fleet(EPOCH)
    propagator = fleet.propagator()

    def sample(offset):
        ecef, teme, _ = propagator.states(EPOCH, np.asarray([offset]))
        sun = sun_direction_teme(EPOCH + timedelta(seconds=float(offset)))
        if not np.isfinite(teme).all():
            raise ValueError("Nonfinite XeoVerse propagation")
        return shadow_margin(teme[0, 0], sun), ecef[0, 0].tolist(), teme[0, 0].tolist()

    transitions = []
    previous = sample(0)[0] < 0
    for second in range(1, 6001):
        state = sample(second)[0] < 0
        if state != previous:
            left, right = float(second - 1), float(second)
            while right - left > 0.001:
                mid = (left + right) / 2
                if (sample(mid)[0] < 0) == previous:
                    left = mid
                else:
                    right = mid
            transitions.append(
                {
                    "offset_s": (left + right) / 2,
                    "eclipsed_after": state,
                    "bracket_s": [left, right],
                }
            )
        previous = state
    exit_s = next(t["offset_s"] for t in transitions if not t["eclipsed_after"])
    entry_s = next(
        t["offset_s"]
        for t in transitions
        if t["eclipsed_after"] and t["offset_s"] > exit_s
    )
    duration = 45.0
    starts = {"sunlit": exit_s + 120, "ingress": entry_s - 33, "egress": exit_s - 12}
    hashes = {
        str(p.relative_to(args.xeoverse)): hashlib.sha256(p.read_bytes()).hexdigest()
        for package in ("x_orbit", "x_links", "x_topology", "x_visualization")
        for p in sorted((args.xeoverse / package).rglob("*.py"))
    }
    identity = {
        "git_head": subprocess.check_output(
            ["git", "-C", str(args.xeoverse), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashes,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "sgp4": __import__("sgp4").__version__,
    }
    write_json(
        args.output / "geometry.json",
        {
            "epoch": EPOCH.isoformat(),
            "satellite_index": 0,
            "fleet": "Hypothetical Walker 6 planes x 24 satellites, 550 km, 53 degrees",
            "source": identity,
            "transitions": transitions,
            "shadow_model": (
                "Cylindrical Earth shadow, spherical radius 6378.137 km; no penumbra"
            ),
            "segments": {
                name: {
                    "start_offset_s": start,
                    "samples": [
                        {
                            "t_s": float(i),
                            "eclipsed": sample(start + i)[0] < 0,
                            "ecef_km": sample(start + i)[1],
                            "teme_km": sample(start + i)[2],
                        }
                        for i in range(46)
                    ],
                }
                for name, start in starts.items()
            },
        },
    )
    geometry_hash = hashlib.sha256(
        (args.output / "geometry.json").read_bytes()
    ).hexdigest()
    for seed in range(args.seeds):
        rng = random.Random(seed)
        requests = []
        for index in range(180):
            interactive = index % 2 == 1
            arrival = rng.uniform(0, 31.5)
            requests.append(
                {
                    "id": f"r{index:04d}",
                    "arrival_s": arrival,
                    "deadline_s": min(duration, arrival + (2 if interactive else 8)),
                    "prompt_tokens": 256 if interactive else 2048,
                    "output_tokens": 32 if interactive else 128,
                    "priority": 0 if interactive else 1,
                    "class": "interactive" if interactive else "onboard",
                }
            )
        for scenario, start in starts.items():
            windows, limits = segment_limits(start, duration, exit_s, entry_s)
            trace = validate_trace(
                {
                    "schema_version": 1,
                    "scenario": scenario,
                    "seed": seed,
                    "duration_s": duration,
                    "windows": windows,
                    "step_limits": limits,
                    "requests": sorted(
                        requests, key=lambda r: (r["arrival_s"], r["id"])
                    ),
                    "provenance": {
                        "geometry_sha256": geometry_hash,
                        "epoch": EPOCH.isoformat(),
                        "start_offset_s": start,
                        "orbital_seconds_per_wall_second": 1,
                        "compute_assumption": (
                            "Pause inference in shadow; keep resident KV"
                        ),
                        "slot_assumption": (
                            "30 ms within 10 s of eclipse; 50 ms at 10..20 s; "
                            "otherwise 100 ms"
                        ),
                        "exclusions": (
                            "No battery/thermal validation, DVFS, ground delivery, "
                            "network emulation or model migration"
                        ),
                    },
                }
            )
            write_json(args.output / f"trace-{scenario}-{seed}.json", trace)
    print(
        json.dumps(
            {"transitions": transitions, "segments": starts, "traces": 3 * args.seeds}
        )
    )


if __name__ == "__main__":
    main()
