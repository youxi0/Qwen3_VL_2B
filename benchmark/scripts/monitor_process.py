#!/usr/bin/env python3
"""执行命令并采集进程 RSS 与 Jetson 统一内存峰值。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


RAM_PATTERN = re.compile(r"RAM\s+(\d+)/(\d+)MB")
SWAP_PATTERN = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
GPU_PATTERN = re.compile(r"GR3D_FREQ\s+(\d+)%")
EMC_PATTERN = re.compile(r"EMC_FREQ\s+(\d+)%")
TEMP_PATTERN = re.compile(r"([A-Za-z0-9_]+)@([0-9.]+)C")


def read_rss_mb(pid: int) -> tuple[float, float]:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return 0.0, 0.0
    rss_kb = 0
    hwm_kb = 0
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kb = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return 0.0, 0.0
    return rss_kb / 1024.0, hwm_kb / 1024.0


def parse_tegrastats(path: Path) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            ram = RAM_PATTERN.search(line)
            if not ram:
                continue
            sample: dict[str, object] = {
                "ram_used_mb": int(ram.group(1)),
                "ram_total_mb": int(ram.group(2)),
            }
            if swap := SWAP_PATTERN.search(line):
                sample["swap_used_mb"] = int(swap.group(1))
            if gpu := GPU_PATTERN.search(line):
                sample["gpu_load_percent"] = int(gpu.group(1))
            if emc := EMC_PATTERN.search(line):
                sample["emc_load_percent"] = int(emc.group(1))
            temperatures = {name: float(value) for name, value in TEMP_PATTERN.findall(line)}
            if temperatures:
                sample["temperatures_c"] = temperatures
            samples.append(sample)

    if not samples:
        return {"samples": 0}

    baseline = int(samples[0]["ram_used_mb"])
    peak = max(int(sample["ram_used_mb"]) for sample in samples)
    max_temp = max(
        (max(sample.get("temperatures_c", {"none": 0.0}).values()) for sample in samples),
        default=0.0,
    )
    return {
        "samples": len(samples),
        "system_ram_baseline_mb": baseline,
        "system_ram_peak_used_mb": peak,
        "system_ram_peak_delta_mb": peak - baseline,
        "system_ram_total_mb": int(samples[0]["ram_total_mb"]),
        "swap_peak_used_mb": max(int(sample.get("swap_used_mb", 0)) for sample in samples),
        "gpu_load_peak_percent": max(int(sample.get("gpu_load_percent", 0)) for sample in samples),
        "emc_load_peak_percent": max(int(sample.get("emc_load_percent", 0)) for sample in samples),
        "temperature_peak_c": max_temp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--tegrastats", required=True, type=Path)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("缺少待执行命令")

    for path in (args.metrics, args.stdout, args.stderr, args.tegrastats):
        path.parent.mkdir(parents=True, exist_ok=True)

    with args.stdout.open("w", encoding="utf-8") as stdout_file, args.stderr.open(
        "w", encoding="utf-8"
    ) as stderr_file, args.tegrastats.open("w", encoding="utf-8") as tegra_file:
        tegra = subprocess.Popen(
            ["tegrastats", "--interval", str(args.interval_ms)],
            stdout=tegra_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start = time.monotonic()
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, text=True)
        peak_rss_mb = 0.0
        peak_hwm_mb = 0.0
        while process.poll() is None:
            rss_mb, hwm_mb = read_rss_mb(process.pid)
            peak_rss_mb = max(peak_rss_mb, rss_mb)
            peak_hwm_mb = max(peak_hwm_mb, hwm_mb)
            time.sleep(0.05)
        exit_code = process.wait()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        tegra.terminate()
        try:
            tegra.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            tegra.kill()
            tegra.wait()

    result: dict[str, object] = {
        "command": command,
        "exit_code": exit_code,
        "elapsed_ms": elapsed_ms,
        "process_peak_rss_mb": max(peak_rss_mb, peak_hwm_mb),
    }
    result.update(parse_tegrastats(args.tegrastats))
    args.metrics.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
