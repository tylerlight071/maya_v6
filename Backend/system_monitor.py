"""MAYA System Monitor — CPU, RAM, GPU, VRAM via psutil + pynvml."""
import asyncio
import contextlib
import json
import time

import websockets

MONITOR_PORT   = 8771
POLL_INTERVAL  = 2.0    # seconds between broadcasts
ALERT_COOLDOWN = 120.0  # minimum seconds between the same alert firing

_clients:       set       = set()
_current_stats: dict      = {}
_alert_message: "str | None" = None
_last_alert_at: float     = 0.0

# Prime pynvml once at import — gracefully skip if no NVIDIA GPU or library missing
_nvml_ok = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    pass


def _collect() -> dict:
    import psutil
    cpu = psutil.cpu_percent(interval=None)
    vm  = psutil.virtual_memory()
    stats: dict = {
        "cpu":        round(cpu, 1),
        "ram_used":   round(vm.used  / 1e9, 1),
        "ram_total":  round(vm.total / 1e9, 1),
        "gpu":        0.0,
        "vram_used":  0.0,
        "vram_total": 0.0,
        "gpu_temp":   0,
    }
    if _nvml_ok:
        try:
            handle           = pynvml.nvmlDeviceGetHandleByIndex(0)
            util             = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem              = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp             = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            stats["gpu"]        = round(util.gpu, 1)
            stats["vram_used"]  = round(mem.used  / 1e9, 1)
            stats["vram_total"] = round(mem.total / 1e9, 1)
            stats["gpu_temp"]   = temp
        except Exception:
            pass
    return stats


def _check_alerts(stats: dict) -> None:
    global _alert_message, _last_alert_at
    now = time.time()
    if now - _last_alert_at < ALERT_COOLDOWN:
        return

    vram_pct = (stats["vram_used"] / stats["vram_total"] * 100) if stats["vram_total"] > 0 else 0
    ram_pct  = (stats["ram_used"]  / stats["ram_total"]  * 100) if stats["ram_total"]  > 0 else 0

    if vram_pct > 90:
        _alert_message = (
            f"VRAM is at {vram_pct:.0f}% "
            f"({stats['vram_used']:.1f}/{stats['vram_total']:.1f} GB) — "
            "consider closing something before the next generation."
        )
        _last_alert_at = now
    elif ram_pct > 92:
        _alert_message = (
            f"RAM is at {ram_pct:.0f}% "
            f"({stats['ram_used']:.1f}/{stats['ram_total']:.1f} GB)."
        )
        _last_alert_at = now


async def _broadcast(msg: dict) -> None:
    raw = json.dumps(msg)
    for ws in list(_clients):
        with contextlib.suppress(Exception):
            await ws.send(raw)


async def _poll() -> None:
    global _current_stats
    import psutil
    psutil.cpu_percent(interval=None)   # prime — first call always returns 0
    await asyncio.sleep(1)
    while True:
        try:
            stats           = _collect()
            _current_stats  = stats
            _check_alerts(stats)
            await _broadcast({"type": "monitor_update", **stats})
        except Exception as e:
            print(f"[monitor] poll error: {e}", flush=True)
        await asyncio.sleep(POLL_INTERVAL)


async def handle_client(websocket) -> None:
    _clients.add(websocket)
    try:
        if _current_stats:
            await websocket.send(json.dumps({"type": "monitor_update", **_current_stats}))
        async for _ in websocket:
            pass  # no client→server messages needed
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


async def run() -> None:
    asyncio.ensure_future(_poll())
    async with websockets.serve(handle_client, "127.0.0.1", MONITOR_PORT):
        print(f"[monitor] ws://127.0.0.1:{MONITOR_PORT}", flush=True)
        await asyncio.Future()
