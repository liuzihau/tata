"""Keep the GPU lightly busy so a cloud idle-monitor doesn't auto-shutdown
the box before you check results in the morning.

Run it after a long job finishes; stop it (Ctrl-C) when you're back.
A hard `--minutes` cap auto-stops it so a forgotten keepalive can't burn
GPU hours indefinitely.

    python gpu_keepalive.py                      # 12 h cap, 5s-busy / 25s-idle cycle
    python gpu_keepalive.py --minutes 480        # stop after 8 h
    python gpu_keepalive.py --busy 10 --idle 20  # higher duty cycle if the
                                                 # idle-monitor still triggers

Tip: launch under tmux or nohup so an SSH disconnect doesn't kill it:
    nohup python gpu_keepalive.py > logs/keepalive.log 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time

import torch


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, default=720.0,
                    help="hard cap — auto-stop after this many minutes "
                         "(default 720 = 12 h)")
    ap.add_argument("--busy", type=float, default=5.0,
                    help="seconds of GPU work per cycle (default 5)")
    ap.add_argument("--idle", type=float, default=25.0,
                    help="seconds to sleep between bursts (default 25). "
                         "Lower this / raise --busy if your idle-monitor "
                         "still triggers.")
    ap.add_argument("--size", type=int, default=4096,
                    help="matmul dimension (default 4096 — ~moderate load)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[keepalive] no CUDA device visible — nothing to keep alive")

    dev = torch.device(args.device)
    # Two fixed operands; each burst discards the product, so values never
    # accumulate / blow up. ~size²·3 floats resident — 4096 → ~400 MiB.
    a = torch.randn(args.size, args.size, device=dev)
    b = torch.randn(args.size, args.size, device=dev)

    deadline = time.time() + args.minutes * 60.0
    print(f"[keepalive] start {_now()}  device={args.device}  "
          f"cap={args.minutes:.0f} min  "
          f"({args.busy:.0f}s busy / {args.idle:.0f}s idle per cycle). "
          f"Ctrl-C to stop.", flush=True)

    cycle = 0
    try:
        while time.time() < deadline:
            t_end = time.time() + args.busy
            while time.time() < t_end:
                _ = torch.mm(a, b)          # product discarded — no blow-up
            torch.cuda.synchronize()
            cycle += 1
            if cycle % 10 == 0:             # heartbeat every ~10 cycles
                left = (deadline - time.time()) / 60.0
                print(f"[keepalive] {_now()}  cycle {cycle}  "
                      f"{left:.0f} min left", flush=True)
            time.sleep(args.idle)
    except KeyboardInterrupt:
        print(f"\n[keepalive] stopped by user at {_now()} "
              f"(after {cycle} cycles).", flush=True)
        return
    print(f"[keepalive] hit the {args.minutes:.0f}-min cap at {_now()} — "
          f"exiting.", flush=True)


if __name__ == "__main__":
    main()
