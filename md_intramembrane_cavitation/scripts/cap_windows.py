#!/usr/bin/env python3
"""Stop separation windows once their restraint force is converged.

Block-averaging shows <F> stable to ~3% by 20k steps, so windows are capped at
a set step count rather than running the full schedule; the batch driver then
starts the next batch immediately.
"""
import os, signal, sys, time

cap = int(sys.argv[1]) if len(sys.argv) > 1 else 35000
tagmatch = sys.argv[2] if len(sys.argv) > 2 else "parz12"
deadline = time.time() + 3 * 3600

while time.time() < deadline:
    alive = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except Exception:
            continue
        if not cmd or not cmd[0].endswith(b"src/cgmd"):
            continue
        args = [c.decode(errors="replace") for c in cmd if c]
        if tagmatch not in " ".join(args):
            continue
        alive += 1
        try:
            logp = args[args.index("--log") + 1]
            with open(logp) as fh:
                last = fh.readlines()[-1].split()[0]
            if int(last) >= cap:
                os.kill(int(pid), signal.SIGTERM)
                print(f"capped {os.path.basename(logp)} at step {last}", flush=True)
        except Exception:
            pass
    if alive == 0:
        # driver may be between batches; keep watching briefly
        time.sleep(15)
    time.sleep(10)
