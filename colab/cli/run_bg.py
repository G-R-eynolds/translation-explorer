# Background driver (runs as a detached process on the VM, so a dropped CLI connection cannot stall or kill it):
# setup, then stages A, B, C1.  The notebook kernel later re-runs setup and C1 (which then finds its cache) before C2.
import traceback, sys
try:
    for f in ["00_setup.py", "A_detour.py", "B_roundtrip.py", "C1_collect.py"]:
        print(f"=== BG START {f}", flush=True)
        exec(open(f"/content/ut/{f}").read(), globals())
        print(f"=== BG OK {f}", flush=True)
    print("=== BG_DONE", flush=True)
except Exception:
    traceback.print_exc(); print("=== BG_FAILED", flush=True); sys.exit(1)
