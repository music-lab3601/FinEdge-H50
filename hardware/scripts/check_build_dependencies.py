"""Compatibility alias for the FPGA dependency preflight."""
import json
from hardware.scripts.build_fpga import dependencies
if __name__ == "__main__":
    result=dependencies()
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["status"]=="passed" else 1)
