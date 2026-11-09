"""Compatibility alias for the centralized configuration command."""
import sys
from finedge import main
if __name__ == "__main__":
    raise SystemExit(main(["configure", *sys.argv[1:]]))
