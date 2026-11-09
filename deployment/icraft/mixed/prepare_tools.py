"""Build the checked-in SDK utilities through the central configuration."""
from tools.build_sdk import build_stage
def main():
    build_stage("mixed")
if __name__=="__main__":
    main()
