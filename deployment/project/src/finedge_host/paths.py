from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
PACKAGE = ROOT / "vendor" / "ed4_model"
RAW = ROOT / "data" / "raw1024"
ARTIFACTS = ROOT / "artifacts"
MANIFEST_SHA = "bf8244237ac5fc4726b33fe2aee49a15b6f5ac2fdb46485bbd6289933fbdabee"
RAW_MANIFEST_SHA = "408b8cb0af81b3a040cf22eef384add6ec9ce4bd48b480b6994353973cda7ae8"
MODEL_ZIP = WORKSPACE / "TCN_ED4_Seed42_Development_Delivery_20261018.zip"
RAW_ZIP = WORKSPACE / "TCN_ED4_seed42_RAW1024_Hardware_Handoff_20261020.zip"
CLASS_NAMES = ("down", "stationary", "up")
CLASS_CN = ("跌", "平", "涨")

