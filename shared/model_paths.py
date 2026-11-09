"""Canonical H50 model and optional raw-delivery locations."""
import json
from shared.paths import ROOT, data_path, work_path

WORKSPACE = ROOT
PACKAGE = ROOT / 'models/cpu'
RAW = data_path('reference/raw1024')
ARTIFACTS = work_path('runtime')
MANIFEST_SHA = json.loads((ROOT / 'models/identity.json').read_text(encoding='utf-8'))['CPU_manifest_sha256']
RAW_MANIFEST_SHA = '408b8cb0af81b3a040cf22eef384add6ec9ce4bd48b480b6994353973cda7ae8'
MODEL_ZIP = data_path('reference/model_delivery.zip')
RAW_ZIP = data_path('reference/raw1024_delivery.zip')
CLASS_NAMES = ('down', 'stationary', 'up')
CLASS_CN = ('跌', '平', '涨')
