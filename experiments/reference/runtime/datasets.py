"""Registered current-project datasets; no pipeline reference enters legacy archive."""
from pathlib import Path

import numpy as np

from software.runtime.delivery import read_json, sha256
from experiments.reference.runtime.itch import BOOK_DTYPE, DATASET, PROFILE, SOURCE_SHA, record_from_book_row
from shared.model_paths import RAW
from software.runtime.stream import Record, RawSource

DATASETS = ('delivery1024', 'nasdaq20190730', 'itch20190730', 'itch20190730_gzip')


class PreparedNasdaqSource:
    def __init__(self, path=DATASET):
        self.path = Path(path)/'prepared'
        self.manifest = read_json(self.path/'manifest.json')
        m = self.manifest
        if m['status'] != 'passed' or m['profile'] != PROFILE or m['source_sha256'] != SOURCE_SHA:
            raise ValueError('Prepared Nasdaq identity/profile is not admitted')
        for name, ref in m['files'].items():
            p = self.path/name
            if p.stat().st_size != ref['bytes'] or sha256(p) != ref['sha256']:
                raise ValueError('Prepared Nasdaq file identity mismatch: '+name)
        self.rows = np.memmap(self.path/'book_records.bin',mode='r',dtype=BOOK_DTYPE)
        self.labels = np.memmap(self.path/'labels.i8.bin',mode='r',dtype='i1')
        self.indices = np.flatnonzero(self.rows['flags'] == 0)
        if len(self.rows) != m['record_count'] or len(self.indices) != m['eligible_count'] or len(self.labels) != len(self.rows):
            raise ValueError('Prepared record counts differ')

    def __len__(self):
        return len(self.indices)

    def record(self, i):
        return record_from_book_row(self.rows[i], i)

    def __iter__(self):
        for i in self.indices:
            yield self.record(int(i))


def open_source(name):
    if name == 'delivery1024': return RawSource()
    if name == 'nasdaq20190730': return PreparedNasdaqSource()
    if name in ('itch20190730', 'itch20190730_gzip'):
        from experiments.reference.runtime.raw_itch import RawItchSource
        return RawItchSource(full_gzip=name.endswith('_gzip'))
    raise ValueError('Unknown registered dataset: '+str(name))


def dataset_info(name):
    if name in ('itch20190730', 'itch20190730_gzip'):
        return {'dataset_id':name, 'date':'2019-07-30', 'symbol':'AAPL',
                'role':'cross_period_engineering_validation', 'runtime_book_reconstruction':True,
                'description':'2019-07-30 AAPL · '+('全市场gzip解析' if name.endswith('_gzip') else 'AAPL原始ITCH事件')+
                              ' → 运行时增量建簿/组窗 · 文件背压验证，非实时网络接收'}
    if name == 'delivery1024':
        return {'dataset_id':name,'date':'2025-12-08','symbol':'AAPL','role':'train_engineering_snippet',
                'description':'2025-12-08 AAPL · 1024条 TRAIN工程片段 · 约4.5秒行情；循环不代表新行情'}
    if name == 'nasdaq20190730':
        return {'dataset_id':name,'date':'2019-07-30','symbol':'AAPL','role':'cross_period_engineering_validation',
                'description':'2019-07-30 AAPL · 原生ITCH全日重建 · 跨时期工程验证；固定模型与归一化'}
    raise ValueError('Unknown dataset')
