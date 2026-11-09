import json
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from .delivery import read_json, sha256, safe_path
from .paths import RAW, RAW_MANIFEST_SHA

PREPROCESS_ID = "ed4-relative-f64-normalize-f32-v1"


@dataclass(frozen=True)
class Record:
    values: np.ndarray
    meta: dict

    @property
    def segment_key(self):
        m = self.meta
        return (m["date"], m["symbol"], m["snippet_id"], m["research_segment_id"], m["native_segment_id"])


@dataclass(frozen=True)
class Window:
    tensor: np.ndarray
    first_meta: dict
    last_meta: dict
    preprocess_ns: int
    pack_ns: int
    entered_ns: int


class RawSource:
    def __init__(self, path=RAW):
        self.path = path
        if sha256(path / "manifest.json") != RAW_MANIFEST_SHA:
            raise ValueError("Unrecognized raw handoff manifest")
        self.manifest = read_json(path / "manifest.json")
        for name in ("raw_book.u64le.bin", "row_metadata.jsonl", "screening_reject_masks.u32le.bin",
                     "normalized_rows.f32le.bin", "windows.json", "window_inputs.f32le.bin",
                     "window_logits.f32le.bin", "window_classes.u8.bin"):
            file = safe_path(path, name)
            expected = self.manifest["files"][name]
            if file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
                raise ValueError(f"Raw payload integrity failure: {name}")
        self.rows = np.fromfile(path / "raw_book.u64le.bin", dtype="<u8").reshape(-1, 40)
        self.metadata = [json.loads(line) for line in (path / "row_metadata.jsonl").read_text("utf-8").splitlines() if line.strip()]
        masks = np.fromfile(path / "screening_reject_masks.u32le.bin", dtype="<u4")
        if len(self.rows) != self.manifest["record_count"] or len(self.rows) != len(self.metadata) or len(masks) != len(self.rows):
            raise ValueError("Raw payload row counts differ")
        for i, m in enumerate(self.metadata):
            if m["sample_row_0based"] != i or m["research_reject_mask"] != int(masks[i]) or int(masks[i]) != 0:
                raise ValueError(f"Raw eligibility or row identity failure at {i}")
        self.rows.flags.writeable = False

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for row, meta in zip(self.rows, self.metadata, strict=True):
            yield Record(row, dict(meta))


class InputBuilder:
    """Transform each accepted record once; return owned channel-first windows."""
    def __init__(self, normalization, packing='legacy', profiler=None):
        self.profiler=profiler
        if packing not in ('legacy','single_copy'):raise ValueError('Unknown window packing')
        self.packing=packing
        self.mean = np.asarray(normalization["mean"], dtype=np.float64)
        self.std = np.asarray(normalization["std"], dtype=np.float64)
        if self.mean.shape != (40,) or self.std.shape != (40,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Invalid forty-channel normalization")
        self.ring = np.empty((100, 40), dtype="<f4")
        self.metas = [None] * 100
        self.reset()

    def reset(self):
        self.count = self.pointer = 0
        self.previous = None
        self.last_preprocess_ns = self.last_entered_ns = 0

    def normalize(self, values):
        if not isinstance(values, np.ndarray) or values.dtype != np.dtype("<u8") or values.shape != (40,):
            raise ValueError("Expected one uint64[40] record")
        prices = values[0::2]
        if np.any(prices == 0) or np.any(prices >= 2**51):
            raise ValueError("Prices must be positive and below 2**51")
        physical = np.array(values, dtype=np.float64, copy=True)
        midpoint = (physical[0] + physical[2]) / 2.0
        physical[0::2] = (physical[0::2] - midpoint) / midpoint
        result = np.asarray((physical - self.mean) / self.std, dtype="<f4")
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite normalized values")
        return result

    def snapshot(self, preprocess_ns=None, entered_ns=None, pack_started=None):
        """Return an owned tensor. Caller synchronizes concurrent ring updates."""
        if self.count < 100:raise ValueError('Window not full')
        probe=self.profiler
        mark=probe.start() if probe is not None else None
        begin=perf_counter_ns() if pack_started is None else pack_started
        if self.packing=='legacy':
            ordered=np.concatenate((self.ring[self.pointer:],self.ring[:self.pointer]),axis=0)
            tensor=np.ascontiguousarray(ordered.T[None,...],dtype='<f4')
        else:
            tensor=np.empty((1,40,100),dtype='<f4')
            tail=100-self.pointer
            tensor[0,:,:tail]=self.ring[self.pointer:].T
            if self.pointer:tensor[0,:,tail:]=self.ring[:self.pointer].T
        result=Window(tensor,dict(self.metas[self.pointer]),dict(self.metas[(self.pointer-1)%100]),
                      self.last_preprocess_ns if preprocess_ns is None else preprocess_ns,
                      perf_counter_ns()-begin,self.last_entered_ns if entered_ns is None else entered_ns)
        if probe is not None:probe.end('tensor_pack',mark)
        return result

    def push(self, record, materialize=True):
        entered = perf_counter_ns()
        probe=self.profiler
        mark=probe.start() if probe is not None else None
        try:
            m = record.meta
            if m["research_reject_mask"] != 0 or m["bid_levels"] < 10 or m["ask_levels"] < 10:
                raise ValueError("Record is not eligible with complete top ten")
            if self.previous is not None:
                if record.segment_key != self.previous.segment_key:
                    self.reset()
                else:
                    old = self.previous.meta
                    if m["source_record_row_0based"] != old["source_record_row_0based"] + 1 or m["sample_row_0based"] != old["sample_row_0based"] + 1:
                        raise ValueError("Discontinuous record rows; window cleared")
                    if m["source_message_position_1based"] <= old["source_message_position_1based"] or m["timestamp_ns_since_exchange_midnight"] < old["timestamp_ns_since_exchange_midnight"]:
                        raise ValueError("Source message or timestamp regressed")
            if probe is not None:
                probe.end('window_validation',mark)
                mark=probe.start()
            normalized = self.normalize(record.values)
            if probe is not None:
                probe.end('normalization',mark)
                mark=probe.start()
            pre_end = perf_counter_ns()
            self.ring[self.pointer] = normalized
            self.metas[self.pointer] = dict(m)
            self.pointer = (self.pointer + 1) % 100
            self.count = min(100, self.count + 1)
            self.previous = record
            self.last_preprocess_ns,self.last_entered_ns=pre_end-entered,entered
            if probe is not None:probe.end('ring_update',mark)
            if self.count < 100:
                return None
            if not materialize:return None
            return self.snapshot(pre_end-entered,entered,pre_end)
        except (ValueError, KeyError, TypeError, OverflowError):
            self.reset()
            raise
