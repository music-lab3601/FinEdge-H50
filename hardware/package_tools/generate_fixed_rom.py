"""Offline hardware ROM compiler; never consumes market records or runs a model.

Only frozen normalization and input scale are used to derive exact INT8
decision thresholds. Floating arithmetic is confined to build time.
"""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,numpy as np
PACKAGE=Path(__file__).resolve().parents[1]
PROJECT=PACKAGE/'01_工程源码/finedge_host_demo'
NORM=PROJECT/'vendor/ed4_model/normalization.json'
CONTRACT=PROJECT/'versions/realtime_integration_v1/contracts/D_native_r3/input_contract.json'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parameters():
 n,c=read(NORM),read(CONTRACT)
 return np.asarray(n['mean'],dtype='f8'),np.asarray(n['std'],dtype='f8'),np.asarray(c['numerical_quantization']['scale_f32_bits'],dtype='<u4').view('<f4')
def quantize(x, scale):
    return np.clip(np.floor(np.add(np.divide(x, scale, dtype='f4'),
                                  np.float32(.5), dtype='f4')), -128, 127).astype('i1')


def _key_to_f64(keys):
    keys = np.asarray(keys, dtype='u8')
    return np.where(keys & np.uint64(1 << 63), keys ^ np.uint64(1 << 63), ~keys).view('f8')


def compile_tables():
    mu, sigma, scale = parameters()
    codes = np.broadcast_to(np.arange(-127, 128, dtype='i2'), (20, 255))
    pf, vf = np.arange(0, 40, 2), np.arange(1, 40, 2)
    def from_y(y, features):
        return quantize(((y - mu[features, None]) / sigma[features, None]).astype('f4'),
                        scale[features, None])
    # Exact first representable binary64 relative value producing each next code.
    lo = np.full((20, 255), 0x400fffffffffffff, dtype='u8')  # ordered(-1.0)
    hi = np.full((20, 255), 0xbff0000000000000, dtype='u8')  # ordered(+1.0)
    assert np.all(from_y(_key_to_f64(lo), pf) < codes)
    assert np.all(from_y(_key_to_f64(hi), pf) >= codes)
    while np.any(lo != hi):
        mid = lo + (hi - lo) // np.uint64(2)
        hit = from_y(_key_to_f64(mid), pf) >= codes
        hi = np.where(hit, mid, hi)
        lo = np.where(hit, lo, mid + np.uint64(1))
    first = _key_to_f64(lo)
    previous = np.nextafter(first, -np.inf)
    assert np.all(from_y(first, pf) >= codes)
    assert np.all(from_y(previous, pf) < codes)
    cutoffs = [[(Fraction(float(previous[j, k])) + Fraction(float(first[j, k]))) / 2
                for k in range(255)] for j in range(20)]
    fraction_bits = max(t.denominator.bit_length() - 1 for row in cutoffs for t in row)
    thresholds = [[int(t * (1 << fraction_bits)) for t in row] for row in cutoffs]
    inclusive = ((first.view('u8') & np.uint64(1)) == 0).astype('u1')
    # Raw uint64 quantities: inverse full uint64->f64->norm->f32->D3 mapping.
    lo = np.zeros((20, 255), dtype='u8')
    hi = np.full((20, 255), np.iinfo('u8').max, dtype='u8')
    assert np.all(from_y(hi.astype('f8'), vf) >= codes)
    while np.any(lo != hi):
        mid = lo + (hi - lo) // np.uint64(2)
        hit = from_y(mid.astype('f8'), vf) >= codes
        hi = np.where(hit, mid, hi)
        lo = np.where(hit, lo, mid + np.uint64(1))
    qty = lo
    assert np.all(from_y(qty.astype('f8'), vf) >= codes)
    lower = np.maximum(qty, np.uint64(1)) - np.uint64(1)
    assert np.all((qty == 0) | (from_y(lower.astype('f8'), vf) < codes))
    width = max(abs(t).bit_length() + 1 for row in thresholds for t in row)
    return dict(schema='H50_exact_integer_preprocess_v1',
                normalization_sha256=sha(NORM), input_contract_sha256=sha(CONTRACT),
                fractional_bits=fraction_bits, threshold_width=width,
                compare_width=34 + fraction_bits,
                quantity_threshold_width=max(int(v).bit_length() for v in qty.flat),
                price_thresholds=thresholds, price_inclusive=inclusive.tolist(),
                quantity_thresholds=[[int(t) for t in row] for row in qty],
                price_domain='all 20 prices uint32; best ask/bid nonzero',
                quantity_domain='uint64',
                rounding='exact current binary64 RNE operations and final binary32 D3 div/add/floor',
                runtime_floating_point=False, approximation=False)


def write_tables():
    tables = compile_tables()
    rtl = OUTPUT / 'rtl'
    rtl.mkdir(parents=True, exist_ok=True)
    tw = tables['threshold_width']
    for lane in range(4):
        words = []
        for group in range(10):
            f = 4 * group + lane
            for k in range(256):
                if k == 255:
                    word = 0
                elif lane & 1:
                    word = tables['quantity_thresholds'][f//2][k]
                else:
                    word = ((tables['price_thresholds'][f//2][k] & ((1 << tw)-1)) << 1
                            | tables['price_inclusive'][f//2][k])
                width = tables['quantity_threshold_width'] if lane & 1 else tw + 1
                words.append(f'{word:0{(width+3)//4}x}\n')
        (rtl / f'fixed_lane{lane}.hex').write_text(''.join(words), encoding='ascii')
    (OUTPUT / 'contract.json').write_text(json.dumps(tables, indent=2) + '\n', encoding='utf-8')
    (rtl / 'fixed_parameters.vh').write_text(
        f'localparam FP_FRAC={tables["fractional_bits"]}, FP_TW={tw}, FP_CW={tables["compare_width"]}, FP_QW={tables["quantity_threshold_width"]};\n',
        encoding='ascii')
    print(json.dumps({k: v for k, v in tables.items() if not k.endswith(('thresholds','inclusive'))}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 OUTPUT=a.out.resolve();OUTPUT.mkdir(parents=True,exist_ok=False)
 write_tables()
