# Generated from contract.json. Do not edit.
import struct
CONTRACT_SHA256 = "81ffd0d31b348281160ca2d8f0d94ec2f9b5cc511cfb19425727b9584928a73c"
EVENT_MAGIC = 0x31454d46
BOOK_MAGIC = 0x314b4246
FMEHEADER = struct.Struct("<IHHQQQ")
FMEHEADER_FIELDS = [('magic', 1), ('version', 1), ('records', 1), ('session_id', 1), ('packet_seq', 1), ('first_raw_ordinal', 1)]
RAWMBO = struct.Struct("<BBHIQQqIBBBBQiI")
RAWMBO_FIELDS = [('length', 1), ('rtype', 1), ('publisher_id', 1), ('instrument_id', 1), ('ts_event', 1), ('order_id', 1), ('price', 1), ('size', 1), ('flags', 1), ('channel_id', 1), ('action', 1), ('side', 1), ('ts_recv', 1), ('ts_in_delta', 1), ('sequence', 1)]
BOOKROW = struct.Struct("<IHHQIIQQQQQQQ2Q40Q")
BOOKROW_FIELDS = [('magic', 1), ('version', 1), ('flags', 1), ('session_id', 1), ('stream_id', 1), ('invalid_reason', 1), ('segment_id', 1), ('profile_row_id', 1), ('source_raw_ordinal', 1), ('event_ts_ns', 1), ('local_accept_ticks', 1), ('local_commit_ticks', 1), ('clock_hz', 1), ('reserved', 2), ('values', 40)]

def event_packet(records, session_id, packet_seq, first_raw_ordinal):
    if not isinstance(records, bytes) or not records or len(records) % RAWMBO.size:
        raise ValueError("Expected original raw 56-byte records")
    count = len(records) // RAWMBO.size
    if not 1 <= count <= 24:
        raise ValueError("Expected 1..24 records")
    if any(type(n) is not int or not 0 < n < 2**64 for n in (session_id, packet_seq, first_raw_ordinal)):
        raise ValueError("Nonzero uint64 identity required")
    if first_raw_ordinal + count - 1 >= 2**64:
        raise ValueError("Raw ordinal overflow")
    return FMEHEADER.pack(EVENT_MAGIC, 1, count, session_id, packet_seq, first_raw_ordinal) + records

def decode_event(data):
    if len(data) < FMEHEADER.size:
        raise ValueError("Short event header")
    magic, version, count, session, seq, first = FMEHEADER.unpack_from(data)
    raw = data[FMEHEADER.size:]
    if magic != EVENT_MAGIC or version != 1:
        raise ValueError("Bad magic/version")
    if not 1 <= count <= 24 or len(raw) != count * RAWMBO.size:
        raise ValueError("Invalid record count/length")
    if not session or not seq or not first or first + count - 1 >= 2**64:
        raise ValueError("Invalid identity")
    return dict(session_id=session, packet_seq=seq, first_raw_ordinal=first, records=count, raw=raw)

def decode_book(data):
    if len(data) != BOOKROW.size:
        raise ValueError("BookRow must be 416 bytes")
    unpacked = iter(BOOKROW.unpack(data)); result = {}
    for name, count in BOOKROW_FIELDS:
        result[name] = next(unpacked) if count == 1 else [next(unpacked) for _ in range(count)]
    if result["magic"] != BOOK_MAGIC or result["version"] != 1 or result["flags"] & ~1:
        raise ValueError("BookRow magic/version/flags")
    if not result["session_id"] or not result["source_raw_ordinal"] or result["stream_id"] != 0:
        raise ValueError("BookRow identity")
    if any(result["reserved"]):
        raise ValueError("Nonzero reserved field")
    valid = bool(result["flags"] & 1)
    if valid and (not result["segment_id"] or result["invalid_reason"]):
        raise ValueError("Valid book has invalid state")
    if not valid and (result["invalid_reason"] not in range(1, 7) or any(result["values"])):
        raise ValueError("Invalid book must carry reason and zero values")
    a, b = result["local_accept_ticks"], result["local_commit_ticks"]
    if a or b:
        if not result["clock_hz"] or b < a:
            raise ValueError("Local clock missing or regressing")
    return result
