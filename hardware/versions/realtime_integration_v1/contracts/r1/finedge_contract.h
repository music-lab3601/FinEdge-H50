/* Generated from contract.json. Do not edit. Little-endian host required. */
#ifndef FINEDGE_CONTRACT_R1_H
#define FINEDGE_CONTRACT_R1_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#define FINEDGE_CONTRACT_SHA256 "81ffd0d31b348281160ca2d8f0d94ec2f9b5cc511cfb19425727b9584928a73c"
#define FINEDGE_EVENT_MAGIC UINT32_C(0x31454d46)
#define FINEDGE_BOOK_MAGIC UINT32_C(0x314b4246)
#define FINEDGE_MAX_RECORDS 24
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t records;
    uint64_t session_id;
    uint64_t packet_seq;
    uint64_t first_raw_ordinal;
} FinEdgeFmeHeader;
typedef struct {
    uint8_t length;
    uint8_t rtype;
    uint16_t publisher_id;
    uint32_t instrument_id;
    uint64_t ts_event;
    uint64_t order_id;
    int64_t price;
    uint32_t size;
    uint8_t flags;
    uint8_t channel_id;
    uint8_t action;
    uint8_t side;
    uint64_t ts_recv;
    int32_t ts_in_delta;
    uint32_t sequence;
} FinEdgeRawMbo;
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint64_t session_id;
    uint32_t stream_id;
    uint32_t invalid_reason;
    uint64_t segment_id;
    uint64_t profile_row_id;
    uint64_t source_raw_ordinal;
    uint64_t event_ts_ns;
    uint64_t local_accept_ticks;
    uint64_t local_commit_ticks;
    uint64_t clock_hz;
    uint64_t reserved[2];
    uint64_t values[40];
} FinEdgeBookRow;
#pragma pack(pop)
#ifdef __cplusplus
#define FE_ASSERT static_assert
#else
#define FE_ASSERT _Static_assert
#endif
FE_ASSERT(sizeof(FinEdgeFmeHeader) == 32, "FmeHeader size");
FE_ASSERT(offsetof(FinEdgeFmeHeader, magic) == 0, "FmeHeader.magic");
FE_ASSERT(offsetof(FinEdgeFmeHeader, version) == 4, "FmeHeader.version");
FE_ASSERT(offsetof(FinEdgeFmeHeader, records) == 6, "FmeHeader.records");
FE_ASSERT(offsetof(FinEdgeFmeHeader, session_id) == 8, "FmeHeader.session_id");
FE_ASSERT(offsetof(FinEdgeFmeHeader, packet_seq) == 16, "FmeHeader.packet_seq");
FE_ASSERT(offsetof(FinEdgeFmeHeader, first_raw_ordinal) == 24, "FmeHeader.first_raw_ordinal");
FE_ASSERT(sizeof(FinEdgeRawMbo) == 56, "RawMbo size");
FE_ASSERT(offsetof(FinEdgeRawMbo, length) == 0, "RawMbo.length");
FE_ASSERT(offsetof(FinEdgeRawMbo, rtype) == 1, "RawMbo.rtype");
FE_ASSERT(offsetof(FinEdgeRawMbo, publisher_id) == 2, "RawMbo.publisher_id");
FE_ASSERT(offsetof(FinEdgeRawMbo, instrument_id) == 4, "RawMbo.instrument_id");
FE_ASSERT(offsetof(FinEdgeRawMbo, ts_event) == 8, "RawMbo.ts_event");
FE_ASSERT(offsetof(FinEdgeRawMbo, order_id) == 16, "RawMbo.order_id");
FE_ASSERT(offsetof(FinEdgeRawMbo, price) == 24, "RawMbo.price");
FE_ASSERT(offsetof(FinEdgeRawMbo, size) == 32, "RawMbo.size");
FE_ASSERT(offsetof(FinEdgeRawMbo, flags) == 36, "RawMbo.flags");
FE_ASSERT(offsetof(FinEdgeRawMbo, channel_id) == 37, "RawMbo.channel_id");
FE_ASSERT(offsetof(FinEdgeRawMbo, action) == 38, "RawMbo.action");
FE_ASSERT(offsetof(FinEdgeRawMbo, side) == 39, "RawMbo.side");
FE_ASSERT(offsetof(FinEdgeRawMbo, ts_recv) == 40, "RawMbo.ts_recv");
FE_ASSERT(offsetof(FinEdgeRawMbo, ts_in_delta) == 48, "RawMbo.ts_in_delta");
FE_ASSERT(offsetof(FinEdgeRawMbo, sequence) == 52, "RawMbo.sequence");
FE_ASSERT(sizeof(FinEdgeBookRow) == 416, "BookRow size");
FE_ASSERT(offsetof(FinEdgeBookRow, magic) == 0, "BookRow.magic");
FE_ASSERT(offsetof(FinEdgeBookRow, version) == 4, "BookRow.version");
FE_ASSERT(offsetof(FinEdgeBookRow, flags) == 6, "BookRow.flags");
FE_ASSERT(offsetof(FinEdgeBookRow, session_id) == 8, "BookRow.session_id");
FE_ASSERT(offsetof(FinEdgeBookRow, stream_id) == 16, "BookRow.stream_id");
FE_ASSERT(offsetof(FinEdgeBookRow, invalid_reason) == 20, "BookRow.invalid_reason");
FE_ASSERT(offsetof(FinEdgeBookRow, segment_id) == 24, "BookRow.segment_id");
FE_ASSERT(offsetof(FinEdgeBookRow, profile_row_id) == 32, "BookRow.profile_row_id");
FE_ASSERT(offsetof(FinEdgeBookRow, source_raw_ordinal) == 40, "BookRow.source_raw_ordinal");
FE_ASSERT(offsetof(FinEdgeBookRow, event_ts_ns) == 48, "BookRow.event_ts_ns");
FE_ASSERT(offsetof(FinEdgeBookRow, local_accept_ticks) == 56, "BookRow.local_accept_ticks");
FE_ASSERT(offsetof(FinEdgeBookRow, local_commit_ticks) == 64, "BookRow.local_commit_ticks");
FE_ASSERT(offsetof(FinEdgeBookRow, clock_hz) == 72, "BookRow.clock_hz");
FE_ASSERT(offsetof(FinEdgeBookRow, reserved) == 80, "BookRow.reserved");
FE_ASSERT(offsetof(FinEdgeBookRow, values) == 96, "BookRow.values");
#undef FE_ASSERT
static inline int finedge_event_header_valid(const FinEdgeFmeHeader *h, size_t bytes) {
    return h->magic == FINEDGE_EVENT_MAGIC && h->version == 1 &&
        h->records >= 1 && h->records <= FINEDGE_MAX_RECORDS &&
        h->session_id != 0 && h->packet_seq != 0 && h->first_raw_ordinal != 0 &&
        h->first_raw_ordinal <= UINT64_MAX - (h->records - 1) &&
        bytes == sizeof(*h) + h->records * sizeof(FinEdgeRawMbo);
}
#endif
