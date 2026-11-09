// Generated from contract.json. Byte zero is bits [7:0]. Do not edit.
package finedge_contract_r1;
localparam string CONTRACT_SHA256 = "81ffd0d31b348281160ca2d8f0d94ec2f9b5cc511cfb19425727b9584928a73c";
localparam logic [31:0] EVENT_MAGIC = 32'h31454d46;
localparam logic [31:0] BOOK_MAGIC = 32'h314b4246;
localparam integer MAX_RECORDS = 24;
typedef struct packed {
    logic [63:0] first_raw_ordinal;
    logic [63:0] packet_seq;
    logic [63:0] session_id;
    logic [15:0] records;
    logic [15:0] version;
    logic [31:0] magic;
} FmeHeader_t;
typedef struct packed {
    logic [31:0] supplier_sequence;
    logic signed [31:0] ts_in_delta;
    logic [63:0] ts_recv;
    logic [7:0] side;
    logic [7:0] action;
    logic [7:0] channel_id;
    logic [7:0] flags;
    logic [31:0] size;
    logic signed [63:0] price;
    logic [63:0] order_id;
    logic [63:0] ts_event;
    logic [31:0] instrument_id;
    logic [15:0] publisher_id;
    logic [7:0] rtype;
    logic [7:0] length;
} RawMbo_t;
typedef struct packed {
    logic [39:0] [63:0] values;
    logic [1:0] [63:0] reserved;
    logic [63:0] clock_hz;
    logic [63:0] local_commit_ticks;
    logic [63:0] local_accept_ticks;
    logic [63:0] event_ts_ns;
    logic [63:0] source_raw_ordinal;
    logic [63:0] profile_row_id;
    logic [63:0] segment_id;
    logic [31:0] invalid_reason;
    logic [31:0] stream_id;
    logic [63:0] session_id;
    logic [15:0] flags;
    logic [15:0] version;
    logic [31:0] magic;
} BookRow_t;
endpackage
