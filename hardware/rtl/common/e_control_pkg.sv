`timescale 1ns/1ps
package e_control_pkg;
 localparam integer DESC_W=1280;
 // LSB-first, byte-zero first. No price/feature/tensor payload in this record.
 typedef struct packed {
  logic [31:0] reserved;
  logic [127:0] cookie;
  logic [255:0] model_hash,quant_hash;
  logic [31:0] flags;
  logic [15:0] layout,dtype;
  logic [31:0] storage_bytes,valid_bytes;
  logic [63:0] max_wait_ticks,event_ts_ns,last_ordinal,first_ordinal,window_id,segment;
  logic [31:0] stream;
  logic [63:0] session;
 } e_desc_t;
 localparam [2:0] FREE=0,WRITING=1,READY=2,READING=3;
 localparam [7:0] OK=0,EXPIRED=1,INVALID_STREAM=2,BAD_DESCRIPTOR=3,WRITE_FAILED=4;
endpackage
