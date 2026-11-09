`timescale 1ns/1ps
// Direct ring history -> whole-window FIFO ->32 independently owned DDR slots.
// INPUT_ENCODED=1 receives fp_fixed_record: freshly encoded integer books.
module tt_native_transport #(
 parameter ROM0="fixed_lane0.hex",ROM1="fixed_lane1.hex",ROM2="fixed_lane2.hex",ROM3="fixed_lane3.hex",
 parameter integer INPUT_ENCODED=0,WATCHDOG=1048576,SLOTS=32
)(
 input wire clk,rst,input wire arm_valid,output wire arm_ready,input wire io_quiescent,
 input wire [63:0] cfg_session,cfg_max_wait,input wire [255:0] cfg_quant_hash,cfg_model_hash,
 input wire s_valid,output wire s_ready,input wire [1279:0] s_data,input wire [383:0] s_id,
 input wire s_emit,s_clear,input wire [63:0] s_window_id,input wire frame_credit,
 output wire offer_valid,input wire offer_ready,output wire [1279:0] offer_desc,output wire [63:0] offer_source_tick,
 input wire write_valid,output wire write_ready,input wire [63:0] write_session,write_generation,
 input wire [31:0] write_slot,input wire [1279:0] write_desc,
 input wire [SLOTS-1:0] allocation_approved,input wire [SLOTS*64-1:0] allocation_bases,allocation_capacities,
 output wire commit_valid,input wire commit_ready,output wire commit_ok,output wire [159:0] commit_token,
 input wire terminal_valid,terminal_sink_ready,output wire terminal_ready,
 input wire [1279:0] terminal_desc,input wire terminal_has_slot,input wire [159:0] terminal_token,
 output wire mem_valid,input wire mem_ready,output wire [31:0] mem_addr,
 output wire [511:0] mem_data,output wire [63:0] mem_keep,output wire mem_last,
 output wire fence_valid,input wire fence_ready,output wire [159:0] fence_token,
 input wire fence_done,input wire [159:0] fence_done_token,input wire fence_error,external_fault,
 output wire quarantine_valid,output wire [159:0] quarantine_token,
 input wire drain_proven,input wire [159:0] drain_token,
 output wire active,output wire [3:0] context_states,
 output wire decision_valid,output wire [7:0] decision_reason,output wire [63:0] decision_window,
 output wire [127:0] decision_cookie,
 output wire [63:0] records,opportunities,gate_selected,gate_skipped,local_admitted,
 output wire [63:0] rejected_copy_busy,rejected_full,rejected_epoch,E_admitted,retired,
 output wire [63:0] features_encoded,records_encoded,invalid_encoded,protocol_errors,output wire protocol_fault,output wire [63:0] scratch_released
);
 wire encoded_valid,encoded_ready;wire [319:0] encoded_data;wire [383:0] encoded_id;
 wire encoded_emit,encoded_clear;wire [63:0] encoded_window;wire encoder_ready;
 generate if(INPUT_ENCODED==0)begin:encode
  tt_encode_native_record #(.ROM0(ROM0),.ROM1(ROM1),.ROM2(ROM2),.ROM3(ROM3)) encoder(
   .clk(clk),.rst(rst),.s_valid(s_valid&&active),.s_ready(encoder_ready),.s_data(s_data),.s_id(s_id),
   .s_emit(s_emit),.s_clear(s_clear),.s_window_id(s_window_id),.m_valid(encoded_valid),.m_ready(encoded_ready),
   .m_data(encoded_data),.m_id(encoded_id),.m_emit(encoded_emit),.m_clear(encoded_clear),.m_window_id(encoded_window),
   .records_encoded(records_encoded),.features_encoded(features_encoded),.invalid_records(invalid_encoded));
  assign s_ready=active&&encoder_ready;
 end else begin:encoded_fixed
  assign encoded_valid=s_valid&&active;assign s_ready=encoded_ready&&active;assign encoded_data=s_data[319:0];
  assign encoded_id=s_id;assign encoded_emit=s_emit;assign encoded_clear=s_clear;assign encoded_window=s_window_id;
  reg [63:0] encoded_count;
  always @(posedge clk)begin
   if(rst)encoded_count<=0;
   else if(s_valid&&s_ready&&!s_clear)encoded_count<=encoded_count+1;
  end
  assign records_encoded=encoded_count;assign features_encoded=encoded_count*64'd40;assign invalid_encoded=0;
 end endgenerate
 wire [127:0] lookup_cookie;wire lookup_found,lookup_bindable,lookup_context;
 wire [1279:0] lookup_desc;wire [383:0] lookup_first,lookup_end;
 wire bind_valid,bind_context,committed_valid,committed_context;
 wire [159:0] bind_token,committed_token;
 wire read_enable,read_context;wire [6:0] read_row;wire [319:0] read_data;
 wire adapter_write_ready;
 wire local_terminal_ready;wire [63:0] history_errors,adapter_errors;
 assign protocol_errors=history_errors+adapter_errors;assign protocol_fault=(|history_errors)||(|adapter_errors);
 assign terminal_ready=local_terminal_ready&&terminal_sink_ready;
 wt_history history(
  .clk(clk),.rst(rst),.arm_valid(arm_valid),.arm_ready(arm_ready),.io_quiescent(io_quiescent),
  .cfg_session(cfg_session),.cfg_max_wait(cfg_max_wait),.cfg_quant_hash(cfg_quant_hash),.cfg_model_hash(cfg_model_hash),
  .s_valid(encoded_valid),.s_ready(encoded_ready),.s_data(encoded_data),.s_id(encoded_id),
  .s_emit(encoded_emit),.s_clear(encoded_clear),.s_window_id(encoded_window),.window_ready(adapter_write_ready&&frame_credit),
  .offer_valid(offer_valid),.offer_ready(offer_ready),.offer_desc(offer_desc),.offer_source_tick(offer_source_tick),
  .lookup_cookie(lookup_cookie),.lookup_found(lookup_found),.lookup_bindable(lookup_bindable),.lookup_context(lookup_context),
  .lookup_desc(lookup_desc),.lookup_first(lookup_first),.lookup_end(lookup_end),
  .bind_valid(bind_valid),.bind_context(bind_context),.bind_token(bind_token),
  .committed_valid(committed_valid),.committed_context(committed_context),.committed_token(committed_token),
  .terminal_valid(terminal_valid&&terminal_sink_ready),.terminal_ready(local_terminal_ready),
  .terminal_cookie(terminal_desc[1247:1120]),.terminal_has_slot(terminal_has_slot),.terminal_token(terminal_token),
  .read_enable(read_enable),.read_context(read_context),.read_row(read_row),.read_data(read_data),
  .active(active),.history_count(),.context_states(context_states),
  .decision_valid(decision_valid),.decision_reason(decision_reason),.decision_window(decision_window),.decision_cookie(decision_cookie),
  .records(records),.opportunities(opportunities),.gate_selected(gate_selected),.gate_skipped(gate_skipped),
  .local_admitted(local_admitted),.rejected_copy_busy(rejected_copy_busy),.rejected_full(rejected_full),.rejected_epoch(rejected_epoch),
  .E_admitted(E_admitted),.retired(retired),.history_resets(),.invalid_records(),.protocol_errors(history_errors),.copy_rows(),.config_errors(),.scratch_released(scratch_released));
 wire start_valid,start_ready,pack_context,payload_valid,payload_ready,payload_last,local_reset;
 wire [511:0] payload_data;wire [6:0] payload_index;

 assign write_ready=active&&adapter_write_ready;
 tt_native_packer packer(.clk(clk),.rst(local_reset),.start_valid(start_valid),.start_ready(start_ready),.start_context(pack_context),
  .read_enable(read_enable),.read_context(read_context),.read_row(read_row),.read_data(read_data),
  .m_valid(payload_valid),.m_ready(payload_ready),.m_data(payload_data),.m_last(payload_last),.m_index(payload_index),
  .windows(),.read_cycles(),.blocked_cycles());
 tt_E_write_adapter #(.WATCHDOG(WATCHDOG),.SLOTS(SLOTS)) adapter(
  .clk(clk),.rst(rst),.allocation_approved(allocation_approved),.allocation_bases(allocation_bases),.allocation_capacities(allocation_capacities),
  .write_valid(write_valid&&active),.write_ready(adapter_write_ready),.write_session(write_session),.write_generation(write_generation),
  .write_slot(write_slot),.write_desc(write_desc),.lookup_cookie(lookup_cookie),.lookup_found(lookup_found),
  .lookup_bindable(lookup_bindable),.lookup_context(lookup_context),.lookup_desc(lookup_desc),.lookup_first(lookup_first),.lookup_end(lookup_end),
  .bind_valid(bind_valid),.bind_context(bind_context),.bind_token(bind_token),
  .committed_valid(committed_valid),.committed_context(committed_context),.committed_token(committed_token),
  .pack_start_valid(start_valid),.pack_start_ready(start_ready),.pack_context(pack_context),
  .payload_valid(payload_valid),.payload_ready(payload_ready),.payload_data(payload_data),.payload_last(payload_last),.payload_index(payload_index),
  .mem_valid(mem_valid),.mem_ready(mem_ready),.mem_addr(mem_addr),.mem_data(mem_data),.mem_keep(mem_keep),.mem_last(mem_last),
  .fence_valid(fence_valid),.fence_ready(fence_ready),.fence_token(fence_token),
  .fence_done(fence_done),.fence_done_token(fence_done_token),.fence_error(fence_error),.external_fault(external_fault||(|history_errors)),
  .commit_valid(commit_valid),.commit_ready(commit_ready),.commit_ok(commit_ok),.commit_token(commit_token),
  .quarantine_valid(quarantine_valid),.quarantine_token(quarantine_token),.drain_proven(drain_proven),.drain_token(drain_token),
  .local_datapath_reset(local_reset),.requests(),.successful(),.failed_drained(),.isolations(),.protocol_errors(adapter_errors));
endmodule
