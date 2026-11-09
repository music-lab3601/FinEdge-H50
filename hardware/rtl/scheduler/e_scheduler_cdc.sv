`timescale 1ns/1ps
module e_scheduler_cdc #(
 parameter integer STREAMS=2,Q_DEPTH=8,CDC_DEPTH=4,SLOTS=4,PER_STREAM_SLOTS=1,
 parameter [STREAMS-1:0] ACTIVE_STREAM_MASK={STREAMS{1'b1}},
 parameter integer SW=(SLOTS>1?$clog2(SLOTS):1)
)(
 input wire reset_n,clk,input wire [STREAMS-1:0] source_clks,
 input wire [STREAMS-1:0] source_valid,output wire [STREAMS-1:0] source_ready,
 input wire [STREAMS*1280-1:0] source_desc,
 input wire arm_valid,output wire arm_ready,input wire [63:0] arm_session,input wire io_quiescent,
 input wire [STREAMS-1:0] invalidate,recover,input wire [STREAMS*64-1:0] stream_segments,
 output wire write_valid,input wire write_ready,output wire [SW-1:0] write_slot,
 output wire [63:0] write_session,write_generation,output wire [1279:0] write_desc,
 input wire commit_valid,output wire commit_ready,input wire [SW-1:0] commit_slot,
 input wire [63:0] commit_session,commit_generation,input wire commit_ok,
 output wire npu_valid,input wire npu_ready,output wire [SW-1:0] npu_slot,
 output wire [63:0] npu_session,npu_generation,output wire [1279:0] npu_desc,
 input wire done_valid,output wire done_ready,input wire [SW-1:0] done_slot,
 input wire [63:0] done_session,done_generation,input wire [127:0] done_result,
 output wire event_valid,input wire event_ready,output wire [1279:0] event_desc,
 output wire event_has_slot,output wire [SW-1:0] event_slot,
 output wire [63:0] event_session,event_generation,event_age,
 output wire [7:0] event_reason,output wire [127:0] event_result,
 output wire active,fault,output wire [63:0] cycle,protocol_errors,
 output wire [STREAMS-1:0] stream_bad,stream_busy,
 output wire [STREAMS*32-1:0] stream_inflight,
 output wire [SLOTS*3-1:0] slot_states,
 output wire [STREAMS*32-1:0] queue_levels,queue_high_water,
 output wire [STREAMS*64-1:0] admitted,served,completed,cancelled,expired,
 output wire [STREAMS*64-1:0] blocked_cycles,wait_total,wait_max,late_started,
 output wire [STREAMS*64-1:0] source_accepted,source_blocked,
 output wire [STREAMS*32-1:0] source_high_water
);
 wire [STREAMS-1:0] qvalid,qready;wire [STREAMS*1280-1:0] qdata;
 genvar i;
 generate for(i=0;i<STREAMS;i=i+1)begin:lanes
 if(ACTIVE_STREAM_MASK[i])begin:enabled
  (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [1:0] active_sync;
  wire source_up;e_reset_sync u_source_reset(source_clks[i],reset_n,source_up);
  wire fifo_ready;wire [$clog2(CDC_DEPTH):0] level;
  reg [63:0] accepted_count,blocked_count;reg [31:0] high;
  always @(posedge source_clks[i] or negedge source_up)begin
   if(!source_up)begin active_sync<=0;accepted_count<=0;blocked_count<=0;high<=0;end
   else begin
    active_sync<={active_sync[0],active};
    if(source_valid[i]&&source_ready[i])accepted_count<=accepted_count+1;
    if(active_sync[1]&&source_valid[i]&&!source_ready[i])blocked_count<=blocked_count+1;
    if(level>high)high<=level;
   end
  end
  assign source_ready[i]=fifo_ready&&active_sync[1];
  assign source_accepted[i*64+:64]=accepted_count;assign source_blocked[i*64+:64]=blocked_count;
  assign source_high_water[i*32+:32]=high;
  e_async_fifo #(.WIDTH(1280),.DEPTH(CDC_DEPTH)) u_cdc(
   .reset_n(reset_n),.wclk(source_clks[i]),.rclk(clk),
   .w_valid(source_valid[i]&&active_sync[1]),.w_ready(fifo_ready),.w_data(source_desc[i*1280+:1280]),
   .r_valid(qvalid[i]),.r_ready(qready[i]),.r_data(qdata[i*1280+:1280]),.w_level(level));
 end else begin:disabled
  assign source_ready[i]=0;assign source_accepted[i*64+:64]=0;assign source_blocked[i*64+:64]=0;
  assign source_high_water[i*32+:32]=0;assign qvalid[i]=0;assign qdata[i*1280+:1280]=0;
 end
 end endgenerate
 e_scheduler #(.STREAMS(STREAMS),.Q_DEPTH(Q_DEPTH),.SLOTS(SLOTS),.PER_STREAM_SLOTS(PER_STREAM_SLOTS),.ACTIVE_STREAM_MASK(ACTIVE_STREAM_MASK)) u_control(
  .clk(clk),.reset_n(reset_n),.arm_valid(arm_valid),.arm_ready(arm_ready),.arm_session(arm_session),.io_quiescent(io_quiescent),
  .invalidate(invalidate),.recover(recover),.ingress_idle(~qvalid),.stream_segments(stream_segments),
  .s_valid(qvalid),.s_ready(qready),.s_desc(qdata),
  .write_valid(write_valid),.write_ready(write_ready),.write_slot(write_slot),.write_session(write_session),.write_generation(write_generation),.write_desc(write_desc),
  .commit_valid(commit_valid),.commit_ready(commit_ready),.commit_slot(commit_slot),.commit_session(commit_session),.commit_generation(commit_generation),.commit_ok(commit_ok),
  .npu_valid(npu_valid),.npu_ready(npu_ready),.npu_slot(npu_slot),.npu_session(npu_session),.npu_generation(npu_generation),.npu_desc(npu_desc),
  .done_valid(done_valid),.done_ready(done_ready),.done_slot(done_slot),.done_session(done_session),.done_generation(done_generation),.done_result(done_result),
  .event_valid(event_valid),.event_ready(event_ready),.event_desc(event_desc),.event_has_slot(event_has_slot),.event_slot(event_slot),
  .event_session(event_session),.event_generation(event_generation),.event_age(event_age),.event_reason(event_reason),.event_result(event_result),
  .active(active),.fault(fault),.cycle(cycle),.protocol_errors(protocol_errors),.stream_bad(stream_bad),.stream_busy(stream_busy),.stream_inflight(stream_inflight),.slot_states(slot_states),
  .queue_levels(queue_levels),.queue_high_water(queue_high_water),.admitted(admitted),.served(served),.completed(completed),.cancelled(cancelled),
  .expired(expired),.blocked_cycles(blocked_cycles),.wait_total(wait_total),.wait_max(wait_max),.late_started(late_started));
endmodule
