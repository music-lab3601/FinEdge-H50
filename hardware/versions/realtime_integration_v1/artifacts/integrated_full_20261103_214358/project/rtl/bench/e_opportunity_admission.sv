`timescale 1ns/1ps
// One-cycle inference-opportunity pulses, separate from raw-market flow control.
// accepted/rejected are local pulses; downstream must observe every rejection.
module e_opportunity_admission(
 input wire clk,reset_n,active,enabled,
 input wire offer_valid,input wire [1279:0] offer_desc,
 output wire queue_valid,input wire queue_ready,output wire [1279:0] queue_desc,
 output wire accepted,output wire rejected,output wire [7:0] reject_reason,
 output wire [1279:0] rejected_desc,
 output reg [63:0] offered_count,accepted_count,rejected_count
);
 wire up;e_reset_sync u_reset(clk,reset_n,up);
 // E FIFO ready is state-based, independent of valid; no combinational loop.
 // Only accepted pulses reach the FIFO; rejected opportunities are not retried.
 assign queue_valid=up&&active&&enabled&&offer_valid&&queue_ready;
 assign queue_desc=offer_desc;assign rejected_desc=offer_desc;
 assign accepted=queue_valid&&queue_ready;
 assign rejected=up&&offer_valid&&!accepted;
 assign reject_reason=!active?8'd3:!enabled?8'd2:8'd1; //not armed/disabled/full
 always @(posedge clk or negedge up)begin
  if(!up)begin offered_count<=0;accepted_count<=0;rejected_count<=0;end
  else begin
   if(offer_valid)offered_count<=offered_count+1;
   if(accepted)accepted_count<=accepted_count+1;
   if(rejected)rejected_count<=rejected_count+1;
  end
 end
endmodule
