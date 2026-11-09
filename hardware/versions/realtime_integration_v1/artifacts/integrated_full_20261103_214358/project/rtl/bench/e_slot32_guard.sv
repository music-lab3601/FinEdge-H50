`timescale 1ns/1ps
// Callback index adapter: reject out-of-range32bit slots BEFORE narrowing.
module e_slot32_guard #(parameter integer SLOTS=2,SW=(SLOTS>1?$clog2(SLOTS):1))(
 input wire clk,reset_n,active,
 input wire s_valid,output wire s_ready,input wire [31:0] s_slot,
 output wire m_valid,input wire m_ready,output wire [SW-1:0] m_slot,
 input wire [SW-1:0] local_slot,output wire [31:0] wire_slot,
 output reg [63:0] rejected_count
);
 wire up;e_reset_sync u_reset(clk,reset_n,up);
 wire legal=s_slot<SLOTS;
 assign s_ready=up&&active&&(legal?m_ready:1'b1);
 assign m_valid=up&&active&&s_valid&&legal;
 assign m_slot=s_slot[SW-1:0];
 assign wire_slot={{(32-SW){1'b0}},local_slot};
 always @(posedge clk or negedge up)begin
  if(!up)rejected_count<=0;
  else if(s_valid&&s_ready&&!legal)rejected_count<=rejected_count+1;
 end
endmodule
