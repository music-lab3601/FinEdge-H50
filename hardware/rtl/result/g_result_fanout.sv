`timescale 1ns/1ps
// Deliver each immutable result once to GP/control and once to network FIFO.
// Branch backpressure must never duplicate the other branch's transaction.
module g_result_fanout(input wire clk,por_n,
 input wire s_valid,output wire s_ready,input wire [383:0] s_data,
 output wire gp_valid,input wire gp_ready,output wire [383:0] gp_data,
 output wire net_valid,input wire net_ready,output wire [383:0] net_data);
 reg gp_taken,net_taken;
 wire up;
 b1_reset_sync reset125(clk,por_n,up);
 assign gp_valid=s_valid&&!gp_taken;assign net_valid=s_valid&&!net_taken;
 assign gp_data=s_data;assign net_data=s_data;
 assign s_ready=gp_taken&&net_taken;
 always @(posedge clk or negedge up)begin
  if(!up)begin gp_taken<=0;net_taken<=0;end
  else begin
   if(gp_valid&&gp_ready)gp_taken<=1;
   if(net_valid&&net_ready)net_taken<=1;
   if(s_valid&&s_ready)begin gp_taken<=0;net_taken<=0;end
  end
 end
endmodule
