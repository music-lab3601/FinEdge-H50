`timescale 1ns/1ps
// Bundled-data, single-entry CDC mailbox. Reset ONLY by common session POR.
// Data remains in hold until the destination has accepted its captured copy.
module b1_reset_sync(input wire clk,por_n,output wire up);
 (* ASYNC_REG="TRUE" *) reg [1:0] sync;
 always @(posedge clk or negedge por_n)
  if(!por_n)sync<=0;else sync<={sync[0],1'b1};
 assign up=sync[1];
endmodule

module b1_mailbox #(parameter WIDTH=32)(
 input wire por_n,input wire s_clk,input wire s_valid,output wire s_ready,
 input wire [WIDTH-1:0] s_data,input wire m_clk,
 output reg m_valid,input wire m_ready,output reg [WIDTH-1:0] m_data
);
 reg [WIDTH-1:0] hold;
 reg request_toggle,ack_toggle;
 wire s_up,m_up;
 b1_reset_sync rs(s_clk,por_n,s_up),rm(m_clk,por_n,m_up);
 (* ASYNC_REG="TRUE" *) reg [1:0] ack_sync,request_sync;
 assign s_ready=s_up && request_toggle==ack_sync[1];
 always @(posedge s_clk or negedge s_up) begin
  if(!s_up)begin request_toggle<=0;hold<=0;ack_sync<=0;end
  else begin
   ack_sync<={ack_sync[0],ack_toggle};
   if(s_valid && s_ready)begin hold<=s_data;request_toggle<=~request_toggle;end
  end
 end
 always @(posedge m_clk or negedge m_up) begin
  if(!m_up)begin request_sync<=0;ack_toggle<=0;m_valid<=0;m_data<=0;end
  else begin
   request_sync<={request_sync[0],request_toggle};
   if(!m_valid && request_sync[1]!=ack_toggle)begin m_data<=hold;m_valid<=1;end
   if(m_valid && m_ready)begin m_valid<=0;ack_toggle<=request_sync[1];end
  end
 end
endmodule
