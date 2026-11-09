`timescale 1ns/1ps
module e_reset_sync(input wire clk,reset_n,output wire up);
 (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [1:0] pipe;
 always @(posedge clk or negedge reset_n)
  if(!reset_n)pipe<=0;else pipe<={pipe[0],1'b1};
 assign up=pipe[1];
endmodule
