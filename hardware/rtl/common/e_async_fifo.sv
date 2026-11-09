`timescale 1ns/1ps
// Power-of-two asynchronous FIFO; Gray pointers, not synchronized payload bits.
// One common asynchronous assertion; release synchronized in BOTH domains.
module e_async_fifo #(
 parameter integer WIDTH=1280,DEPTH=4,AW=$clog2(DEPTH)
)(
 input wire reset_n,wclk,rclk,
 input wire w_valid,output wire w_ready,input wire [WIDTH-1:0] w_data,
 output wire r_valid,input wire r_ready,output wire [WIDTH-1:0] r_data,
 output wire [AW:0] w_level
);
 wire w_up,r_up;e_reset_sync u_wreset(wclk,reset_n,w_up);e_reset_sync u_rreset(rclk,reset_n,r_up);
 (* ram_style="distributed" *) reg [WIDTH-1:0] mem[0:DEPTH-1];
 reg [AW:0] wr_bin,rd_bin;
 (* DONT_TOUCH="TRUE" *) reg [AW:0] wr_gray,rd_gray;
 (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [AW:0] rgray_sync1,rgray_sync2;
 (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [AW:0] wgray_sync1,wgray_sync2;
 reg full,empty;
 wire put=w_valid&&w_ready,get=r_valid&&r_ready;
 wire [AW:0] wn=wr_bin+put,rn=rd_bin+get;
 wire [AW:0] wg=(wn>>1)^wn,rg=(rn>>1)^rn;
 wire [AW:0] inverted_read={~rgray_sync2[AW:AW-1],rgray_sync2[AW-2:0]};
 function [AW:0] gray_to_bin(input [AW:0] gray);
  integer j;begin gray_to_bin[AW]=gray[AW];for(j=AW-1;j>=0;j=j-1)gray_to_bin[j]=gray_to_bin[j+1]^gray[j];end
 endfunction
 assign w_ready=w_up&&!full;
 assign r_valid=r_up&&!empty;
 assign r_data=mem[rd_bin[AW-1:0]];
 assign w_level=wr_bin-gray_to_bin(rgray_sync2);
 // Separate no-reset RAM write process is required for LUTRAM inference.
 always @(posedge wclk)if(put)mem[wr_bin[AW-1:0]]<=w_data;
 always @(posedge wclk or negedge w_up)begin
  if(!w_up)begin wr_bin<=0;wr_gray<=0;full<=0;rgray_sync1<=0;rgray_sync2<=0;end
  else begin
   rgray_sync1<=rd_gray;rgray_sync2<=rgray_sync1;
   wr_bin<=wn;wr_gray<=wg;full<=wg==inverted_read;
  end
 end
 always @(posedge rclk or negedge r_up)begin
  if(!r_up)begin rd_bin<=0;rd_gray<=0;empty<=1;wgray_sync1<=0;wgray_sync2<=0;end
  else begin
   wgray_sync1<=wr_gray;wgray_sync2<=wgray_sync1;
   rd_bin<=rn;rd_gray<=rg;empty<=rg==wgray_sync2;
  end
 end
 initial begin
  if(DEPTH<4 || (DEPTH&(DEPTH-1))!=0)$fatal(1,"E async FIFO DEPTH must be power-of-two >=4");
 end
endmodule
