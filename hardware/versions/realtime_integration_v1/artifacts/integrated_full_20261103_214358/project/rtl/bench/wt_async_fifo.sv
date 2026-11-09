`timescale 1ns/1ps
// Dual-clock RAM FIFO with one registered prefetch word. The read pointer
// advances only after its RAM word has been copied into the output register.
module wt_async_fifo #(
 parameter integer WIDTH=545,DEPTH=512,AW=$clog2(DEPTH),parameter STYLE="block"
)(input wire reset_n,wclk,rclk,input wire w_valid,output wire w_ready,
 input wire [WIDTH-1:0] w_data,output reg r_valid,input wire r_ready,
 output reg [WIDTH-1:0] r_data,output wire [AW:0] w_level);
 wire wu,ru;e_reset_sync wr_reset(wclk,reset_n,wu);e_reset_sync rd_reset(rclk,reset_n,ru);
 (* ram_style=STYLE *) reg [WIDTH-1:0] mem[0:DEPTH-1];
 reg [AW:0] wb,rb;
 (* DONT_TOUCH="TRUE" *) reg [AW:0] wg,rg;
 (* ASYNC_REG="TRUE",SHREG_EXTRACT="NO" *) reg [AW:0] rg1,rg2,wg1,wg2;
 reg full;
 wire put=w_valid&&w_ready;
 wire fetch=ru&&(!r_valid||r_ready)&&(rg!=wg2);
 wire [AW:0] wn=wb+put,rn=rb+fetch;
 wire [AW:0] wgn=(wn>>1)^wn,rgn=(rn>>1)^rn;
 function [AW:0] ungray(input [AW:0] x);
  integer j;begin ungray[AW]=x[AW];for(j=AW-1;j>=0;j=j-1)ungray[j]=ungray[j+1]^x[j];end
 endfunction
 assign w_ready=wu&&!full;
 assign w_level=wb-ungray(rg2);
 always @(posedge wclk)if(put)mem[wb[AW-1:0]]<=w_data;
 always @(posedge rclk)if(fetch)r_data<=mem[rb[AW-1:0]];
 always @(posedge wclk or negedge wu)begin
  if(!wu)begin wb<=0;wg<=0;rg1<=0;rg2<=0;full<=0;end
  else begin
   rg1<=rg;rg2<=rg1;wb<=wn;wg<=wgn;
   full<=wgn=={~rg2[AW:AW-1],rg2[AW-2:0]};
  end
 end
 always @(posedge rclk or negedge ru)begin
  if(!ru)begin rb<=0;rg<=0;wg1<=0;wg2<=0;r_valid<=0;end
  else begin
   wg1<=wg;wg2<=wg1;rb<=rn;rg<=rgn;
   if(fetch)r_valid<=1;else if(r_valid&&r_ready)r_valid<=0;
  end
 end
 initial if(DEPTH<4||(DEPTH&(DEPTH-1))!=0)$fatal(1,"FIFO depth must be power of two >=4");
endmodule
