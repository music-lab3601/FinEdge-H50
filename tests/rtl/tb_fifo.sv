`timescale 1ns/1ps
module tb_fifo;
 reg wc=0,rc=0;always #3.5 wc=~wc;always #5 rc=~rc;
 reg reset_n=0,wv=0,rr=0;wire wr,rv;wire [63:0] rd;wire [4:0] level;
 integer sent=0,got=0,phase=0,stall_count=0;reg stalled=0;reg [63:0] held;
 wire [63:0] value={32'(phase),32'(sent)};
 wt_async_fifo #(.WIDTH(64),.DEPTH(16)) dut(reset_n,wc,rc,wv,wr,value,rv,rr,rd,level);
 always @(negedge wc)wv=reset_n&&sent<10000&&$urandom_range(0,4)!=0;
 always @(negedge rc)rr=reset_n&&$urandom_range(0,4)>1;
 always @(posedge wc)if(reset_n)begin
  if(level>16)$fatal(1,"FIFO write level overflow");
  if(wv&&wr)sent=sent+1;
 end
 always @(posedge rc)begin
  if(!reset_n)stalled=0;
  else begin
   if(stalled&&(!rv||rd!==held))$fatal(1,"FIFO stalled output changed");
   stalled=rv&&!rr;held=rd;
   if(rv&&!rr)stall_count=stall_count+1;
   if(rv&&rr)begin
    if(rd!=={32'(phase),32'(got)})$fatal(1,"FIFO order got%h phase%0d seq%0d",rd,phase,got);
    got=got+1;
   end
  end
 end
 initial begin
  repeat(10)@(negedge wc);reset_n=1;
  wait(got>3000);@(negedge wc);reset_n=0;wv=0;rr=0;
  repeat(8)@(negedge rc);sent=0;got=0;phase=1;reset_n=1;
  wait(got==10000);repeat(10)@(negedge rc);
  if(stall_count==0||rv)$fatal(1,"FIFO tail/backpressure");
  $display("WT_FIFO_PASS words=10000 reset_while_active=1 stalls=%0d",stall_count);$finish;
 end
 initial begin #5000000;$fatal(1,"FIFO timeout %0d/%0d",got,sent);end
endmodule
