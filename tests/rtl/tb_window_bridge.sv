`timescale 1ns/1ps
module tb_window_bridge #(parameter integer RESPONSE_DELAY=3);
 reg c=0,d=0;always #4 c=~c;always #2.5 d=~d;
 reg por=0,fv=0,fl=0,cv=0;wire fr,credit,cr,done;reg [31:0] fa;reg [511:0] fd;reg [159:0] ct;wire [159:0] dt;
 reg ov=0,ow=0;reg [31:0] oa=32'h80000;reg [511:0] od=0;wire ore,orp;wire [511:0] ord;
 wire bv,br,bw,bfast;wire [31:0] ba;wire [511:0] bd;reg breply=0;reg [511:0] brd=0;wire fault;
 wire [63:0] written,read_bytes,controls,peak,capture,lastcap,lasttransfer,retries;
 g_payload_bridge #(.DEPTH(256),.TIMEOUT(8000)) dut(por,c,d,fv,fr,fa,fd,fl,credit,cv,cr,ct,done,dt,
  ov,ore,oa,ow,od,orp,ord,bv,br,bw,bfast,ba,bd,breply,brd,fault,written,read_bytes,controls,peak,capture,lastcap,lasttransfer,retries);
 reg [511:0] ram[0:2047],expected[0:2047],queued_data[0:2047];integer visible_at[0:2047];
 integer tick=0,latency=0,frames=0,reads=0,writes=0,completed=0,outcount=0;
 reg [31:0] request_address;reg request_write,request_fast;reg [511:0] request_data;
 reg held_valid=0;reg [545:0] held_request;integer k,idx,frame,group,lane,word,start_cycle=0,capture_end=0;
 reg corrupt=0,allow_fault=0;
 integer proof_tick=0,prefetch_tick=0,min_compare_age=1000000,compare_checks=0;
 reg prefetch_pending=0,reference_held_valid=0;reg [511:0] reference_held;
 always @(posedge d)begin
  proof_tick=proof_tick+1;
  if(!por)begin prefetch_pending=0;reference_held_valid=0;end
  else begin
   if(prefetch_pending)begin reference_held=dut.expected_q;reference_held_valid=1;prefetch_pending=0;end
   if(dut.state==2||dut.state==5)begin prefetch_tick=proof_tick;prefetch_pending=1;reference_held_valid=0;end
   if(dut.state==7&&dut.payload_reply)begin
    if(proof_tick-prefetch_tick<2||!reference_held_valid||dut.expected_q!==reference_held)
     $fatal(1,"Expected RAM/control multicycle contract violated");
    if(proof_tick-prefetch_tick<min_compare_age)min_compare_age=proof_tick-prefetch_tick;
    compare_checks=compare_checks+1;
   end
  end
 end
 assign br=por&&latency==0&&tick%7!=0;
 function [511:0] pattern(input integer frame_id,input integer beat);
  integer j;begin for(j=0;j<16;j=j+1)pattern[j*32+:32]=32'h6ad30000^(frame_id<<12)^(beat<<4)^j;end
 endfunction
 always @(posedge d)begin
  tick=tick+1;breply<=0;
  for(k=0;k<2048;k=k+1)if(visible_at[k]!=0&&tick>=visible_at[k])begin ram[k]<=queued_data[k];visible_at[k]=0;end
  if(!por)begin latency=0;held_valid=0;end
  else begin
   if(held_valid&&(!bv||{bfast,bw,ba,bd}!==held_request))$fatal(1,"DDR request changed under stall");
   held_valid=bv&&!br;held_request={bfast,bw,ba,bd};
   if(bv&&br)begin
    request_address=ba;request_write=bw;request_fast=bfast;request_data=bd;latency=RESPONSE_DELAY;
    if(RESPONSE_DELAY==0)begin breply<=1;brd<=ram[ba>>6];end
    if(bw)begin
     if(!bfast)$fatal(1,"Unexpected control write in payload test");
     idx=ba>>6;
     if(bd!==expected[idx])$fatal(1,"DDR reorder/data mismatch addr%h",ba);
     queued_data[idx]=bd;visible_at[idx]=tick+((idx%128==74)?1000:7);writes=writes+1;
    end else reads=reads+1;
   end else if(latency>0)begin
    latency=latency-1;
    if(latency==0)begin breply<=1;brd<=ram[request_address>>6];end
   end
  end
 end
 always @(posedge c)if(por)begin
  if(fault&&!allow_fault)$fatal(1,"Unexpected FIFO/window fault local%0d engine%0d state%0d loaded%0d address%h",dut.local_fault,dut.engine_fault,dut.state,dut.loaded,dut.load_address);
  if(done)begin
   if(dt!=={64'd77,64'(completed+1),32'(completed)})$fatal(1,"Window completion identity/order %h",dt);
   for(word=0;word<75;word=word+1)if(ram[completed*128+word]!==expected[completed*128+word])$fatal(1,"Published before full DDR visibility");
   completed=completed+1;
  end
  if(orp)begin if(ord!==512'habcdef)$fatal(1,"Separate output channel");outcount=outcount+1;end
 end
 task send_frame(input integer n,input integer malformed);
  integer send_index;
  begin
   for(group=0;group<25;group=group+1)for(lane=0;lane<3;lane=lane+1)begin
    send_index=lane*25+group;
    @(negedge c);fv=1;fa=n*8192+(malformed&&group==24&&lane==2?0:send_index*64);fd=pattern(n,send_index);fl=group==24&&lane==2;
    do @(posedge c);while(!fr);@(negedge c);fv=0;
   end
   ct={64'd77,64'(n+1),32'(n)};cv=1;do @(posedge c);while(!cr);@(negedge c);cv=0;
  end
 endtask
 initial begin
  for(k=0;k<2048;k=k+1)begin ram[k]=0;expected[k]=0;visible_at[k]=0;end
  ram[2048-1]=512'habcdef;oa=(2048-1)*64;
  for(frame=0;frame<8;frame=frame+1)for(word=0;word<75;word=word+1)expected[frame*128+word]=pattern(frame,word);
  repeat(10)@(negedge c);por=1;repeat(10)@(negedge c);
  for(frame=0;frame<8;frame=frame+1)send_frame(frame,0);
  wait(completed==8);repeat(8)@(negedge c);
  if(written!=8*4800||retries==0||writes!=8*75||outcount<1)$fatal(1,"Window byte/retry/output counters written%0d retries%0d outputs%0d",written,retries,outcount);
  allow_fault=1;
  send_frame(8,1);wait(fault);repeat(10)@(negedge c);
  if(completed!=8||credit)$fatal(1,"Malformed window published");
  if(compare_checks==0||(RESPONSE_DELAY==0&&min_compare_age!=2))$fatal(1,"Earliest response MCP coverage missing");
  $display("WT_VERIFY_MCP_PASS response_delay=%0d checks=%0d minimum_prefetch_to_compare_cycles=%0d stable_reference=1",RESPONSE_DELAY,compare_checks,min_compare_age);
  $display("WT_BRIDGE_PASS frames=%0d writes=%0d reads=%0d verify_retries=%0d FIFO_peak=%0d last_capture_cycles=%0d last_transfer_cycles=%0d malformed_rejected=1",completed,writes,reads,retries,peak,lastcap,lasttransfer);$finish;
 end
 initial begin
  wait(por);repeat(80)@(negedge c);
  repeat(20)begin
   ov=1;ow=0;do @(posedge c);while(!ore);@(negedge c);ov=0;wait(orp);repeat(17)@(negedge c);
  end
 end
 initial begin #5000000;$fatal(1,"Window test timeout completed%0d",completed);end
endmodule
