`timescale 1ns/1ps
// One command and one verified completion per4800B window. Data uses a
// BRAM asynchronous FIFO; no per-beat acknowledgement crosses the domains.
module g_payload_bridge #(parameter DEPTH=512,AW=$clog2(DEPTH),TIMEOUT=262144)(
 input wire por_n,clk125,clk200,
 input wire f_valid,output wire f_ready,input wire [31:0] f_addr,input wire [511:0] f_data,input wire f_last,
 output wire frame_credit,
 input wire fence_valid,output wire fence_ready,input wire [159:0] fence_token,
 output wire fence_done,output wire [159:0] fence_done_token,
 input wire output_valid,output wire output_ready,input wire [31:0] output_addr,
 input wire output_write,input wire [511:0] output_wdata,
 output wire output_reply,output wire [511:0] output_data,
 output wire b_valid,input wire b_ready,output wire b_write,output wire b_fast_write,output wire [31:0] b_addr,output wire [511:0] b_data,
 input wire b_reply,input wire [511:0] b_rdata,output wire fault,
 output reg [63:0] written_bytes,read_bytes,control_written_bytes,
 output reg [63:0] fifo_peak,capture_cycles,last_capture_cycles,last_transfer_cycles,verify_retries);
 wire u125,u200;b1_reset_sync r125(clk125,por_n,u125);b1_reset_sync r200(clk200,por_n,u200);
 reg local_fault,engine_fault;(* ASYNC_REG="TRUE" *)reg [1:0] fault_sync;
 assign fault=local_fault||fault_sync[1];
 wire data_wr,data_rv,data_rr;wire [544:0] data_rd;wire [AW:0] level;
 reg collecting,frame_full;reg [6:0] received;reg [31:0] base;reg [63:0] capture_start,clock125;
 wire [3:0] unused_cmd_level,unused_done_level;
 wire cmd_wr,cmd_rv,cmd_rr;wire [191:0] cmd_rd;
 wire command_fire=fence_valid&&fence_ready;
 assign frame_credit=u125&&!fault&&!collecting&&!frame_full&&level<=DEPTH-75&&cmd_wr;
 assign f_ready=u125&&!fault&&!frame_full&&(collecting||frame_credit)&&data_wr;
 wire put=f_valid&&f_ready;
 assign fence_ready=u125&&!fault&&frame_full&&cmd_wr;
 wt_async_fifo #(.WIDTH(545),.DEPTH(DEPTH)) payload_fifo(por_n,clk125,clk200,put,data_wr,{f_last,f_addr,f_data},data_rv,data_rr,data_rd,level);
 wt_async_fifo #(.WIDTH(192),.DEPTH(8),.STYLE("distributed")) window_fifo(por_n,clk125,clk200,command_fire,cmd_wr,{fence_token,base},cmd_rv,cmd_rr,cmd_rd,unused_cmd_level);
 wire done_wv,done_wr,done_rv;wire [255:0] done_wdata,done_rdata;
 wt_async_fifo #(.WIDTH(256),.DEPTH(8),.STYLE("distributed")) complete_fifo(por_n,clk200,clk125,done_wv,done_wr,done_wdata,done_rv,1'b1,done_rdata,unused_done_level);
 assign fence_done=done_rv&&!fault;assign fence_done_token=done_rdata[159:0];
 // Low-latency NPU marker/output channel remains separate from the window FIFO.
 wire oc_valid,oc_ready,or_ready,oc_sready;wire [544:0] oc_data;wire [511:0] or_data;
 reg output_pending,output_is_write;
 b1_mailbox #(545) output_command(por_n,clk125,output_valid&&!output_pending&&!fault,oc_sready,{output_write,output_addr,output_wdata},clk200,oc_valid,oc_ready,oc_data);
 wire output_response_valid;
 b1_mailbox #(512) output_response(por_n,clk200,output_response_valid,or_ready,b_rdata,clk125,output_reply,1'b1,or_data);
 assign output_data=or_data;
 assign output_ready=oc_sready&&!output_pending&&!fault;
 always @(posedge clk125 or negedge u125)begin
  if(!u125)begin local_fault<=0;fault_sync<=0;collecting<=0;frame_full<=0;received<=0;base<=0;clock125<=0;capture_start<=0;
   written_bytes<=0;read_bytes<=0;control_written_bytes<=0;output_pending<=0;output_is_write<=0;
   fifo_peak<=0;capture_cycles<=0;last_capture_cycles<=0;last_transfer_cycles<=0;verify_retries<=0;end
  else begin
   clock125<=clock125+1;fault_sync<={fault_sync[0],engine_fault};
   if(level>fifo_peak)fifo_peak<=level;
   if(output_valid&&output_ready&&!output_pending)begin output_pending<=1;output_is_write<=output_write;end
   if(output_reply)begin
    if(!output_pending)local_fault<=1;
    output_pending<=0;
    if(output_is_write)control_written_bytes<=control_written_bytes+64;else read_bytes<=read_bytes+64;
   end
   if(put)begin
    if(!collecting)begin base<=f_addr;collecting<=1;received<=1;capture_start<=clock125;if(f_addr[5:0]||f_last)local_fault<=1;end
    else begin
     if(f_last!=(received==74)||received>=75||f_addr<base||f_addr-base>=4800||f_addr[5:0])local_fault<=1;
     received<=received+1;
     if(received==74)begin frame_full<=1;last_capture_cycles<=clock125-capture_start+1;capture_cycles<=capture_cycles+clock125-capture_start+1;end
    end
   end
   if(command_fire)begin collecting<=0;frame_full<=0;received<=0;end
   if(done_rv)begin written_bytes<=written_bytes+4800;last_transfer_cycles<=done_rdata[223:160];verify_retries<=verify_retries+{32'd0,done_rdata[255:224]};end
  end
 end
 localparam IDLE=0,LOAD=1,WPREF=2,WSEND=3,WWAIT=4,RPREF=5,RSEND=6,RWAIT=7,DONE=8;
 reg [3:0] state;reg [6:0] loaded,index,accepted;reg [31:0] window_base;reg [159:0] window_token;
 (* ram_style="block" *)reg [511:0] expected[0:127];
 reg [511:0] expected_q;reg [74:0] seen;reg [31:0] stall,verify_ticks,retries;
 reg [63:0] clock200,started;
 wire [31:0] load_address=data_rd[543:512],offset=load_address-window_base;
 // Separate BRAM output/address arithmetic from identity and ownership checks.
 // This keeps the entire FIFO/DDR engine at its actual 200MHz clock.
 reg load_pending,load_range_ok,load_last;reg [6:0] load_index;
 reg [511:0] load_data;
 wire load_fire=data_rv&&data_rr;
 wire valid_load=load_range_ok&&load_index<75&&!seen[load_index]&&load_last==(loaded==74);
 assign cmd_rr=u200&&!engine_fault&&state==IDLE;
 assign data_rr=u200&&!engine_fault&&state==LOAD&&accepted<75;
 wire payload_command=state==WSEND||state==RSEND;
 reg bus_pending,bus_output,arbitration_held,held_output;
 wire choose_output=arbitration_held?held_output:oc_valid;
 assign b_valid=u200&&!engine_fault&&!bus_pending&&(oc_valid||payload_command);
 assign b_write=choose_output?oc_data[544]:state==WSEND;
 assign b_fast_write=!choose_output&&state==WSEND;
 assign b_addr=choose_output?oc_data[543:512]:(window_base+{19'd0,index,6'd0});
 assign b_data=choose_output?oc_data[511:0]:expected_q;
 assign oc_ready=b_valid&&b_ready&&choose_output;
 wire payload_fire=b_valid&&b_ready&&!choose_output;
 wire payload_reply=b_reply&&bus_pending&&!bus_output;
 assign output_response_valid=b_reply&&bus_pending&&bus_output;
 assign done_wv=u200&&!engine_fault&&state==DONE;
 assign done_wdata={retries,clock200-started,window_token};
 always @(posedge clk200)begin
  // An invalid beat may touch only this private scratch RAM. Its fault is
  // latched before the engine can issue any DDR write or publish completion.
  // Do not put the long validation cone on hundreds of BRAM write-enable pins.
  if(state==LOAD&&load_pending)expected[load_index]<=load_data;
  if(state==WPREF||state==RPREF)expected_q<=expected[index];
 end
 always @(posedge clk200 or negedge u200)begin
  if(!u200)begin state<=IDLE;engine_fault<=0;loaded<=0;accepted<=0;index<=0;window_base<=0;window_token<=0;seen<=0;
   load_pending<=0;load_range_ok<=0;load_last<=0;load_index<=0;load_data<=0;
   stall<=0;verify_ticks<=0;retries<=0;clock200<=0;started<=0;bus_pending<=0;bus_output<=0;arbitration_held<=0;held_output<=0;end
  else if(!engine_fault)begin
   clock200<=clock200+1;
   load_pending<=load_fire;
   if(load_fire)begin
    load_index<=offset[12:6];load_data<=data_rd[511:0];load_last<=data_rd[544];
    load_range_ok<=load_address>=window_base&&offset<4800&&load_address[5:0]==0;
    accepted<=accepted+1'b1;
   end
   if(b_valid&&!b_ready)begin arbitration_held<=1;held_output<=choose_output;end
   if(b_valid&&b_ready)begin bus_pending<=1;bus_output<=choose_output;arbitration_held<=0;end
   if(b_reply)begin
    if(!bus_pending)engine_fault<=1;
    else begin bus_pending<=0;if(bus_output&&!or_ready)engine_fault<=1;end
   end
   if(state!=IDLE&&state!=DONE)begin
    if(load_fire||load_pending||payload_fire||payload_reply)stall<=0;
    else if(stall==TIMEOUT-1)engine_fault<=1;else stall<=stall+1;
   end
   if(state==RPREF||state==RSEND||state==RWAIT)begin
    if(verify_ticks==TIMEOUT-1)engine_fault<=1;else verify_ticks<=verify_ticks+1;
   end
   case(state)
    IDLE:if(cmd_rv)begin
     window_token<=cmd_rd[191:32];window_base<=cmd_rd[31:0];loaded<=0;accepted<=0;seen<=0;index<=0;stall<=0;verify_ticks<=0;retries<=0;started<=clock200;state<=LOAD;
     if(cmd_rd[63:32]>=32||cmd_rd[191:128]==0||cmd_rd[127:64]==0||cmd_rd[5:0]!=0)engine_fault<=1;
    end
    LOAD:if(load_pending)begin
     if(!valid_load)engine_fault<=1;
     else begin seen[load_index]<=1;loaded<=loaded+1;if(loaded==74)begin index<=0;state<=WPREF;end end
    end
    WPREF:state<=WSEND;
    WSEND:if(payload_fire)state<=WWAIT;
    WWAIT:if(payload_reply)begin if(index==74)begin index<=0;state<=RPREF;end else begin index<=index+1;state<=WPREF;end end
    RPREF:state<=RSEND;
    RSEND:if(payload_fire)state<=RWAIT;
    RWAIT:if(payload_reply)begin
     if(b_rdata===expected_q)begin
      if(index==74)state<=DONE;else begin index<=index+1;state<=RPREF;end
     end else begin retries<=retries+1;state<=RPREF;end
    end
    DONE:if(done_wr)state<=IDLE;
    default:engine_fault<=1;
   endcase
  end
 end
endmodule
