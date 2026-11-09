// GP region400d2000: bounded control/status only. No tensor payload over GP.
// A request remains immutable until the ARM reports completion of that exact
// resident invocation. The response mailbox is an application result queue.
module g_predict_control #(parameter SLOTS=32)(input wire por_n,c_clk,b_clk,
 input wire [31:0] awaddr,awdata,input wire awvalid,output wire awready,
 input wire [31:0] araddr,input wire arvalid,output wire arready,
 output reg [31:0] rdata,output reg rvalid,input wire rready,
 output reg configured,output reg arm_pulse,
 output wire [SLOTS*64-1:0] slot_bases,slot_caps,output reg [63:0] output_base,
 output reg [5:0] active_slots,output reg [63:0] age_limit_ticks,output reg [15:0] h50_records,
 input wire npu_valid,output wire npu_ready,input wire [255:0] npu_request,
 output wire done_valid,input wire done_ready,output wire [159:0] done_token,
 input wire result_valid,output wire result_ready,input wire [383:0] result_data,
 input wire [63:0] status_data,output wire [7:0] status_index,input wire datapath_fault,output reg fault);
 wire cu,bu;b1_reset_sync cr(c_clk,por_n,cu),br(b_clk,por_n,bu);
 localparam CFW=14+64+64+6+16;
 reg [13:0] base_hi_c,base_hi_b;reg [63:0] out_c,age_c;reg [5:0] slots_c;reg [15:0] h50_c;
 generate for(genvar sl=0;sl<SLOTS;sl=sl+1)begin:allocations
  localparam [4:0] SLOT_INDEX=sl;
  assign slot_bases[sl*64+:64]={32'd0,base_hi_b,SLOT_INDEX,13'd0};
  assign slot_caps[sl*64+:64]=64'd8192;
 end endgenerate
 reg cfg_pending,cfg_sent;wire cfg_ready,cfg_valid;wire [CFW-1:0] cfg_data;
 b1_mailbox #(CFW) cfg(por_n,c_clk,cfg_pending,cfg_ready,{h50_c,slots_c,age_c,out_c,base_hi_c},b_clk,cfg_valid,1'b1,cfg_data);
 wire request_c;wire [255:0] request_data;reg request_ack;
 b1_mailbox #(256) launch(por_n,b_clk,npu_valid,npu_ready,npu_request,c_clk,request_c,request_ack,request_data);
 reg done_pending;wire done_c_ready;reg [159:0] done_c_token;
 b1_mailbox #(160) done(por_n,c_clk,done_pending,done_c_ready,done_c_token,b_clk,done_valid,done_ready,done_token);
 wire result_c;wire [383:0] result_c_data;reg result_ack;
 b1_mailbox #(384) results(por_n,b_clk,result_valid,result_ready,result_data,c_clk,result_c,result_ack,result_c_data);
 reg [7:0] index_c;wire query_ready,query_b;wire [7:0] query_index;reg query_pending,query_wait;
 b1_mailbox #(8) query(por_n,c_clk,query_pending,query_ready,index_c,b_clk,query_b,1'b1,query_index);
 assign status_index=query_index;
 reg reply_pending;reg [63:0] reply_hold;wire reply_ready,reply_c;wire [63:0] reply_data;
 b1_mailbox #(64) reply(por_n,b_clk,reply_pending,reply_ready,reply_hold,c_clk,reply_c,1'b1,reply_data);
 reg [63:0] status_snapshot;reg [31:0] query_sequence;
 (* ASYNC_REG="TRUE" *) reg [1:0] datapath_fault_sync;
 wire write_fire=awvalid&&awready;wire read_fire=arvalid&&arready;
 wire [11:0] wo=awaddr[11:0],ro=araddr[11:0];
 assign awready=cu;assign arready=cu&&!rvalid;
 always @(posedge b_clk or negedge bu)begin
  if(!bu)begin configured<=0;arm_pulse<=0;base_hi_b<=0;output_base<=0;active_slots<=32;age_limit_ticks<=250000;h50_records<=50;reply_pending<=0;reply_hold<=0;end
  else begin
   arm_pulse<=0;
   if(cfg_valid)begin {h50_records,active_slots,age_limit_ticks,output_base,base_hi_b}<=cfg_data;configured<=1;arm_pulse<=1;end
   if(reply_pending&&reply_ready)reply_pending<=0;
   if(query_b)begin reply_hold<=status_data;reply_pending<=1;end
  end
 end
 always @(posedge c_clk or negedge cu)begin
  if(!cu)begin base_hi_c<=0;out_c<=0;age_c<=250000;slots_c<=32;h50_c<=50;cfg_pending<=0;cfg_sent<=0;
   done_pending<=0;done_c_token<=0;request_ack<=0;result_ack<=0;fault<=0;
   index_c<=0;query_pending<=0;query_wait<=0;status_snapshot<=0;query_sequence<=0;rvalid<=0;rdata<=0;datapath_fault_sync<=0;end
  else begin
   request_ack<=0;result_ack<=0;datapath_fault_sync<={datapath_fault_sync[0],datapath_fault};
   if(cfg_pending&&cfg_ready)cfg_pending<=0;
   if(done_pending&&done_c_ready)done_pending<=0;
   if(query_pending&&query_ready)query_pending<=0;
   if(reply_c)begin status_snapshot<=reply_data;query_sequence<=query_sequence+1;query_wait<=0;end
   if(rvalid&&rready)rvalid<=0;
   if(read_fire)begin
    rvalid<=1;rdata<=0;
    case(ro)
     12'h000:rdata<=32'h34514746; //FGQ4
     12'h004:rdata<=4;
     12'h008:rdata<={26'd0,query_wait,(fault||datapath_fault_sync[1]),done_pending,cfg_sent,result_c,request_c};
     12'h030:rdata<=query_sequence;
     12'h034:rdata<=status_snapshot[31:0];12'h038:rdata<=status_snapshot[63:32];
     default:begin
      if(ro>=12'h100&&ro<12'h120)rdata<=request_data[(ro[4:2])*32+:32];
      if(ro>=12'h180&&ro<12'h1b0)rdata<=result_c_data[((ro-12'h180)>>2)*32+:32];
     end
    endcase
   end
   if(write_fire)case(wo)
    12'h020:if(!cfg_sent)out_c[31:0]<=awdata;else fault<=1;
    12'h024:if(!cfg_sent&&cfg_ready&&awdata==32'h41524d34)begin cfg_pending<=1;cfg_sent<=1;end else fault<=1;
    12'h028:if(request_c&&!done_pending&&done_c_ready&&awdata==32'h444f4e45)begin
     done_c_token<=request_data[159:0];done_pending<=1;request_ack<=1;end else fault<=1;
    12'h02c:if(result_c&&awdata==32'h54414b45)result_ack<=1;else fault<=1;
    12'h030:if(!query_wait&&query_ready&&awdata<256)begin index_c<=awdata[7:0];query_pending<=1;query_wait<=1;end else fault<=1;
    12'h040:if(!cfg_sent)age_c[31:0]<=awdata;else fault<=1;
    12'h044:if(!cfg_sent)age_c[63:32]<=awdata;else fault<=1;
    12'h048:if(!cfg_sent&&awdata>0&&awdata<=SLOTS)slots_c<=awdata[5:0];else fault<=1;
    12'h04c:if(!cfg_sent&&awdata<=65535)h50_c<=awdata[15:0];else fault<=1;
    12'h800:if(!cfg_sent&&awdata[17:0]==0)base_hi_c<=awdata[31:18];else fault<=1;
    default:fault<=1;
   endcase
  end
 end
endmodule
