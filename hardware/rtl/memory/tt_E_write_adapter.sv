`timescale 1ns/1ps
// E_control_r1 write/commit adapter. No FREE list, NPU launch or release.
// G supplies two approved allocation extents. Unknown drainage is quarantined;
// only matching external proof permits local writer recovery and failed commit.
module tt_E_write_adapter #(
 parameter [255:0] QUANT_HASH=256'h0eb1627ef8f83a72a63d1d846120458baa55a3be1e9a1a9a6902218f37e037f4,
 parameter [255:0] MIXED_HASH=256'ha9a4822bcbce441a6565d938be3f052c33a36464c20ff8cfeb3a5125299fcc44,
 parameter [255:0] FALLBACK_HASH=256'hde6af192e8781e5509fd33769475f2abb7277f802fce9dcb77e9879760fb3293,
 parameter integer WATCHDOG=1048576,SLOTS=32
)(
 input wire clk,rst,
 input wire [SLOTS-1:0] allocation_approved,input wire [SLOTS*64-1:0] allocation_bases,allocation_capacities,
 input wire write_valid,output wire write_ready,input wire [63:0] write_session,write_generation,
 input wire [31:0] write_slot,input wire [1279:0] write_desc,
 output wire [127:0] lookup_cookie,input wire lookup_found,lookup_bindable,lookup_context,
 input wire [1279:0] lookup_desc,input wire [383:0] lookup_first,lookup_end,
 output reg bind_valid,bind_context,output reg [159:0] bind_token,
 output reg committed_valid,committed_context,output reg [159:0] committed_token,
 output wire pack_start_valid,input wire pack_start_ready,output wire pack_context,
 input wire payload_valid,output wire payload_ready,input wire [511:0] payload_data,input wire payload_last,input wire [6:0] payload_index,
 output wire mem_valid,input wire mem_ready,output wire [31:0] mem_addr,output wire [511:0] mem_data,
 output wire [63:0] mem_keep,output wire mem_last,
 output wire fence_valid,input wire fence_ready,output wire [159:0] fence_token,
 input wire fence_done,input wire [159:0] fence_done_token,input wire fence_error,external_fault,
 output wire commit_valid,input wire commit_ready,output wire commit_ok,output wire [159:0] commit_token,
 output wire quarantine_valid,output wire [159:0] quarantine_token,
 input wire drain_proven,input wire [159:0] drain_token,
 output wire local_datapath_reset,output reg [63:0] requests,successful,failed_drained,isolations,protocol_errors
);
 localparam IDLE=0,CHECK=1,BIND=2,START=3,STREAM=4,FENCE=5,FAULT=6,VALIDATE=7;
 reg [2:0] state;reg [1279:0] request_desc;reg [159:0] token;
 reg context_id;reg [63:0] base,capacity;reg [6:0] beats;
 reg address_ok_q,format_ok_q,identity_ok_q;
 wire [31:0] slot=token[31:0];wire [64:0] extent_end={1'b0,base}+{1'b0,capacity};
 wire valid_address=slot<SLOTS&&allocation_approved[slot[$clog2(SLOTS)-1:0]]&&base[63:32]==0&&base[5:0]==0&&capacity>=4800&&extent_end<=65'h100000000;
 wire valid_format=request_desc==lookup_desc&&request_desc[95:64]==0&&request_desc[511:480]==4000&&request_desc[543:512]==4800&&request_desc[559:544]==1&&request_desc[575:560]==1&&request_desc[1279:1248]==0&&request_desc[863:608]==QUANT_HASH&&(request_desc[1119:864]==MIXED_HASH||request_desc[1119:864]==FALLBACK_HASH);
 wire valid_identity=lookup_found&&lookup_bindable&&token[159:96]==lookup_desc[63:0]&&token[95:32]!=0&&lookup_first[127:0]==lookup_end[127:0];
 assign write_ready=!rst&&state==IDLE&&!external_fault;
 assign lookup_cookie=request_desc[1247:1120];
 assign pack_context=context_id;assign pack_start_valid=!rst&&state==START;
 assign mem_valid=!rst&&state==STREAM&&payload_valid&&!external_fault;
 assign payload_ready=!rst&&state==STREAM&&mem_ready&&!external_fault;
 assign mem_addr=base[31:0]+{19'd0,payload_index,6'd0};assign mem_data=payload_data;assign mem_keep=64'hffffffffffffffff;assign mem_last=payload_last;
 assign fence_valid=!rst&&state==FENCE&&!external_fault;assign fence_token=token;
 assign local_datapath_reset=rst;
 assign quarantine_valid=!rst&&state==FAULT;assign quarantine_token=token;
 // Completed whole-window tokens can wait independently of the capture front.
 (* ram_style="distributed" *)reg [159:0] completions[0:SLOTS-1];
 reg [$clog2(SLOTS)-1:0] ch,ct;reg [$clog2(SLOTS):0] count;
 assign commit_valid=!rst&&count!=0&&state!=FAULT&&!external_fault;
 assign commit_token=completions[ch];assign commit_ok=1'b1;
 wire pop=commit_valid&&commit_ready;
 always @(posedge clk)begin
  if(fence_done)completions[ct]<=fence_done_token;
 end
 always @(posedge clk)begin
  if(rst)begin
   state<=IDLE;request_desc<=0;token<=0;context_id<=0;base<=0;capacity<=0;beats<=0;
   address_ok_q<=0;format_ok_q<=0;identity_ok_q<=0;
   bind_valid<=0;bind_context<=0;bind_token<=0;committed_valid<=0;committed_context<=0;committed_token<=0;
   requests<=0;successful<=0;failed_drained<=0;isolations<=0;protocol_errors<=0;ch<=0;ct<=0;count<=0;
  end else begin
   bind_valid<=0;committed_valid<=0;
   case({fence_done,pop})
    2'b10:count<=count+1'b1;2'b01:count<=count-1'b1;default:;
   endcase
   if(fence_done)begin
    if(count==SLOTS&&!pop)begin state<=FAULT;protocol_errors<=protocol_errors+1;end
    ct<=ct+1'b1;
   end
   if(pop)begin ch<=ch+1'b1;successful<=successful+1;end
   if(external_fault||fence_error||(fence_done&&count==SLOTS&&!pop))begin state<=FAULT;isolations<=isolations+1;end
   else case(state)
    IDLE:if(write_valid&&write_ready)begin
     token<={write_session,write_generation,write_slot};request_desc<=write_desc;
     base<=write_slot<SLOTS?allocation_bases[write_slot*64+:64]:0;
     capacity<=write_slot<SLOTS?allocation_capacities[write_slot*64+:64]:0;requests<=requests+1;state<=CHECK;
    end
    CHECK:begin
     context_id<=lookup_context;
     address_ok_q<=valid_address;format_ok_q<=valid_format;identity_ok_q<=valid_identity;state<=VALIDATE;
    end
    VALIDATE:begin
     if(address_ok_q&&format_ok_q&&identity_ok_q)begin bind_valid<=1;bind_context<=context_id;bind_token<=token;state<=BIND;end
     else begin state<=FAULT;protocol_errors<=protocol_errors+1;end
    end
    BIND:state<=START;
    START:if(pack_start_ready)begin beats<=0;state<=STREAM;end
    STREAM:if(mem_valid&&mem_ready)begin
     if(payload_index>=75||payload_last!=(beats==74))begin state<=FAULT;protocol_errors<=protocol_errors+1;end
     else if(beats==74)begin
      // The FIFO now owns the entire immutable input; release only metadata.
      committed_valid<=1;committed_context<=context_id;committed_token<=token;state<=FENCE;
     end else beats<=beats+1;
    end
    FENCE:if(fence_ready)state<=IDLE;
    FAULT:;
    default:state<=FAULT;
   endcase
   if(fence_error)begin state<=FAULT;protocol_errors<=protocol_errors+1;end
  end
 end
endmodule
