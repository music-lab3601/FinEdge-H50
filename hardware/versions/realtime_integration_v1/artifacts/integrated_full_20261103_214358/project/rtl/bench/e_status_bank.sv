`timescale 1ns/1ps
// Optional coherent read-only register bank. ALL inputs/requests use clk.
// A PS bus in another domain must use a transaction bridge, not bitwise sync.
module e_status_bank #(parameter integer STREAMS=2,CONTROL_HZ=200000000)(
 input wire clk,reset_n,snapshot,input wire active,fault,
 input wire [63:0] cycle,protocol_errors,input wire [STREAMS-1:0] stream_bad,stream_busy,
 input wire [STREAMS*32-1:0] queue_levels,queue_high_water,stream_inflight,
 input wire [STREAMS*64-1:0] admitted,served,completed,cancelled,expired,blocked_cycles,wait_total,wait_max,late_started,
 input wire read_valid,output wire read_ready,input wire [11:0] read_address,
 output reg response_valid,input wire response_ready,output reg [31:0] response_data,output reg response_error,
 output reg snapshot_valid,output reg [31:0] snapshot_sequence
);
 wire up;e_reset_sync u_reset(clk,reset_n,up);
 reg [63:0] tick_hold,errors_hold;
 reg [31:0] status_hold,bad_hold,busy_hold;
 reg [31:0] levels[0:STREAMS-1],high[0:STREAMS-1];
 reg [31:0] inflight_hold[0:STREAMS-1];
 reg [63:0] counters[0:STREAMS-1][0:8];
 reg [31:0] selected;reg bad_address;integer stream,word,index,i,j;
 assign read_ready=up&&(!response_valid||response_ready);
 always @*begin
  selected=0;bad_address=0;stream=0;word=0;index=0;
  if(read_address[1:0]!=0)bad_address=1;
  else if(read_address<12'h100)begin
   case(read_address)
    12'h000:selected=32'h31534345;
    12'h004:selected=2|(STREAMS<<8);
    12'h008:selected=snapshot_sequence;
    12'h00c:selected=status_hold;
    12'h010:selected=tick_hold[31:0];12'h014:selected=tick_hold[63:32];
    12'h018:selected=errors_hold[31:0];12'h01c:selected=errors_hold[63:32];
    12'h020:selected=bad_hold;12'h024:selected=busy_hold;
    12'h028:selected=CONTROL_HZ;12'h02c:selected={31'd0,snapshot_valid};
    default:bad_address=1;
   endcase
  end else begin
   stream=(read_address-12'h100)>>7;word=(read_address-12'h100)&127;
   if(stream>=STREAMS)bad_address=1;
   else if(word==0)selected=levels[stream];
   else if(word==4)selected=high[stream];
   else if(word==80)selected=inflight_hold[stream];
   else if(word>=8&&word<80)begin
    index=(word-8)>>3;
    selected=word[2]?counters[stream][index][63:32]:counters[stream][index][31:0];
   end else bad_address=1;
  end
 end
 always @(posedge clk or negedge up)begin
  if(!up)begin
   response_valid<=0;response_data<=0;response_error<=0;snapshot_valid<=0;snapshot_sequence<=0;
   tick_hold<=0;errors_hold<=0;status_hold<=0;bad_hold<=0;busy_hold<=0;
   for(i=0;i<STREAMS;i=i+1)begin levels[i]<=0;high[i]<=0;inflight_hold[i]<=0;for(j=0;j<9;j=j+1)counters[i][j]<=0;end
  end else begin
   if(response_valid&&response_ready)response_valid<=0;
   if(read_valid&&read_ready)begin response_valid<=1;response_data<=selected;response_error<=bad_address;end
   if(snapshot)begin
    snapshot_valid<=1;snapshot_sequence<=snapshot_sequence+1;tick_hold<=cycle;errors_hold<=protocol_errors;
    status_hold<={30'd0,fault,active};bad_hold<=stream_bad;busy_hold<=stream_busy;
    for(i=0;i<STREAMS;i=i+1)begin
     levels[i]<=queue_levels[i*32+:32];high[i]<=queue_high_water[i*32+:32];
     inflight_hold[i]<=stream_inflight[i*32+:32];
     counters[i][0]<=admitted[i*64+:64];counters[i][1]<=served[i*64+:64];
     counters[i][2]<=completed[i*64+:64];counters[i][3]<=cancelled[i*64+:64];
     counters[i][4]<=expired[i*64+:64];counters[i][5]<=blocked_cycles[i*64+:64];
     counters[i][6]<=wait_total[i*64+:64];counters[i][7]<=wait_max[i*64+:64];counters[i][8]<=late_started[i*64+:64];
    end
   end
  end
 end
endmodule
