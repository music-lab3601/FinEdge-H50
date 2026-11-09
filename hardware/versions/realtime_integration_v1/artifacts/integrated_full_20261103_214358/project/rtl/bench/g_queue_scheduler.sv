`timescale 1ns/1ps
// One stream, configurable FIFO of separately owned DDR inputs. Window FIFO
// capture and verified DDR commit have independent lifetimes; DDR lifetime ends at ordered retirement.
module g_queue_scheduler #(parameter SLOTS=32,SW=$clog2(SLOTS))(
 input wire clk,reset_n,arm_valid,input wire [63:0] arm_session,
 input wire [5:0] cfg_slots,input wire [63:0] age_limit_ticks,
 input wire [15:0] h50_records,input wire [63:0] now_tick,latest_rows,latest_segment,
 input wire s_valid,output wire s_ready,input wire [1279:0] s_desc,input wire [63:0] s_source_tick,
 output reg write_valid,input wire write_ready,output reg [SW-1:0] write_slot,
 output wire [63:0] write_session,output reg [63:0] write_generation,output reg [1279:0] write_desc,
 input wire commit_valid,output wire commit_ready,input wire [SW-1:0] commit_slot,
 input wire [63:0] commit_session,commit_generation,input wire commit_ok,
 output reg npu_valid,input wire npu_ready,output wire [SW-1:0] npu_slot,
 output wire [63:0] npu_session,npu_generation,output wire [1279:0] npu_desc,
 input wire done_valid,output wire done_ready,input wire [SW-1:0] done_slot,
 input wire [63:0] done_session,done_generation,input wire [127:0] done_result,
 output reg event_valid,input wire event_ready,output wire [1279:0] event_desc,
 output reg event_has_slot,output wire [SW-1:0] event_slot,
 output wire [63:0] event_session,event_generation,
 output reg [7:0] event_reason,output reg [127:0] event_result,
 output reg [63:0] event_source_tick,event_launch_tick,
 output reg active,fault,output reg [63:0] protocol_errors,
 output reg [5:0] occupied,queue_peak,
 output reg [63:0] admitted,served,completed,cancelled,expired,
 output reg [63:0] expired_h50,expired_age,completed_h50_late,completed_age_late,late_started,
 output wire quiescent);
 import e_control_pkg::*;
 localparam [255:0] QHASH=256'h0eb1627ef8f83a72a63d1d846120458baa55a3be1e9a1a9a6902218f37e037f4;
 localparam [255:0] MHASH=256'ha9a4822bcbce441a6565d938be3f052c33a36464c20ff8cfeb3a5125299fcc44;
 wire up;e_reset_sync reset(clk,reset_n,up);
 reg [63:0] session,generation;
 reg [2:0] owner[0:SLOTS-1];reg failed[0:SLOTS-1];
 (* ram_style="block" *) reg [63:0] metadata[0:SLOTS*8-1];
 reg [63:0] read_data;reg [511:0] head_meta;
 reg meta_writing,head_loading,read_pending;reg [2:0] meta_word,pending_word;reg [3:0] fetch_word;reg [63:0] write_source_tick;
 reg [SW-1:0] head,tail,writer_slot;
 reg head_loaded,npu_busy;reg [SW:0] writing_count;wire writer_active=writing_count!=0;
 reg [63:0] writer_generation,head_launch_tick;
 reg [1279:0] cancelled_desc;
 wire fetch_start=active&&occupied!=0&&!head_loaded&&!head_loading&&!(meta_writing&&write_slot==head);
 wire fetch_issue=head_loading&&fetch_word<8;
 wire [511:0] stored_meta={write_generation,write_source_tick,write_desc[1183:1120],write_desc[415:352],write_desc[351:288],write_desc[287:224],write_desc[223:160],write_desc[159:96]};
 wire push=s_valid&&s_ready,pop=event_valid&&event_ready&&event_has_slot;
 wire [63:0] head_gen=head_meta[511:448],head_source=head_meta[447:384];
 e_desc_t incoming,front;
 always @*begin
  incoming=s_desc;front=0;
  front.session=session;front.segment=head_meta[63:0];front.window_id=head_meta[127:64];
  front.first_ordinal=head_meta[191:128];front.last_ordinal=head_meta[255:192];
  front.event_ts_ns=head_meta[319:256];front.cookie={session,head_meta[383:320]};
  front.max_wait_ticks=age_limit_ticks;front.valid_bytes=4000;front.storage_bytes=4800;
  front.dtype=1;front.layout=1;front.quant_hash=QHASH;front.model_hash=MHASH;
 end
 wire in_h50=h50_records!=0&&latest_rows!=0&&latest_rows-1>=incoming.window_id&&latest_rows-1-incoming.window_id>=h50_records;
 wire in_age=age_limit_ticks!=0&&now_tick>=s_source_tick&&now_tick-s_source_tick>age_limit_ticks;
 wire malformed=incoming.session!=session||incoming.stream!=0||incoming.segment==0||
  incoming.first_ordinal==0||incoming.last_ordinal<incoming.first_ordinal||incoming.flags!=0||incoming.reserved!=0;
 wire in_invalid=incoming.segment!=latest_segment;
 wire cancel_input=malformed||in_invalid||in_h50||in_age;
 wire head_h50=h50_records!=0&&latest_rows!=0&&latest_rows-1>=front.window_id&&latest_rows-1-front.window_id>=h50_records;
 wire head_age=age_limit_ticks!=0&&now_tick>=head_source&&now_tick-head_source>age_limit_ticks;
 assign s_ready=up&&active&&!fault&&!write_valid&&!meta_writing&&!event_valid&&!done_valid&&!commit_valid&&
                   (cancel_input||(occupied<cfg_slots));
 assign commit_ready=up&&active&&!event_valid&&!done_valid;
 assign done_ready=up&&active&&!event_valid;
 assign write_session=session;assign npu_session=session;assign npu_generation=head_gen;
 assign npu_slot=head;assign npu_desc=front;
 assign event_desc=event_has_slot?front:cancelled_desc;assign event_slot=head;
 assign event_session=session;assign event_generation=event_has_slot?head_gen:0;
 assign quiescent=occupied==0&&!write_valid&&!writer_active&&!npu_valid&&!npu_busy&&!event_valid&&!meta_writing&&!head_loading&&!read_pending;
 always @(posedge clk)begin
  if(meta_writing)metadata[{write_slot,meta_word}]<=stored_meta[meta_word*64+:64];
  if(fetch_issue)read_data<=metadata[{head,fetch_word[2:0]}];
 end
 integer i;
 always @(posedge clk or negedge up)begin
  if(!up)begin
   active<=0;fault<=0;protocol_errors<=0;session<=0;generation<=0;
   head<=0;tail<=0;occupied<=0;queue_peak<=0;head_loaded<=0;head_meta<=0;meta_writing<=0;meta_word<=0;write_source_tick<=0;head_loading<=0;read_pending<=0;fetch_word<=0;pending_word<=0;
   write_valid<=0;write_slot<=0;write_generation<=0;write_desc<=0;
   writing_count<=0;writer_slot<=0;writer_generation<=1;npu_valid<=0;npu_busy<=0;head_launch_tick<=0;
   event_valid<=0;event_has_slot<=0;event_reason<=0;event_result<=0;event_source_tick<=0;event_launch_tick<=0;cancelled_desc<=0;
   admitted<=0;served<=0;completed<=0;cancelled<=0;expired<=0;expired_h50<=0;expired_age<=0;
   completed_h50_late<=0;completed_age_late<=0;late_started<=0;
   for(i=0;i<SLOTS;i=i+1)begin owner[i]<=FREE;failed[i]<=0;end
  end else if(!fault)begin
   if(arm_valid&&!active)begin
    if(arm_session==0||cfg_slots==0||cfg_slots>SLOTS)begin fault<=1;protocol_errors<=protocol_errors+1;end
    else begin active<=1;session<=arm_session;end
   end
   if(meta_writing)begin if(meta_word==7)meta_writing<=0;else meta_word<=meta_word+1'b1;end
   if(fetch_start)begin head_loading<=1;fetch_word<=0;end
   read_pending<=fetch_issue;
   if(fetch_issue)begin pending_word<=fetch_word[2:0];fetch_word<=fetch_word+1'b1;end
   if(read_pending)begin
    head_meta[pending_word*64+:64]<=read_data;
    if(pending_word==7)begin head_loaded<=1;head_loading<=0;end
   end
   if(push)begin
    admitted<=admitted+1;
    if(cancel_input)begin
     cancelled_desc<=incoming;event_valid<=1;event_has_slot<=0;event_source_tick<=s_source_tick;event_launch_tick<=0;
     event_reason<=malformed?BAD_DESCRIPTOR:in_invalid?INVALID_STREAM:EXPIRED;
     event_result<=0;event_result[104]<=in_h50;event_result[105]<=in_age;
    end else if(generation==64'hffffffffffffffff||owner[tail]!=FREE)begin fault<=1;protocol_errors<=protocol_errors+1;end
    else begin
     meta_writing<=1;meta_word<=0;write_source_tick<=s_source_tick;
     generation<=generation+1;owner[tail]<=WRITING;failed[tail]<=0;
     write_valid<=1;write_slot<=tail;write_generation<=generation+1;write_desc<=incoming;
     tail<=tail==cfg_slots-1?0:tail+1'b1;
     if(occupied+1'b1>queue_peak)queue_peak<=occupied+1'b1;
    end
   end
   case({push&&!cancel_input&&generation!=64'hffffffffffffffff,pop})
    2'b10:occupied<=occupied+1'b1;2'b01:occupied<=occupied-1'b1;default:;
   endcase
   case({write_valid&&write_ready,commit_valid&&commit_ready})
    2'b10:writing_count<=writing_count+1'b1;2'b01:writing_count<=writing_count-1'b1;default:;
   endcase
   if(write_valid&&write_ready)begin
    if(writing_count==SLOTS)begin fault<=1;protocol_errors<=protocol_errors+1;end
    write_valid<=0;
   end
   if(commit_valid&&commit_ready)begin
    if(!writer_active||commit_slot!=writer_slot||commit_session!=session||
       commit_generation!=writer_generation||owner[commit_slot]!=WRITING)begin fault<=1;protocol_errors<=protocol_errors+1;end
    else begin owner[commit_slot]<=READY;failed[commit_slot]<=!commit_ok;writer_slot<=writer_slot==cfg_slots-1?0:writer_slot+1'b1;writer_generation<=writer_generation+1;end
   end
   if(npu_valid&&npu_ready)begin
    npu_valid<=0;npu_busy<=1;owner[head]<=READING;served<=served+1;head_launch_tick<=now_tick;
    if(head_h50||head_age)late_started<=late_started+1;
   end
   if(event_valid&&event_ready)begin
    event_valid<=0;
    if(event_reason==OK)begin
     completed<=completed+1;
     if(event_result[104])completed_h50_late<=completed_h50_late+1;
     if(event_result[105])completed_age_late<=completed_age_late+1;
    end else begin
     cancelled<=cancelled+1;
     if(event_reason==EXPIRED)begin
      expired<=expired+1;
      if(event_result[104])expired_h50<=expired_h50+1;
      if(event_result[105])expired_age<=expired_age+1;
     end
    end
    if(event_has_slot)begin
     owner[head]<=FREE;head<=head==cfg_slots-1?0:head+1'b1;
     head_loaded<=0;head_loading<=0;read_pending<=0;fetch_word<=0;npu_busy<=0;
    end
   end
   if(done_valid&&done_ready)begin
    if(!head_loaded||done_session!=session||done_slot!=head||done_generation!=head_gen||
       owner[head]!=READING||!npu_busy)begin fault<=1;protocol_errors<=protocol_errors+1;end
    else begin
     event_valid<=1;event_has_slot<=1;event_reason<=OK;
     event_source_tick<=head_source;event_launch_tick<=head_launch_tick;
     event_result<=done_result;event_result[104]<=head_h50;event_result[105]<=head_age;
    end
   end else if(active&&head_loaded&&!event_valid&&!push&&!npu_busy&&!npu_valid&&occupied!=0&&owner[head]==READY)begin
    if(failed[head]||front.segment!=latest_segment||head_h50||head_age)begin
     event_valid<=1;event_has_slot<=1;event_source_tick<=head_source;event_launch_tick<=0;
     event_reason<=failed[head]?WRITE_FAILED:front.segment!=latest_segment?INVALID_STREAM:EXPIRED;
     event_result<=0;event_result[104]<=head_h50;event_result[105]<=head_age;
    end else npu_valid<=1;
   end
  end
 end
endmodule
