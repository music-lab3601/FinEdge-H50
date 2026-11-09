`timescale 1ns/1ps
// E owns metadata and slot lifecycle; F owns tensor bytes/address/visibility.
// Configurable1/2 slots per stream, ordered retirement, ONE NPU request total.
module e_scheduler #(
 parameter integer STREAMS=2,Q_DEPTH=8,SLOTS=4,PER_STREAM_SLOTS=1,
 parameter [STREAMS-1:0] ACTIVE_STREAM_MASK={STREAMS{1'b1}},
 parameter integer SW=(SLOTS>1?$clog2(SLOTS):1),QW=$clog2(Q_DEPTH),NW=$clog2(STREAMS)
)(
 input wire clk,reset_n,
 input wire arm_valid,output wire arm_ready,input wire [63:0] arm_session,input wire io_quiescent,
 input wire [STREAMS-1:0] invalidate,recover,ingress_idle,
 input wire [STREAMS*64-1:0] stream_segments,
 input wire [STREAMS-1:0] s_valid,output wire [STREAMS-1:0] s_ready,
 input wire [STREAMS*1280-1:0] s_desc,
 output reg write_valid,input wire write_ready,output reg [SW-1:0] write_slot,
 output wire [63:0] write_session,write_generation,output wire [1279:0] write_desc,
 input wire commit_valid,output wire commit_ready,input wire [SW-1:0] commit_slot,
 input wire [63:0] commit_session,commit_generation,input wire commit_ok,
 output reg npu_valid,input wire npu_ready,output reg [SW-1:0] npu_slot,
 output wire [63:0] npu_session,npu_generation,output wire [1279:0] npu_desc,
 input wire done_valid,output wire done_ready,input wire [SW-1:0] done_slot,
 input wire [63:0] done_session,done_generation,input wire [127:0] done_result,
 output reg event_valid,input wire event_ready,output reg [1279:0] event_desc,
 output reg event_has_slot,output reg [SW-1:0] event_slot,
 output reg [63:0] event_session,event_generation,event_age,
 output reg [7:0] event_reason,output reg [127:0] event_result,
 output reg active,fault,output reg [63:0] cycle,protocol_errors,
 output wire [STREAMS-1:0] stream_bad,stream_busy,
 output wire [STREAMS*32-1:0] stream_inflight,
 output wire [SLOTS*3-1:0] slot_states,
 output wire [STREAMS*32-1:0] queue_levels,queue_high_water,
 output wire [STREAMS*64-1:0] admitted,served,completed,cancelled,expired,
 output wire [STREAMS*64-1:0] blocked_cycles,wait_total,wait_max,late_started
);
 import e_control_pkg::*;
 wire up;e_reset_sync u_reset(clk,reset_n,up);
 reg [63:0] session,generation;
 reg [STREAMS-1:0] bad;
 reg [1:0] inflight[0:STREAMS-1];
 reg order_rd[0:STREAMS-1],order_wr[0:STREAMS-1];
 reg [SW-1:0] order_slots[0:STREAMS-1][0:1];
 reg [63:0] bad_segment[0:STREAMS-1];
 wire [1279:0] qheads[0:STREAMS-1];
 wire [63:0] q_head_time[0:STREAMS-1];
 reg [QW-1:0] rd[0:STREAMS-1],wr[0:STREAMS-1];
 reg [QW:0] count[0:STREAMS-1];
 reg [31:0] high[0:STREAMS-1];
 reg [63:0] num_admit[0:STREAMS-1],num_serve[0:STREAMS-1],num_complete[0:STREAMS-1];
 reg [63:0] num_cancel[0:STREAMS-1],num_expire[0:STREAMS-1],num_block[0:STREAMS-1];
 reg [63:0] num_wait[0:STREAMS-1],max_wait[0:STREAMS-1];
 reg [63:0] num_late[0:STREAMS-1];
 reg [2:0] owner[0:SLOTS-1];
 reg [63:0] gen[0:SLOTS-1],when_admitted[0:SLOTS-1];
 reg [1279:0] descs[0:SLOTS-1];
 reg [NW-1:0] flow[0:SLOTS-1];
 reg issued[0:SLOTS-1];
 reg [7:0] slot_failure[0:SLOTS-1];
 reg [NW-1:0] rr,nrr;
 reg npu_busy;
 reg [SW-1:0] running_slot;
 reg [NW-1:0] event_flow;
 wire [STREAMS-1:0] push;
 reg [STREAMS-1:0] pop;
 integer sel,free_slot,ready_slot,k,j,ix;
 reg [7:0] queue_reason,ready_reason;
 e_desc_t head,ready_head;
 reg [1:0] pick_phase;
 reg pick_is_ready,pick_has_free,pick_malformed;
 reg [NW-1:0] pick_flow;
 reg [SW-1:0] pick_slot;
 reg [63:0] pick_tick,pick_age;
 e_desc_t pick_desc;
 // Phase1 subtracts age, phase2 compares it. pick_age trails live age by1 tick.
 wire [63:0] decision_age=pick_age+64'd1;
 wire control_idle=active&&!fault&&!event_valid&&!done_valid&&!commit_valid;
 wire select_ready=control_idle&&pick_phase==0&&!npu_busy&&!npu_valid&&ready_slot>=0;
 wire select_queue=control_idle&&pick_phase==0&&!select_ready&&!write_valid&&sel>=0;
 wire choose_ready=control_idle&&pick_phase==2&&pick_is_ready;
 wire choose_queue=control_idle&&pick_phase==2&&!pick_is_ready;
 // Queued cancellations must not pass an older same-stream slot.
 wire queue_take=choose_queue&&((queue_reason!=OK&&inflight[pick_flow]==0) || (queue_reason==OK&&pick_has_free));
 assign arm_ready=up&&!active&&io_quiescent;
 assign done_ready=up&&active&&!event_valid;
 assign commit_ready=up&&active&&!event_valid&&!done_valid;
 assign write_session=session;assign write_generation=gen[write_slot];assign write_desc=descs[write_slot];
 assign npu_session=session;assign npu_generation=gen[npu_slot];assign npu_desc=descs[npu_slot];
 assign stream_bad=bad;
 genvar g;
 generate for(g=0;g<STREAMS;g=g+1)begin:metrics
  (* ram_style="distributed" *) reg [1343:0] queue_storage[0:Q_DEPTH-1];
  always @(posedge clk)if(push[g])queue_storage[wr[g]]<={cycle,s_desc[g*1280+:1280]};
  assign {q_head_time[g],qheads[g]}=queue_storage[rd[g]];
  assign s_ready[g]=up&&active&&!fault&&ACTIVE_STREAM_MASK[g]&&count[g]<Q_DEPTH;
  assign stream_busy[g]=inflight[g]!=0;
  assign stream_inflight[g*32+:32]=inflight[g];
  assign push[g]=s_ready[g]&&s_valid[g];
  assign queue_levels[g*32+:32]=count[g];assign queue_high_water[g*32+:32]=high[g];
  assign admitted[g*64+:64]=num_admit[g];assign served[g*64+:64]=num_serve[g];
  assign completed[g*64+:64]=num_complete[g];assign cancelled[g*64+:64]=num_cancel[g];
  assign expired[g*64+:64]=num_expire[g];assign blocked_cycles[g*64+:64]=num_block[g];
  assign wait_total[g*64+:64]=num_wait[g];assign wait_max[g*64+:64]=max_wait[g];
  assign late_started[g*64+:64]=num_late[g];
 end
 for(g=0;g<SLOTS;g=g+1)begin:owners assign slot_states[g*3+:3]=owner[g];end endgenerate
 // Round robin advances only on an actual queue decision / locked NPU offer.
 always @*begin
  sel=-1;free_slot=-1;ready_slot=-1;
  for(k=STREAMS-1;k>=0;k=k-1)begin
   ix=(rr+k)&(STREAMS-1);
   if(count[ix]!=0&&inflight[ix]<PER_STREAM_SLOTS&&ACTIVE_STREAM_MASK[ix])sel=ix;
  end
  for(k=SLOTS-1;k>=0;k=k-1)if(owner[k]==FREE)free_slot=k;
  for(k=STREAMS-1;k>=0;k=k-1)begin
   ix=(nrr+k)&(STREAMS-1);
   if(inflight[ix]!=0&&owner[order_slots[ix][order_rd[ix]]]==READY)ready_slot=order_slots[ix][order_rd[ix]];
  end
  head='0;if(sel>=0)head=qheads[sel];
  ready_head='0;if(ready_slot>=0)ready_head=descs[ready_slot];
  queue_reason=OK;ready_reason=OK;
  if(pick_phase==2)begin
   if(bad[pick_flow]||invalidate[pick_flow])queue_reason=INVALID_STREAM;
   else if(pick_malformed)queue_reason=BAD_DESCRIPTOR;
   else if(pick_desc.max_wait_ticks!=0&&pick_age>=pick_desc.max_wait_ticks)queue_reason=EXPIRED;
   if(slot_failure[pick_slot]!=OK)ready_reason=slot_failure[pick_slot];
   else if(bad[pick_flow]||invalidate[pick_flow])ready_reason=INVALID_STREAM;
   else if(pick_desc.max_wait_ticks!=0&&pick_age>=pick_desc.max_wait_ticks)ready_reason=EXPIRED;
  end
  pop=0;if(queue_take&&(queue_reason!=OK||generation!=64'hffffffffffffffff))pop[pick_flow]=1;
 end
 integer i;
 reg [63:0] measured_wait;
 always @(posedge clk or negedge up)begin
  if(!up)begin
   active<=0;fault<=0;session<=0;generation<=0;cycle<=0;protocol_errors<=0;bad<=0;
   rr<=0;nrr<=0;write_valid<=0;npu_valid<=0;npu_busy<=0;running_slot<=0;write_slot<=0;npu_slot<=0;
   event_valid<=0;event_desc<=0;event_has_slot<=0;event_slot<=0;event_session<=0;event_generation<=0;
   event_reason<=0;event_age<=0;event_result<=0;event_flow<=0;
   pick_phase<=0;pick_is_ready<=0;pick_has_free<=0;pick_malformed<=0;
   pick_flow<=0;pick_slot<=0;pick_tick<=0;pick_age<=0;pick_desc<='0;
   for(i=0;i<STREAMS;i=i+1)begin
    rd[i]<=0;wr[i]<=0;count[i]<=0;high[i]<=0;bad_segment[i]<=0;
    inflight[i]<=0;order_rd[i]<=0;order_wr[i]<=0;order_slots[i][0]<=0;order_slots[i][1]<=0;
    num_admit[i]<=0;num_serve[i]<=0;num_complete[i]<=0;num_cancel[i]<=0;num_expire[i]<=0;
    num_block[i]<=0;num_wait[i]<=0;max_wait[i]<=0;num_late[i]<=0;
   end
   for(i=0;i<SLOTS;i=i+1)begin owner[i]<=FREE;gen[i]<=0;descs[i]<=0;when_admitted[i]<=0;flow[i]<=0;issued[i]<=0;slot_failure[i]<=OK;end
  end else begin
   cycle<=cycle+1;
   if(select_ready)begin
    pick_phase<=1;pick_is_ready<=1;pick_slot<=ready_slot;pick_flow<=flow[ready_slot];
    pick_desc<=descs[ready_slot];pick_tick<=when_admitted[ready_slot];pick_has_free<=0;
   end else if(select_queue)begin
    pick_phase<=1;pick_is_ready<=0;pick_flow<=sel;pick_desc<=head;pick_tick<=q_head_time[sel];
    pick_has_free<=free_slot>=0;pick_slot<=free_slot>=0?free_slot:0;
   end else if(pick_phase==1)begin
    pick_phase<=2;pick_age<=cycle-pick_tick;
    pick_malformed<=pick_desc.session!=session||pick_desc.stream!=pick_flow||pick_desc.segment==0||
      pick_desc.segment!=stream_segments[pick_flow*64+:64]||pick_desc.first_ordinal==0||
      pick_desc.last_ordinal<pick_desc.first_ordinal||pick_desc.valid_bytes==0||
      pick_desc.storage_bytes<pick_desc.valid_bytes||pick_desc.reserved!=0||pick_desc.flags!=0||pick_desc.max_wait_ticks[63];
   end else if(pick_phase==2)begin
    pick_age<=pick_age+1;
    if(choose_ready||choose_queue)pick_phase<=0;
   end
   // Skip a temporarily non-retirable head so other streams can prefetch.
   if(choose_queue&&!queue_take)rr<=pick_flow+1'b1;
   if(arm_valid&&arm_ready)begin
    if(arm_session!=0)begin session<=arm_session;active<=1;end
    else protocol_errors<=protocol_errors+1;
   end
   for(i=0;i<STREAMS;i=i+1)begin
    if(invalidate[i])begin
     if(!bad[i])bad_segment[i]<=stream_segments[i*64+:64];
     bad[i]<=1;
    end else if(recover[i]&&bad[i])begin
     if(count[i]==0&&inflight[i]==0&&ingress_idle[i]&&stream_segments[i*64+:64]!=0&&
        stream_segments[i*64+:64]!=bad_segment[i])bad[i]<=0;
    end
    if(push[i])begin
     wr[i]<=(wr[i]==Q_DEPTH-1)?0:wr[i]+1'b1;num_admit[i]<=num_admit[i]+1;
    end
    if(pop[i])rd[i]<=(rd[i]==Q_DEPTH-1)?0:rd[i]+1'b1;
    case({push[i],pop[i]})
     2'b10:begin count[i]<=count[i]+1;if(count[i]+1>high[i])high[i]<=count[i]+1;end
     2'b01:count[i]<=count[i]-1;
     default:;
    endcase
    if(active&&s_valid[i]&&!s_ready[i])num_block[i]<=num_block[i]+1;
   end
   if(write_valid&&write_ready)begin issued[write_slot]<=1;write_valid<=0;end
   if(npu_valid&&npu_ready)begin
    owner[npu_slot]<=READING;running_slot<=npu_slot;npu_busy<=1;npu_valid<=0;
    num_serve[flow[npu_slot]]<=num_serve[flow[npu_slot]]+1;
    measured_wait=cycle-when_admitted[npu_slot];
    num_wait[flow[npu_slot]]<=num_wait[flow[npu_slot]]+measured_wait;
    if(measured_wait>max_wait[flow[npu_slot]])max_wait[flow[npu_slot]]<=measured_wait;
    if(descs[npu_slot][416+:64]!=0&&measured_wait>descs[npu_slot][416+:64])num_late[flow[npu_slot]]<=num_late[flow[npu_slot]]+1;
   end
   if(event_valid&&event_ready)begin
    event_valid<=0;
    if(event_reason==OK)num_complete[event_flow]<=num_complete[event_flow]+1;
    else begin
     num_cancel[event_flow]<=num_cancel[event_flow]+1;
     if(event_reason==EXPIRED)num_expire[event_flow]<=num_expire[event_flow]+1;
    end
    if(event_has_slot)begin
     owner[event_slot]<=FREE;issued[event_slot]<=0;slot_failure[event_slot]<=OK;
     inflight[flow[event_slot]]<=inflight[flow[event_slot]]-1'b1;
     order_rd[flow[event_slot]]<=(PER_STREAM_SLOTS==1)?1'b0:~order_rd[flow[event_slot]];
     if(npu_busy&&running_slot==event_slot)npu_busy<=0;
    end
   end
   if(done_valid&&done_ready)begin
    if(done_slot>=SLOTS)protocol_errors<=protocol_errors+1;
    else if(done_session!=session||done_generation!=gen[done_slot]||owner[done_slot]!=READING||
            !npu_busy||running_slot!=done_slot||order_slots[flow[done_slot]][order_rd[flow[done_slot]]]!=done_slot)protocol_errors<=protocol_errors+1;
    else begin
     event_valid<=1;event_desc<=descs[done_slot];event_has_slot<=1;event_slot<=done_slot;
     event_session<=session;event_generation<=gen[done_slot];event_age<=cycle-when_admitted[done_slot];
     event_reason<=(bad[flow[done_slot]]||invalidate[flow[done_slot]])?INVALID_STREAM:OK;event_result<=done_result;event_flow<=flow[done_slot];
    end
   end else if(commit_valid&&commit_ready)begin
    if(commit_slot>=SLOTS)protocol_errors<=protocol_errors+1;
    else if(commit_session!=session||commit_generation!=gen[commit_slot]||owner[commit_slot]!=WRITING||!issued[commit_slot])protocol_errors<=protocol_errors+1;
    else begin
     // Even failed writes retire through the per-stream head, never immediately.
     owner[commit_slot]<=READY;slot_failure[commit_slot]<=commit_ok?OK:WRITE_FAILED;
    end
   end else if(choose_ready)begin
    nrr<=pick_flow+1'b1;
    if(ready_reason!=OK)begin
     event_valid<=1;event_desc<=pick_desc;event_has_slot<=1;event_slot<=pick_slot;
     event_session<=session;event_generation<=gen[pick_slot];event_age<=decision_age;
     event_reason<=ready_reason;event_result<=0;event_flow<=pick_flow;
    end else begin npu_valid<=1;npu_slot<=pick_slot;end
   end else if(queue_take)begin
    rr<=pick_flow+1'b1;
    if(queue_reason!=OK)begin
     event_valid<=1;event_desc<=pick_desc;event_has_slot<=0;event_slot<=0;event_session<=session;event_generation<=0;
     event_age<=decision_age;event_reason<=queue_reason;event_result<=0;event_flow<=pick_flow;
    end else if(generation==64'hffffffffffffffff)begin
     fault<=1; // Deliberately no token wrap. Coordinator cancels whole epoch.
    end else begin
     generation<=generation+1;gen[pick_slot]<=generation+1;owner[pick_slot]<=WRITING;
     descs[pick_slot]<=pick_desc;when_admitted[pick_slot]<=pick_tick;flow[pick_slot]<=pick_flow;issued[pick_slot]<=0;
     inflight[pick_flow]<=inflight[pick_flow]+1'b1;
     order_slots[pick_flow][order_wr[pick_flow]]<=pick_slot;
     order_wr[pick_flow]<=(PER_STREAM_SLOTS==1)?1'b0:~order_wr[pick_flow];
     slot_failure[pick_slot]<=OK;write_valid<=1;write_slot<=pick_slot;
    end
   end
  end
 end
 initial begin
  if(STREAMS!=2&&STREAMS!=4)$fatal(1,"E STREAMS must be2 or4");
  if(Q_DEPTH<2 || (Q_DEPTH&(Q_DEPTH-1))!=0)$fatal(1,"E Q_DEPTH must be power-of-two >=2");
  if(SLOTS<1)$fatal(1,"E SLOTS must be positive");
  if(PER_STREAM_SLOTS<1||PER_STREAM_SLOTS>2||PER_STREAM_SLOTS>SLOTS)$fatal(1,"E PER_STREAM_SLOTS must be1/2 and <=SLOTS");
 end
endmodule
