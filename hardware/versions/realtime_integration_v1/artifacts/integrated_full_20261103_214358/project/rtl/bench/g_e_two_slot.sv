// E_control_r1 specialization for the actual one-AAPL/two-context F producer.
// F already retains at most two immutable contexts: a second descriptor FIFO
// cannot increase capacity. Allocate the DDR slot directly at offer handshake.
// Two outstanding slots, ordered retirement, one NPU, exact epoch/generation.
module g_e_two_slot #(
 parameter STREAMS=2,Q_DEPTH=8,SLOTS=2,PER_STREAM_SLOTS=2,
 parameter [STREAMS-1:0] ACTIVE_STREAM_MASK=1,
 parameter SW=1,QW=$clog2(Q_DEPTH),NW=1
)(input wire clk,reset_n,arm_valid,output wire arm_ready,input wire [63:0] arm_session,input wire io_quiescent,
 input wire [STREAMS-1:0] invalidate,recover,ingress_idle,input wire [STREAMS*64-1:0] stream_segments,
 input wire [STREAMS-1:0] s_valid,output wire [STREAMS-1:0] s_ready,input wire [STREAMS*1280-1:0] s_desc,
 output reg write_valid,input wire write_ready,output reg [SW-1:0] write_slot,
 output wire [63:0] write_session,write_generation,output wire [1279:0] write_desc,
 input wire commit_valid,output wire commit_ready,input wire [SW-1:0] commit_slot,
 input wire [63:0] commit_session,commit_generation,input wire commit_ok,
 output reg npu_valid,input wire npu_ready,output reg [SW-1:0] npu_slot,
 output wire [63:0] npu_session,npu_generation,output wire [1279:0] npu_desc,
 input wire done_valid,output wire done_ready,input wire [SW-1:0] done_slot,
 input wire [63:0] done_session,done_generation,input wire [127:0] done_result,
 output reg event_valid,input wire event_ready,output wire [1279:0] event_desc,
 output reg event_has_slot,output reg [SW-1:0] event_slot,
 output wire [63:0] event_session,event_generation,output reg [63:0] event_age,
 output reg [7:0] event_reason,output reg [127:0] event_result,
 output reg active,fault,output reg [63:0] cycle,protocol_errors,
 output wire [STREAMS-1:0] stream_bad,stream_busy,output wire [STREAMS*32-1:0] stream_inflight,
 output wire [SLOTS*3-1:0] slot_states,output wire [STREAMS*32-1:0] queue_levels,queue_high_water,
 output wire [STREAMS*64-1:0] admitted,served,completed,cancelled,expired,blocked_cycles,wait_total,wait_max,late_started);
 import e_control_pkg::*;
 wire up;e_reset_sync reset(clk,reset_n,up);
 reg [63:0] session,generation;reg [1:0] count;
 reg [2:0] owner[0:1];reg [1279:0] desc[0:1];reg [63:0] gen[0:1],accepted_at[0:1];
 reg issued[0:1],failed[0:1];reg order[0:1];reg head,tail;
 reg npu_busy,invalid;reg [63:0] invalid_segment;reg [1279:0] cancelled_desc;
 reg [63:0] n_admit,n_serve,n_complete,n_cancel,n_expire,n_block;
 wire [SW-1:0] first=order[head];wire free_any=owner[0]==FREE||owner[1]==FREE;
 wire [SW-1:0] free_slot=owner[0]==FREE?0:1;
 e_desc_t incoming,front;always @*begin incoming=s_desc[1279:0];front=desc[first];end
 wire malformed=incoming.session!=session||incoming.stream!=0||incoming.segment==0||incoming.first_ordinal==0||
  incoming.last_ordinal<incoming.first_ordinal||incoming.valid_bytes==0||incoming.storage_bytes<incoming.valid_bytes||
  incoming.flags!=0||incoming.reserved!=0||incoming.max_wait_ticks[63];
 wire input_cancel=malformed||invalid||invalidate[0]||incoming.segment!=stream_segments[63:0];
 wire input_expired=incoming.max_wait_ticks!=0&&incoming.max_wait_ticks==1;
 assign arm_ready=up&&!active&&io_quiescent;
 assign s_ready={1'b0,up&&active&&!fault&&!write_valid&&!event_valid&&!done_valid&&!commit_valid&&
                   (input_cancel?count==0:(count<2&&free_any))};
 wire push=s_valid[0]&&s_ready[0];wire pop=event_valid&&event_ready&&event_has_slot;
 assign done_ready=up&&active&&!event_valid;
 assign commit_ready=up&&active&&!event_valid&&!done_valid;
 assign write_session=session;assign write_generation=gen[write_slot];assign write_desc=desc[write_slot];
 assign npu_session=session;assign npu_generation=gen[npu_slot];assign npu_desc=desc[npu_slot];
 assign event_desc=event_has_slot?desc[event_slot]:cancelled_desc;
 assign event_session=session;assign event_generation=event_has_slot?gen[event_slot]:0;
 assign stream_bad={1'b0,invalid};assign stream_busy={1'b0,count!=0};assign stream_inflight={32'd0,30'd0,count};
 assign slot_states={owner[1],owner[0]};assign queue_levels=0;assign queue_high_water=0;
 assign admitted={64'd0,n_admit};assign served={64'd0,n_serve};assign completed={64'd0,n_complete};
 assign cancelled={64'd0,n_cancel};assign expired={64'd0,n_expire};assign blocked_cycles={64'd0,n_block};
 assign wait_total=0;assign wait_max=0;assign late_started=0;
 wire [63:0] age=cycle-accepted_at[first];
 wire deadline=front.max_wait_ticks!=0&&age>front.max_wait_ticks;
 integer i;
 always @(posedge clk or negedge up)begin
  if(!up)begin
   active<=0;fault<=0;cycle<=0;protocol_errors<=0;session<=0;generation<=0;count<=0;head<=0;tail<=0;
   write_valid<=0;write_slot<=0;npu_valid<=0;npu_slot<=0;npu_busy<=0;invalid<=0;invalid_segment<=0;
   event_valid<=0;event_has_slot<=0;event_slot<=0;event_age<=0;event_reason<=0;event_result<=0;cancelled_desc<=0;
   n_admit<=0;n_serve<=0;n_complete<=0;n_cancel<=0;n_expire<=0;n_block<=0;
   for(i=0;i<2;i=i+1)begin owner[i]<=FREE;desc[i]<=0;gen[i]<=0;accepted_at[i]<=0;issued[i]<=0;failed[i]<=0;order[i]<=0;end
  end else begin
   cycle<=cycle+1;
   if(arm_valid&&arm_ready)begin if(arm_session==0)protocol_errors<=protocol_errors+1;else begin active<=1;session<=arm_session;end end
   if(invalidate[0])begin invalid<=1;if(!invalid)invalid_segment<=stream_segments[63:0];end
   else if(recover[0]&&count==0&&ingress_idle[0]&&stream_segments[63:0]!=0&&stream_segments[63:0]!=invalid_segment)invalid<=0;
   if(s_valid[0]&&!s_ready[0]&&active)n_block<=n_block+1;
   if(push)begin
    n_admit<=n_admit+1;
    if(input_cancel)begin
     cancelled_desc<=incoming;event_has_slot<=0;event_slot<=0;event_valid<=1;event_age<=0;event_result<=0;
     event_reason<=malformed?BAD_DESCRIPTOR:INVALID_STREAM;
    end else if(generation==64'hffffffffffffffff)begin fault<=1;end
    else begin
     generation<=generation+1;gen[free_slot]<=generation+1;desc[free_slot]<=incoming;accepted_at[free_slot]<=cycle;
     owner[free_slot]<=WRITING;issued[free_slot]<=0;failed[free_slot]<=0;order[tail]<=free_slot;tail<=~tail;
     write_valid<=1;write_slot<=free_slot;
    end
   end
   case({push&&!input_cancel&&generation!=64'hffffffffffffffff,pop})
    2'b10:count<=count+1;2'b01:count<=count-1;default:;
   endcase
   if(write_valid&&write_ready)begin issued[write_slot]<=1;write_valid<=0;end
   if(npu_valid&&npu_ready)begin owner[npu_slot]<=READING;npu_valid<=0;npu_busy<=1;n_serve<=n_serve+1;end
   if(event_valid&&event_ready)begin
    event_valid<=0;
    if(event_reason==OK)n_complete<=n_complete+1;
    else begin n_cancel<=n_cancel+1;if(event_reason==EXPIRED)n_expire<=n_expire+1;end
    if(event_has_slot)begin owner[event_slot]<=FREE;issued[event_slot]<=0;head<=~head;npu_busy<=0;end
   end
   if(done_valid&&done_ready)begin
    if(done_session!=session||done_generation!=gen[done_slot]||owner[done_slot]!=READING||!npu_busy||done_slot!=first)protocol_errors<=protocol_errors+1;
    else begin
     event_valid<=1;event_has_slot<=1;event_slot<=done_slot;event_age<=cycle-accepted_at[done_slot];event_result<=done_result;
     event_reason<=(invalid||invalidate[0]||front.segment!=stream_segments[63:0])?INVALID_STREAM:OK;
    end
   end else if(commit_valid&&commit_ready)begin
    if(commit_session!=session||commit_generation!=gen[commit_slot]||owner[commit_slot]!=WRITING||!issued[commit_slot])protocol_errors<=protocol_errors+1;
    else begin owner[commit_slot]<=READY;failed[commit_slot]<=!commit_ok;end
   end else if(active&&!fault&&!event_valid&&!npu_busy&&!npu_valid&&count!=0&&owner[first]==READY)begin
    if(failed[first]||invalid||invalidate[0]||front.segment!=stream_segments[63:0]||deadline)begin
     event_valid<=1;event_has_slot<=1;event_slot<=first;event_result<=0;event_age<=age;
     event_reason<=failed[first]?WRITE_FAILED:(invalid||invalidate[0]||front.segment!=stream_segments[63:0])?INVALID_STREAM:EXPIRED;
    end else begin npu_valid<=1;npu_slot<=first;end
   end
  end
 end
 initial if(STREAMS!=2||SLOTS!=2||PER_STREAM_SLOTS!=2||ACTIVE_STREAM_MASK!=1)$fatal(1,"G compact E supports single stream/two slots only");
endmodule
