// B: in-order raw commit; completion includes accepted complete top10. No normalization/NPU.
// Raw DBN message byte offsets match provider's 56-byte MBO schema.
module fb_book_core #(parameter ORDER_BITS=17,LEVEL_BITS=14,LEVEL_BANK_BITS=5,STATUS_MAX=8)(
 input wire clk,rst,enable,input wire [31:0] order_base,
 input wire [63:0] day_ns,start_ns,end_ns,
 input wire [7:0] status_count,input wire [STATUS_MAX*64-1:0] status_ts,
 input wire [STATUS_MAX-1:0] status_allowed,input wire [15:0] publisher_id,input wire [31:0] instrument_id,
 input wire s_valid,output wire s_ready,input wire [447:0] s_data,input wire [63:0] s_ordinal,
 output reg m_valid,input wire m_ready,output reg [2559:0] m_book,output reg [255:0] m_identity,
 output reg mem_valid,input wire mem_ready,output reg mem_write,output reg [31:0] mem_addr,
 output reg [511:0] mem_wdata,input wire mem_rsp_valid,input wire [511:0] mem_rdata,input wire mem_error,
 output wire fault,output wire [31:0] error,
 output reg [63:0] raw_count,mutation_count,eligible_count,output wire initialized,
 output reg done_valid,output reg [63:0] done_ordinal,done_ts,
 output reg done_mutated,done_eligible,output wire [4:0] phase,
 output wire [31:0] lookup_max_probe,output wire [63:0] lookup_probe_steps
);
 localparam IDLE=0,LOOKUP=1,FOUND=2,DELTA1=3,DELTA1_DONE=4,DELTA2=5,DELTA2_DONE=6,
 COMMIT=7,COMMIT_DONE=8,SNAPSHOT=9,SNAPSHOT_DONE=10,CLEAR=11,CLEAR_DONE=12,PRICE_DIV=13,FINISH=14,ROW_PENDING=15;
 reg [3:0] state;reg local_fault,previous_eligible,book_started;reg [31:0] local_error;
 reg [63:0] previous_ts,ts,ordinal,key,segment,row;reg [7:0] action;
 reg side,delete_order;reg [31:0] price,size,new_size,old_p,old_q;reg old_s;
 reg [447:0] message;
 reg event_mutated,event_eligible,event_status_ok;
 integer status_index,next_status_index,k_status;
 // The table is the frozen STATUS-derived gate, configured before enable.
 // Every raw event advances it, including ignored T/F/N and R messages.
 always @*begin
  next_status_index=-1;
  for(integer g=0;g<STATUS_MAX;g=g+1)
   if(g<status_count && status_ts[g*64+:64]<=s_data[64+:64])next_status_index=g;
 end
 assign phase={1'b0,state};
 reg [63:0] dividend,quotient;reg [17:0] remainder;reg [5:0] div_step;
 reg [17:0] next_rem;reg [63:0] next_quotient;
 wire order_lookup_ready,order_found_valid,order_found,order_busy,order_commit_ready,order_fault;
 wire [31:0] order_price,order_size,order_error;wire order_side;
 wire omv,omw;wire [31:0] oma;wire [511:0] omd;
 always @*begin mem_valid=omv;mem_write=omw;mem_addr=oma;mem_wdata=omd;end
 wire level_ready,level_done,level_snap_ready,level_snap_valid,level_complete,level_fault,level_init;
 wire [2559:0] level_snapshot;wire [31:0] level_error,ask_levels,bid_levels;
 wire clear_now=state==CLEAR;
 wire level_update=state==DELTA1||state==DELTA2;
 wire signed [64:0] negative_size=-$signed({33'b0,(action==8'h43 ? size:old_q)});
 wire signed [64:0] level_delta=(action==8'h41||state==DELTA2)?$signed({33'b0,size}):negative_size;
 wire [31:0] level_price=(action==8'h41||state==DELTA2)?price:old_p;
 wire level_side=(action==8'h41||state==DELTA2)?side:old_s;
 assign initialized=level_init;
 assign fault=local_fault||order_fault||level_fault;
 assign error=local_fault?local_error:order_fault?(32'h10000|order_error):level_fault?(32'h20000|level_error):0;
 assign s_ready=enable&&initialized&&!fault&&state==IDLE&&(!m_valid||m_ready);
 fb_order_store #(.BITS(ORDER_BITS)) orders(
  .clk(clk),.rst(rst),.clear(clear_now),.base(order_base),.lookup_key(key),.lookup_valid(state==LOOKUP),.lookup_ready(order_lookup_ready),
  .found_valid(order_found_valid),.found_ready(state==FOUND),.found(order_found),.old_price(order_price),.old_size(order_size),.old_side(order_side),
  .commit_valid(state==COMMIT),.commit_ready(order_commit_ready),.commit_write(1'b1),.commit_delete(delete_order),
  .commit_side(side),.commit_price(action==8'h43 ? old_p:price),.commit_size(new_size),
  .mem_valid(omv),.mem_ready(mem_ready),.mem_write(omw),.mem_addr(oma),.mem_wdata(omd),
  .mem_rsp_valid(mem_rsp_valid),.mem_rdata(mem_rdata),.mem_error(mem_error),.busy(order_busy),.fault(order_fault),.error(order_error),.max_probe(lookup_max_probe),.probe_steps(lookup_probe_steps));
 fm_level_book #(.BITS(LEVEL_BITS),.BANK_BITS(LEVEL_BANK_BITS)) levels(
  .clk(clk),.rst(rst),.clear(clear_now),.update_valid(level_update),.update_ready(level_ready),
  .update_price(level_price),.update_side(level_side),.update_delta(level_delta),.done_valid(level_done),.done_ready(state==DELTA1_DONE||state==DELTA2_DONE),
  .snapshot_request(state==SNAPSHOT),.snapshot_ready(level_snap_ready),.snapshot_valid(level_snap_valid),.snapshot_accept(state==SNAPSHOT_DONE),
  .snapshot_complete(level_complete),.snapshot(level_snapshot),.fault(level_fault),.error(level_error),.initialized(level_init),.ask_levels(ask_levels),.bid_levels(bid_levels));
 task fail(input [31:0] why);begin local_fault<=1;local_error<=why;m_valid<=0;end endtask
 reg [63:0] raw_price;reg valid_book;
 always @(posedge clk)begin
  if(rst)begin state<=IDLE;local_fault<=0;local_error<=0;raw_count<=0;mutation_count<=0;eligible_count<=0;
   m_valid<=0;previous_ts<=0;previous_eligible<=0;book_started<=0;segment<=0;row<=0;
   status_index<=-1;event_mutated<=0;event_eligible<=0;event_status_ok<=0;
   done_valid<=0;done_ordinal<=0;done_ts<=0;done_mutated<=0;done_eligible<=0;end
  else if(!fault)begin
   done_valid<=0;
   if(m_valid&&m_ready)m_valid<=0;
   case(state)
    IDLE:if(s_valid&&s_ready)begin
     raw_count<=raw_count+1;message<=s_data;ts<=s_data[64+:64];ordinal<=s_ordinal;
     event_mutated<=0;event_eligible<=0;
     event_status_ok<=next_status_index>=0 && status_allowed[next_status_index] && s_data[64+:64]!=status_ts[next_status_index*64+:64];
     if(next_status_index!=status_index)previous_eligible<=0;
     status_index<=next_status_index;
     if(s_ordinal!=raw_count+1||s_data[0+:8]!=14||s_data[16+:16]!=publisher_id||s_data[32+:32]!=instrument_id||s_data[64+:64]<previous_ts)fail(1);
     else begin
      previous_ts<=s_data[64+:64];action<=s_data[304+:8];side<=s_data[312+:8]==8'h42;
      key<=s_data[128+:64];size<=s_data[256+:32];raw_price=s_data[192+:64];
      if(s_data[304+:8]==8'h52)state<=CLEAR;
      else if(s_data[304+:8]==8'h54||s_data[304+:8]==8'h46||s_data[304+:8]==8'h4e)state<=FINISH;
      else if(!book_started||s_data[128+:64]==0||s_data[256+:32]==0||
       !(s_data[312+:8]==8'h41||s_data[312+:8]==8'h42)||(s_data[288+:8]&8'h54)!=0)fail(2);
      else if(!(s_data[304+:8]==8'h41||s_data[304+:8]==8'h43||s_data[304+:8]==8'h4d))fail(3);
      else if(s_data[304+:8]!=8'h43&&(raw_price==0||raw_price>=64'h7fffffffffffffff))fail(4);
      else if(s_data[304+:8]==8'h43)begin event_mutated<=1;price<=0;state<=LOOKUP;end
      else begin event_mutated<=1;dividend<=raw_price;quotient<=0;remainder<=0;div_step<=0;state<=PRICE_DIV;end
     end
    end
    CLEAR:begin previous_eligible<=0;book_started<=1;state<=CLEAR_DONE;end
    CLEAR_DONE:if(level_init)state<=FINISH;
    PRICE_DIV:begin
     next_rem={remainder[16:0],dividend[63]};
     next_quotient={quotient[62:0],next_rem>=100000};
     remainder<=next_rem>=100000?next_rem-100000:next_rem;
     dividend<=dividend<<1;quotient<=next_quotient;
     if(div_step==63)begin
      if((next_rem>=100000?next_rem-100000:next_rem)!=0||next_quotient[63:32]!=0||next_quotient==0)fail(4);
      else begin price<=next_quotient[31:0];state<=LOOKUP;end
     end else div_step<=div_step+1;
    end
    LOOKUP:if(order_lookup_ready)state<=FOUND;
    FOUND:if(order_found_valid)begin
     old_p<=order_price;old_q<=order_size;old_s<=order_side;
     if(action==8'h41&&order_found)fail(5);
     else if(action!=8'h41&&(!order_found||order_side!=side))fail(6);
     else if(action==8'h43&&size>order_size)fail(7);
     else begin new_size<=action==8'h43 ? order_size-size:size;delete_order<=action==8'h43&&order_size==size;state<=DELTA1;end
    end
    DELTA1:if(level_ready)state<=DELTA1_DONE;
    DELTA1_DONE:if(level_done)state<=action==8'h4d ? DELTA2:COMMIT;
    DELTA2:if(level_ready)state<=DELTA2_DONE;
    DELTA2_DONE:if(level_done)state<=COMMIT;
    COMMIT:if(order_commit_ready)state<=COMMIT_DONE;
    COMMIT_DONE:if(!order_busy)begin row<=mutation_count;mutation_count<=mutation_count+1;state<=SNAPSHOT;end
    SNAPSHOT:if(level_snap_ready)state<=SNAPSHOT_DONE;
    SNAPSHOT_DONE:if(level_snap_valid)begin
     valid_book=level_complete&&ts>=start_ns&&ts<end_ns&&event_status_ok&&level_snapshot[128+:64]>0&&level_snapshot[128+:64]<level_snapshot[0+:64];
     if(valid_book)begin
      if(!previous_eligible)segment<=segment+1;
      previous_eligible<=1;event_eligible<=1;m_book<=level_snapshot;
      m_identity<={ts-day_ns-64'd18000000000000,ordinal,row,previous_eligible?segment:segment+1};m_valid<=1;
     end else previous_eligible<=0;
     state<=valid_book?ROW_PENDING:FINISH;
    end
    ROW_PENDING:if(m_valid&&m_ready)begin eligible_count<=eligible_count+1;state<=FINISH;end
    FINISH:begin
     done_valid<=1;done_ordinal<=ordinal;done_ts<=ts;
     done_mutated<=event_mutated;done_eligible<=event_eligible;state<=IDLE;
    end
    default:fail(8);
   endcase
  end
 end
endmodule


