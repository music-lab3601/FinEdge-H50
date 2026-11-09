// G exact pipeline: RAM mux -> captured word -> quantity register -> controls.
// Exact aggregate quantities with a sparse hash table and ordered price links.
// 32 BRAM banks compare candidate insertion neighbours in parallel. No price
// range clipping, no offline list of future prices, no discarded deep levels.
module fm_level_book #(parameter BITS=14,BANK_BITS=5)(
 input wire clk,rst,clear,
 input wire update_valid,output wire update_ready,input wire [31:0] update_price,
 input wire update_side,input wire signed [64:0] update_delta,
 output reg done_valid,input wire done_ready,
 input wire snapshot_request,output wire snapshot_ready,
 output reg snapshot_valid,input wire snapshot_accept,output reg snapshot_complete,
 output reg [2559:0] snapshot,
 output reg fault,output reg [31:0] error,output wire initialized,
 output reg [31:0] ask_levels,bid_levels
);
 localparam N=1<<BITS,BANKS=1<<BANK_BITS,ROWS=1<<(BITS-BANK_BITS);
 localparam INIT=0,IDLE=1,LOOK=2,MATCH=3,SCAN=4,SCAN_USE=5,STORE=6,
 PRE_READ=7,PRE_USE=8,PRE_WRITE=9,NEXT_READ=10,NEXT_USE=11,NEXT_WRITE=12,DONE=13,
 SNAP_START=14,SNAP_READ=15,SNAP_USE=16,SNAP_DONE=17,REDUCE=18,ADD_QUANTITY=19,APPLY_QUANTITY=20;
 reg [4:0] state;reg [BITS-BANK_BITS-1:0] init_row,scan_row;
 reg [BITS-1:0] index,chosen,tombstone,prev_index,next_index,scan_prev,scan_next;
 reg tomb_valid,prev_valid,next_valid,scan_prev_valid,scan_next_valid;
 reg [31:0] price,quality,scan_prev_quality,scan_next_quality,probe_count;
 reg bpv[0:BANKS-1],bnv[0:BANKS-1];reg [31:0] bpq[0:BANKS-1],bnq[0:BANKS-1];
 reg [BITS-1:0] bpi[0:BANKS-1],bni[0:BANKS-1];reg [BANK_BITS-1:0] reduce_bank;
 reg side,deleting;reg signed [64:0] delta;
 reg [128:0] word,scratch;reg [BITS-1:0] ask_head,bid_head;reg ask_valid,bid_valid;
 reg [3:0] rank;reg snap_side;reg [BITS-1:0] ask_cursor,bid_cursor;
 wire read_en=state==LOOK||state==SCAN||state==PRE_READ||state==NEXT_READ||state==SNAP_READ;
 wire [BITS-1:0] read_index=state==PRE_READ?prev_index:state==NEXT_READ?next_index:
                 state==SNAP_READ?(snap_side?bid_cursor:ask_cursor):index;
 wire [BITS-BANK_BITS-1:0] read_row=state==SCAN?scan_row:read_index[BITS-1:BANK_BITS];
 wire write_en=state==STORE||state==PRE_WRITE||state==NEXT_WRITE;
 wire [BITS-1:0] write_index=state==PRE_WRITE?prev_index:state==NEXT_WRITE?next_index:chosen;
 wire [128:0] write_word=state==STORE?word:scratch;
 wire [128:0] bank_data[0:BANKS-1];
 reg [BANK_BITS-1:0] read_bank;
 always @(posedge clk)if(read_en)read_bank<=read_index[BANK_BITS-1:0];
 generate for(genvar b=0;b<BANKS;b=b+1)begin:banks
  fm_level_ram #(.ADDR_BITS(BITS-BANK_BITS)) ram(.clk(clk),
   .we(state==INIT||(write_en&&write_index[BANK_BITS-1:0]==b)),
   .wa(state==INIT?init_row:write_index[BITS-1:BANK_BITS]),.wd(state==INIT?129'b0:write_word),
   .re(read_en),.ra(read_row),.rd(bank_data[b]));
 end endgenerate
 wire [128:0] selected=bank_data[read_bank];
 assign initialized=state!=INIT;assign update_ready=state==IDLE&&!fault&&!clear;
 assign snapshot_ready=state==IDLE&&!update_valid&&!fault&&!clear;
 reg [128:0] candidate;reg [31:0] cq,pq,nq;reg pv,nv;reg [BITS-1:0] pi,ni;
 reg signed [65:0] quantity_reg;integer k;reg [128:0] temp;
 task fail(input [31:0] why);begin fault<=1;error<=why;done_valid<=0;snapshot_valid<=0;end endtask
 function [BITS-1:0] hash(input [31:0] p,input s);
  begin hash=(p^(p>>12)^(p>>24)^(s?32'h9e3779b9:0));end
 endfunction
 always @(posedge clk)begin
  if(rst)begin state<=INIT;init_row<=0;fault<=0;error<=0;done_valid<=0;snapshot_valid<=0;
   ask_levels<=0;bid_levels<=0;ask_valid<=0;bid_valid<=0;end
  else if(!fault)begin
   if(clear)begin
    if(state!=IDLE)fail(1);
    else begin state<=INIT;init_row<=0;ask_levels<=0;bid_levels<=0;ask_valid<=0;bid_valid<=0;end
   end else case(state)
    INIT:if(init_row==ROWS-1)state<=IDLE;else init_row<=init_row+1'b1;
    IDLE:begin
     if(update_valid)begin
      if(update_price==0||update_delta==0)fail(2);
      else begin price<=update_price;side<=update_side;quality<=update_side?update_price:~update_price;
       delta<=update_delta;index<=hash(update_price,update_side);probe_count<=0;tomb_valid<=0;state<=LOOK;end
     end else if(snapshot_request)begin rank<=0;snap_side<=0;snapshot<=0;state<=SNAP_START;end
    end
    LOOK:state<=MATCH;
    MATCH:begin
     if(selected[111:110]==1&&selected[31:0]==price&&selected[112]==side)begin
      word<=selected;chosen<=index;state<=ADD_QUANTITY;
     end else if(selected[111:110]==0)begin
      if(delta<0)fail(4);
      else begin chosen<=tomb_valid?tombstone:index;scan_row<=0;scan_prev_valid<=0;scan_next_valid<=0;
       scan_prev_quality<=32'hffffffff;scan_next_quality<=0;
       for(k=0;k<BANKS;k=k+1)begin bpv[k]<=0;bnv[k]<=0;end
       state<=SCAN;end
     end else if(probe_count==N-1)fail(5);
     else begin
      if(selected[111:110]==2&&!tomb_valid)begin tombstone<=index;tomb_valid<=1;end
      index<=index+1'b1;probe_count<=probe_count+1;state<=LOOK;
     end
    end
    ADD_QUANTITY:begin
     quantity_reg<=$signed({2'b0,word[95:32]})+$signed({delta[64],delta});
     state<=APPLY_QUANTITY;
    end
    APPLY_QUANTITY:begin
      if(quantity_reg<0||quantity_reg[65:64]!=0)fail(3);
      else begin
       temp=word;temp[95:32]=quantity_reg[63:0];chosen<=index;
       if(quantity_reg==0)begin
        temp[111:110]=2;deleting<=1;prev_index<=word[113+:BITS];prev_valid<=word[128];
        next_index<=word[96+:BITS];next_valid<=word[127];
        if(side)begin bid_levels<=bid_levels-1;if(!word[128])begin bid_head<=word[96+:BITS];bid_valid<=word[127];end end
        else begin ask_levels<=ask_levels-1;if(!word[128])begin ask_head<=word[96+:BITS];ask_valid<=word[127];end end
       end else begin deleting<=0;prev_valid<=0;next_valid<=0;end
       word<=temp;state<=STORE;
      end
    end
    SCAN:state<=SCAN_USE;
    SCAN_USE:begin
     for(k=0;k<BANKS;k=k+1)begin
      candidate=bank_data[k];cq=side?candidate[31:0]:~candidate[31:0];
      if(candidate[111:110]==1&&candidate[112]==side)begin
       if(cq>quality&&(!bpv[k]||cq<bpq[k]))begin bpv[k]<=1;bpq[k]<=cq;bpi[k]<=(scan_row<<BANK_BITS)|k;end
       if(cq<quality&&(!bnv[k]||cq>bnq[k]))begin bnv[k]<=1;bnq[k]<=cq;bni[k]<=(scan_row<<BANK_BITS)|k;end
      end
     end
     if(scan_row==ROWS-1)begin reduce_bank<=0;state<=REDUCE;end
     else begin scan_row<=scan_row+1'b1;state<=SCAN;end
    end
    REDUCE:begin
     pv=scan_prev_valid;nv=scan_next_valid;pq=scan_prev_quality;nq=scan_next_quality;pi=scan_prev;ni=scan_next;
     if(bpv[reduce_bank]&&(!pv||bpq[reduce_bank]<pq))begin pv=1;pq=bpq[reduce_bank];pi=bpi[reduce_bank];end
     if(bnv[reduce_bank]&&(!nv||bnq[reduce_bank]>nq))begin nv=1;nq=bnq[reduce_bank];ni=bni[reduce_bank];end
     scan_prev_valid<=pv;scan_next_valid<=nv;scan_prev_quality<=pq;scan_next_quality<=nq;scan_prev<=pi;scan_next<=ni;
     if(reduce_bank==BANKS-1)begin
      temp=0;temp[31:0]=price;temp[95:32]=delta[63:0];temp[111:110]=1;temp[112]=side;
      temp[113+:BITS]=pi;temp[128]=pv;temp[96+:BITS]=ni;temp[127]=nv;
      word<=temp;prev_index<=pi;prev_valid<=pv;next_index<=ni;next_valid<=nv;deleting<=0;
      if(side)begin bid_levels<=bid_levels+1;if(!pv)begin bid_head<=chosen;bid_valid<=1;end end
      else begin ask_levels<=ask_levels+1;if(!pv)begin ask_head<=chosen;ask_valid<=1;end end
      state<=STORE;
     end else reduce_bank<=reduce_bank+1'b1;
    end
    STORE:if(prev_valid)state<=PRE_READ;else if(next_valid)state<=NEXT_READ;else state<=DONE;
    PRE_READ:state<=PRE_USE;
    PRE_USE:begin temp=selected;temp[96+:BITS]=deleting?next_index:chosen;temp[127]=deleting?next_valid:1'b1;scratch<=temp;state<=PRE_WRITE;end
    PRE_WRITE:if(next_valid)state<=NEXT_READ;else state<=DONE;
    NEXT_READ:state<=NEXT_USE;
    NEXT_USE:begin temp=selected;temp[113+:BITS]=deleting?prev_index:chosen;temp[128]=deleting?prev_valid:1'b1;scratch<=temp;state<=NEXT_WRITE;end
    NEXT_WRITE:state<=DONE;
    DONE:begin done_valid<=1;if(done_valid&&done_ready)begin done_valid<=0;state<=IDLE;end end
    SNAP_START:begin
     snapshot_complete<=ask_levels>=10&&bid_levels>=10;
     if(ask_levels<10||bid_levels<10)state<=SNAP_DONE;
     else begin ask_cursor<=ask_head;bid_cursor<=bid_head;state<=SNAP_READ;end
    end
    SNAP_READ:state<=SNAP_USE;
    SNAP_USE:begin
     if(selected[111:110]!=1||selected[112]!=snap_side)fail(6);
     else begin
      snapshot[(rank*4+(snap_side?2:0))*64+:64]<={32'b0,selected[31:0]};
      snapshot[(rank*4+(snap_side?3:1))*64+:64]<=selected[95:32];
      if(rank<9&&!selected[127])fail(7);
      else if(snap_side)begin bid_cursor<=selected[96+:BITS];snap_side<=0;if(rank==9)state<=SNAP_DONE;else begin rank<=rank+1;state<=SNAP_READ;end end
      else begin ask_cursor<=selected[96+:BITS];snap_side<=1;state<=SNAP_READ;end
     end
    end
    SNAP_DONE:begin snapshot_valid<=1;if(snapshot_valid&&snapshot_accept)begin snapshot_valid<=0;state<=IDLE;end end
    default:fail(8);
   endcase
  end
 end
endmodule

