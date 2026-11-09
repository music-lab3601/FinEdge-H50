`timescale 1ns/1ps
module tb_history;
 reg c=0;always #4 c=~c;reg rst=1,arm=0,sv=0;wire sr;
 reg [319:0] data;reg [383:0] id;reg [63:0] win;
 wire ov;wire [1279:0] desc;reg [1279:0] current;reg busy=0;integer delay_count=0;
 wire found,bindable,ctx;wire [383:0] first,last;wire [1279:0] looked;
 reg bind_v=0,bindctx=0;reg [159:0] token;
 wire startready,pv,plast,re;wire [511:0] pd;wire [6:0] pi,rr;
 wire [63:0] unused_windows,unused_reads,unused_blocked;
 wire rctx;wire [319:0] rd;reg start=0,cap=0,capctx=0;wire active;
 wire [63:0] records,ops,admit,reject,busy_reject,retired,errors,released;
 integer sent=0,packets=0,j,index,row,feature,slot;reg [511:0] expected;reg [74:0] seen=0;
 wt_history history(.clk(c),.rst(rst),.arm_valid(arm),.arm_ready(),.io_quiescent(1'b1),.cfg_session(64'd1),.cfg_max_wait(64'd0),
 .cfg_quant_hash(256'h0eb1627ef8f83a72a63d1d846120458baa55a3be1e9a1a9a6902218f37e037f4),.cfg_model_hash(256'ha9a4822bcbce441a6565d938be3f052c33a36464c20ff8cfeb3a5125299fcc44),
 .s_valid(sv),.s_ready(sr),.s_data(data),.s_id(id),.s_emit(1'b1),.s_clear(1'b0),.s_window_id(win),.window_ready(!busy),
 .offer_valid(ov),.offer_ready(1'b1),.offer_desc(desc),.offer_source_tick(),
 .lookup_cookie(current[1247:1120]),.lookup_found(found),.lookup_bindable(bindable),.lookup_context(ctx),.lookup_desc(looked),.lookup_first(first),.lookup_end(last),
 .bind_valid(bind_v),.bind_context(bindctx),.bind_token(token),.committed_valid(cap),.committed_context(capctx),.committed_token(token),
 .terminal_valid(cap),.terminal_ready(),.terminal_cookie(current[1247:1120]),.terminal_has_slot(1'b1),.terminal_token(token),
 .read_enable(re),.read_context(rctx),.read_row(rr),.read_data(rd),.active(active),.history_count(),.context_states(),
 .decision_valid(),.decision_reason(),.decision_window(),.decision_cookie(),.records(records),.opportunities(ops),.gate_selected(),.gate_skipped(),.local_admitted(admit),
 .rejected_copy_busy(busy_reject),.rejected_full(reject),.rejected_epoch(),.E_admitted(),.retired(retired),.history_resets(),.invalid_records(),.protocol_errors(errors),.copy_rows(),.config_errors(),.scratch_released(released));
 tt_native_packer pack(c,rst,start,startready,bindctx,re,rctx,rr,rd,pv,1'b1,pd,plast,pi,unused_windows,unused_reads,unused_blocked);
 always @(posedge c)if(!rst)begin
  if(errors)$fatal(1,"Direct history identity error %0d",errors);
  if(ov)begin
   if(busy)$fatal(1,"unreserved ring read");busy<=1;current<=desc;delay_count<=3;seen<=0;
  end
  bind_v<=0;start<=0;cap<=0;
  if(delay_count>0)begin
   delay_count<=delay_count-1;
   if(delay_count==2)begin
    if(!found||!bindable||first[191:128]!=current[223:160]-99||last[191:128]!=current[223:160])$fatal(1,"prefetched first/last identity");
    bind_v<=1;bindctx<=ctx;capctx<=ctx;token<={64'd1,64'(packets+1),32'd0};
   end
   if(delay_count==1)begin if(!startready)$fatal(1,"packer busy at lease");start<=1;end
  end
  if(pv)begin
   if(pi>=75||seen[pi])$fatal(1,"duplicate physical beat");seen[pi]<=1;
   expected=0;
   for(j=0;j<64;j=j+1)begin
    feature=(pi/25)*16+j%16;row=(pi%25)*4+j/16;
    if(feature<40)expected[j*8+:8]=((current[223:160]-99+row)*7+feature)&255;
   end
   if(pd!==expected)$fatal(1,"Direct ring data differs window%0d index%0d",current[223:160],pi);
   if(plast)begin
    if(pi!=74||((seen| (75'd1<<pi))!={75{1'b1}}))$fatal(1,"incomplete window");
    packets<=packets+1;cap<=1;busy<=0;
   end
  end
 end
 initial begin
  repeat(8)@(negedge c);rst=0;arm=1;@(negedge c);arm=0;wait(active);
  for(sent=0;sent<1000;sent=sent+1)begin
   @(negedge c);sv=1;win=sent;
   id={64'(sent+100000),64'(sent+1),64'(sent+1000),64'(sent),64'(sent<500?1:2),64'd1};
   for(j=0;j<40;j=j+1)data[j*8+:8]=(sent*7+j)&255;
  end
  @(negedge c);sv=0;wait(!busy);repeat(8)@(negedge c);
  if(records!=1000||ops!=802||admit!=packets||retired!=packets||released!=packets||ops!=admit+reject+busy_reject||packets<4)$fatal(1,"ring conservation");
  $display("WT_HISTORY_PASS rows=%0d opportunities=%0d windows=%0d ring_wraps=3 segment_change=1",records,ops,packets);$finish;
 end
 initial begin #2000000;$fatal(1,"history timeout");end
endmodule
