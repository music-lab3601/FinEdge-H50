`timescale 1ns/1ps
// Rolling history only; NO full-window scratch/snapshot RAM. A window is
// offered atomically only when E, the capture front and FIFO have credit.
module wt_history #(
 parameter [255:0] QUANT_HASH=256'h0eb1627ef8f83a72a63d1d846120458baa55a3be1e9a1a9a6902218f37e037f4,
 parameter [255:0] MIXED_HASH=256'ha9a4822bcbce441a6565d938be3f052c33a36464c20ff8cfeb3a5125299fcc44,
 parameter [255:0] FALLBACK_HASH=256'hde6af192e8781e5509fd33769475f2abb7277f802fce9dcb77e9879760fb3293
)(input wire clk,rst,arm_valid,output wire arm_ready,input wire io_quiescent,
 input wire [63:0] cfg_session,cfg_max_wait,input wire [255:0] cfg_quant_hash,cfg_model_hash,
 input wire s_valid,output wire s_ready,input wire [319:0] s_data,input wire [383:0] s_id,
 input wire s_emit,s_clear,input wire [63:0] s_window_id,input wire window_ready,
 output wire offer_valid,input wire offer_ready,output reg [1279:0] offer_desc,output wire [63:0] offer_source_tick,
 input wire [127:0] lookup_cookie,output reg lookup_found,lookup_bindable,lookup_context,
 output reg [1279:0] lookup_desc,output reg [383:0] lookup_first,lookup_end,
 input wire bind_valid,bind_context,input wire [159:0] bind_token,
 input wire committed_valid,committed_context,input wire [159:0] committed_token,
 input wire terminal_valid,output wire terminal_ready,input wire [127:0] terminal_cookie,
 input wire terminal_has_slot,input wire [159:0] terminal_token,
 input wire read_enable,read_context,input wire [6:0] read_row,output reg [319:0] read_data,
 output reg active,output reg [6:0] history_count,output wire [3:0] context_states,
 output reg decision_valid,output reg [7:0] decision_reason,output reg [63:0] decision_window,
 output reg [127:0] decision_cookie,
 output reg [63:0] records,opportunities,gate_selected,gate_skipped,local_admitted,
 output reg [63:0] rejected_copy_busy,rejected_full,rejected_epoch,E_admitted,retired,
 output reg [63:0] history_resets,invalid_records,protocol_errors,copy_rows,config_errors,scratch_released);
 reg [7:0] head,start[0:1];reg [383:0] previous,first[0:1],last[0:1];
 reg [1279:0] desc[0:1];reg owned[0:1],bound[0:1];reg [159:0] token[0:1];
 reg [63:0] session,next_cookie,max_wait,next_row;reg [255:0] quant_hash,model_hash;
 reg context_valid,awaiting_bind;
 (* ram_style="block" *) reg [319:0] ring[0:255];
 (* ram_style="block" *) reg [383:0] identities[0:255];
 reg [383:0] id_q;reg read_check;reg [191:0] expected_id;
 wire identity_ok=s_id[63:0]==session&&s_id[127:64]!=0&&s_id[255:192]!=0;
 wire same_context=context_valid&&s_id[127:0]==previous[127:0];
 wire continuous=previous[191:128]!=64'hffffffffffffffff&&s_id[191:128]==next_row&&
                 s_id[255:192]>previous[255:192]&&s_id[319:256]>=previous[319:256];
 wire fresh=!same_context||!continuous;
 wire [6:0] next_count=fresh?1:history_count<100?history_count+1'b1:100;
 assign s_ready=!rst&&active;
 wire input_fire=s_valid&&s_ready;
 wire record_fire=input_fire&&!s_clear&&identity_ok;
 wire eligible=record_fire&&next_count==100&&s_emit;
 wire free_valid=!owned[0]||!owned[1];wire free_context=owned[0];
 // E.ready does not depend on E.valid: this is an atomic non-blocking offer.
 assign offer_valid=eligible&&free_valid&&!awaiting_bind&&window_ready&&offer_ready&&next_cookie!=64'hffffffffffffffff;
 assign offer_source_tick=s_id[383:320];
 assign arm_ready=!rst&&!active&&io_quiescent;
 assign terminal_ready=!rst&&active;
 assign context_states={owned[1]?2'd3:2'd0,owned[0]?2'd3:2'd0};
 integer k,terminal_context;
 always @*begin
  offer_desc=0;offer_desc[63:0]=session;offer_desc[159:96]=s_id[127:64];
  offer_desc[223:160]=s_window_id;offer_desc[287:224]=id_q[255:192];
  offer_desc[351:288]=s_id[255:192];offer_desc[415:352]=s_id[319:256];offer_desc[479:416]=max_wait;
  offer_desc[511:480]=4000;offer_desc[543:512]=4800;offer_desc[559:544]=1;offer_desc[575:560]=1;
  offer_desc[863:608]=quant_hash;offer_desc[1119:864]=model_hash;offer_desc[1247:1120]={session,next_cookie};
  lookup_found=0;lookup_bindable=0;lookup_context=0;lookup_desc=0;lookup_first=0;lookup_end=0;terminal_context=-1;
  for(k=0;k<2;k=k+1)begin
   if(owned[k]&&desc[k][1247:1120]==lookup_cookie)begin
    lookup_found=1;lookup_bindable=!bound[k];lookup_context=k;
    lookup_desc=desc[k];lookup_first=first[k];lookup_end=last[k];
   end
   if(owned[k]&&desc[k][1247:1120]==terminal_cookie)terminal_context=k;
  end
 end
 wire [7:0] ring_address=start[read_context]+read_row;
 wire [7:0] prefetch_address=head+(record_fire?8'd1:8'd0)-8'd99;
 // One write port and one read port per RAM. Identity prefetch is idle-time
 // work, so direct data reads need no replicated identity RAM.
 always @(posedge clk)begin
  if(record_fire)begin ring[head]<=s_data;identities[head]<={64'd0,s_id[319:0]};end
  id_q<=identities[read_enable?ring_address:prefetch_address];
  if(read_enable)read_data<=ring[ring_address];
 end
 reg [2:0] errors;
 integer j;
 always @(posedge clk)begin
  if(rst)begin
   active<=0;history_count<=0;head<=0;context_valid<=0;previous<=0;awaiting_bind<=0;
   session<=0;next_cookie<=1;max_wait<=0;next_row<=0;quant_hash<=0;model_hash<=0;
   read_check<=0;expected_id<=0;decision_valid<=0;decision_reason<=0;decision_window<=0;decision_cookie<=0;
   records<=0;opportunities<=0;gate_selected<=0;gate_skipped<=0;local_admitted<=0;
   rejected_copy_busy<=0;rejected_full<=0;rejected_epoch<=0;E_admitted<=0;retired<=0;
   history_resets<=0;invalid_records<=0;protocol_errors<=0;copy_rows<=0;config_errors<=0;scratch_released<=0;
   for(j=0;j<2;j=j+1)begin owned[j]<=0;bound[j]<=0;start[j]<=0;first[j]<=0;last[j]<=0;desc[j]<=0;token[j]<=0;end
  end else begin
   errors=0;decision_valid<=0;
   read_check<=read_enable;
   if(read_enable)begin
    expected_id<={last[read_context][191:128]-64'd99+{57'd0,read_row},last[read_context][127:0]};
    copy_rows<=copy_rows+1;
    if(!owned[read_context]||!bound[read_context])errors=errors+1;
   end
   if(read_check&&id_q[191:0]!==expected_id)errors=errors+1;
   if(arm_valid&&arm_ready)begin
    if(cfg_session==0||cfg_max_wait[63]||cfg_quant_hash!=QUANT_HASH||(cfg_model_hash!=MIXED_HASH&&cfg_model_hash!=FALLBACK_HASH))config_errors<=config_errors+1;
    else begin active<=1;session<=cfg_session;max_wait<=cfg_max_wait;quant_hash<=cfg_quant_hash;model_hash<=cfg_model_hash;end
   end
   if(input_fire)begin
    if(s_clear||!identity_ok)begin context_valid<=0;history_count<=0;history_resets<=history_resets+1;if(!identity_ok)invalid_records<=invalid_records+1;end
    else begin
     records<=records+1;head<=head+1'b1;history_count<=next_count;previous<=s_id;next_row<=s_id[191:128]+1;context_valid<=1;
     if(fresh)history_resets<=history_resets+1;
     if(next_count==100)begin
      opportunities<=opportunities+1;decision_valid<=1;decision_window<=s_window_id;decision_cookie<=0;
      if(!s_emit)begin gate_skipped<=gate_skipped+1;decision_reason<=1;end
      else begin
       gate_selected<=gate_selected+1;
       if(next_cookie==64'hffffffffffffffff)begin rejected_epoch<=rejected_epoch+1;decision_reason<=4;end
       else if(offer_valid)begin
        local_admitted<=local_admitted+1;E_admitted<=E_admitted+1;next_cookie<=next_cookie+1;awaiting_bind<=1;
        owned[free_context]<=1;bound[free_context]<=0;start[free_context]<=head-99;
        first[free_context]<=id_q;last[free_context]<={64'd0,s_id[319:0]};desc[free_context]<=offer_desc;
        decision_cookie<={session,next_cookie};decision_reason<=0;
       end else begin rejected_full<=rejected_full+1;decision_reason<=3;end
      end
     end
    end
   end
   if(bind_valid)begin
    if(owned[bind_context]&&!bound[bind_context]&&bind_token[159:96]==session)begin bound[bind_context]<=1;token[bind_context]<=bind_token;awaiting_bind<=0;end
    else errors=errors+1;
   end
   // Captured into the immutable window FIFO, independent of DDR visibility.
   if(committed_valid)begin
    if(owned[committed_context]&&bound[committed_context]&&token[committed_context]==committed_token)begin owned[committed_context]<=0;bound[committed_context]<=0;scratch_released<=scratch_released+1;end
    else errors=errors+1;
   end
   if(terminal_valid&&terminal_ready)begin
    if(terminal_has_slot)begin if(terminal_token[159:96]!=session||terminal_token[95:32]==0)errors=errors+1;else retired<=retired+1;end
    else if(terminal_context>=0&&!bound[terminal_context])begin owned[terminal_context]<=0;retired<=retired+1;awaiting_bind<=0;end
    else errors=errors+1;
   end
   if(errors!=0)protocol_errors<=protocol_errors+errors;
  end
 end
endmodule
