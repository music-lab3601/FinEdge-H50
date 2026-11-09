// Real B book -> fused exact integer preprocessing + INT8 ->
// E two-slot scheduler -> shared DDR -> NPU -> hardware logits/argmax.
// ARM starts/waits for the resident NPU; ARM never copies an input tensor.
module g_predict_path #(
 parameter SLOTS=32,SW=$clog2(SLOTS),
 parameter ROM0="fixed_lane0.hex",ROM1="fixed_lane1.hex",ROM2="fixed_lane2.hex",ROM3="fixed_lane3.hex"
)(input wire por_n,clk,clk200,arm_pulse,configured,input wire [63:0] session,
 input wire [SLOTS*64-1:0] slot_bases,slot_caps,input wire [63:0] output_base,
 input wire [5:0] active_slots,input wire [63:0] age_limit_ticks,input wire [15:0] h50_records,
 input wire book_valid,output wire book_ready,input wire [2559:0] book_data,input wire [255:0] book_identity,
 input wire [63:0] book_accept_tick,
 output wire npu_request_valid,input wire npu_request_ready,output wire [255:0] npu_request,
 input wire npu_done_valid,output wire npu_done_ready,input wire [159:0] npu_done_token,
 output wire result_valid,input wire result_ready,output wire [383:0] result_data,
 input wire [7:0] status_index,output reg [63:0] status_data,
 output wire b_valid,input wire b_ready,output wire b_write,output wire b_fast_write,output wire [31:0] b_addr,output wire [511:0] b_data,
 input wire b_reply,input wire [511:0] b_rdata,input wire memory_fault,output wire fault);
 wire up;b1_reset_sync reset(clk,por_n,up);
 reg [63:0] tick;always @(posedge clk)if(!up)tick<=0;else tick<=tick+1;
 wire norm_valid,norm_ready,norm_fault,norm_in_ready;wire [319:0] code_record;
 wire [255:0] norm_id;wire [31:0] norm_error;
 assign book_ready=norm_in_ready&&configured&&!fault;
 fp_fixed_record #(.ROM0(ROM0),.ROM1(ROM1),.ROM2(ROM2),.ROM3(ROM3)) normalize(.clk(clk),.rst(!up),.s_valid(book_valid&&configured&&!fault),.s_ready(norm_in_ready),
  .s_book(book_data),.s_identity(book_identity),.m_valid(norm_valid),.m_ready(norm_ready),
  .m_data(code_record),.m_identity(norm_id),.fault(norm_fault),.error(norm_error));
 reg [63:0] row_accept_tick,row_number;
 always @(posedge clk)begin
  if(!up)begin row_accept_tick<=0;row_number<=0;end
  else begin
   if(book_valid&&book_ready)row_accept_tick<=book_accept_tick;
   if(norm_valid&&norm_ready)row_number<=row_number+1;
  end
 end
 wire frame_credit;wire [63:0] fifo_peak,capture_cycles,last_capture_cycles,last_transfer_cycles,verify_retries;
 wire [SLOTS-1:0] approved;
 wire offer_valid,offer_ready;wire [1279:0] offer_desc;
 wire write_valid,write_ready;wire [SW-1:0] write_slot;wire [63:0] write_session,write_generation;wire [1279:0] write_desc;
 wire commit_valid,commit_ready,commit_ok;wire [159:0] commit_token;
 wire event_valid,event_ready,event_has_slot;wire [SW-1:0] event_slot;wire [1279:0] event_desc;
 wire [63:0] event_session,event_generation;wire [7:0] event_reason;wire [127:0] event_result;
 wire mem_valid,mem_ready,mem_last;wire [31:0] mem_addr;wire [511:0] mem_data;wire [63:0] mem_keep;
 wire fence_valid,fence_ready,fence_done;wire [159:0] fence_token,fence_done_token;
 wire quarantine;wire [159:0] quarantine_token;
 wire active;wire [3:0] contexts;wire decision_valid;wire [7:0] decision_reason;
 wire [63:0] decision_window;wire [127:0] decision_cookie;
 wire [63:0] records,opportunities,gate_skipped,local_admitted,reject_busy,reject_full,reject_epoch,E_admitted,retired;
 wire [63:0] encoded_rows,encoded_features,invalid_encoded,f_errors,scratch_released,offer_source_tick;wire f_protocol_fault;
 tt_native_transport #(.INPUT_ENCODED(1),.ROM0(ROM0),.ROM1(ROM1),.ROM2(ROM2),.ROM3(ROM3),.SLOTS(SLOTS)) transport(
  .clk(clk),.rst(!up),.arm_valid(arm_pulse),.arm_ready(),.io_quiescent(1'b1),
  .cfg_session(session),.cfg_max_wait(age_limit_ticks),
  .cfg_quant_hash(256'h0eb1627ef8f83a72a63d1d846120458baa55a3be1e9a1a9a6902218f37e037f4),
  .cfg_model_hash(256'ha9a4822bcbce441a6565d938be3f052c33a36464c20ff8cfeb3a5125299fcc44),
  .s_valid(norm_valid),.s_ready(norm_ready),.s_data({960'd0,code_record}),.s_id({row_accept_tick,norm_id,session}),
  .s_emit(1'b1),.s_clear(1'b0),.s_window_id(row_number),.frame_credit(frame_credit),
  .offer_valid(offer_valid),.offer_ready(offer_ready),.offer_desc(offer_desc),.offer_source_tick(offer_source_tick),
  .write_valid(write_valid),.write_ready(write_ready),.write_session(write_session),.write_generation(write_generation),
  .write_slot({{(32-SW){1'b0}},write_slot}),.write_desc(write_desc),.allocation_approved(approved),
  .allocation_bases(slot_bases),.allocation_capacities(slot_caps),
  .commit_valid(commit_valid),.commit_ready(commit_ready),.commit_ok(commit_ok),.commit_token(commit_token),
  .terminal_valid(event_valid),.terminal_sink_ready(event_ready),.terminal_ready(),.terminal_desc(event_desc),
  .terminal_has_slot(event_has_slot),.terminal_token({event_session,event_generation,{(32-SW){1'b0}},event_slot}),
  .mem_valid(mem_valid),.mem_ready(mem_ready),.mem_addr(mem_addr),.mem_data(mem_data),.mem_keep(mem_keep),.mem_last(mem_last),
  .fence_valid(fence_valid),.fence_ready(fence_ready),.fence_token(fence_token),.fence_done(fence_done),.fence_done_token(fence_done_token),
  .fence_error(1'b0),.external_fault(memory_fault),.quarantine_valid(quarantine),.quarantine_token(quarantine_token),
  .drain_proven(1'b0),.drain_token(160'd0),.active(active),.context_states(contexts),
  .decision_valid(decision_valid),.decision_reason(decision_reason),.decision_window(decision_window),.decision_cookie(decision_cookie),
  .records(records),.opportunities(opportunities),.gate_selected(),.gate_skipped(gate_skipped),.local_admitted(local_admitted),
  .rejected_copy_busy(reject_busy),.rejected_full(reject_full),.rejected_epoch(reject_epoch),.E_admitted(E_admitted),.retired(retired),
  .features_encoded(encoded_features),.records_encoded(encoded_rows),.invalid_encoded(invalid_encoded),.protocol_errors(f_errors),.protocol_fault(f_protocol_fault),.scratch_released(scratch_released));
 generate for(genvar sl=0;sl<SLOTS;sl=sl+1)begin:approve
  assign approved[sl]=configured&&sl<active_slots;
 end endgenerate
 wire npu_valid;wire [SW-1:0] npu_slot;wire [63:0] npu_session,npu_generation;wire [1279:0] npu_desc;
 reg sentinel_pending,sentinel_ready;wire npu_accept_ready=npu_request_ready&&sentinel_ready;
 assign npu_request_valid=npu_valid&&sentinel_ready;
 assign npu_request={32'd30,npu_desc[223:160],npu_session,npu_generation,{(32-SW){1'b0}},npu_slot};
 wire e_commit_valid,e_commit_ready;wire [SW-1:0] e_commit_slot;wire [63:0] bad_commits;
 e_slot32_guard #(.SLOTS(SLOTS),.SW(SW)) guard(.clk(clk),.reset_n(por_n),.active(active),.s_valid(commit_valid),.s_ready(commit_ready),
  .s_slot(commit_token[31:0]),.m_valid(e_commit_valid),.m_ready(e_commit_ready),.m_slot(e_commit_slot),
  .local_slot(write_slot),.wire_slot(),.rejected_count(bad_commits));
 wire e_done_valid,e_done_ready;reg [159:0] completed_token;reg [127:0] completed_result;
 wire e_fault;wire [63:0] e_errors;wire [5:0] occupied,queue_peak;wire scheduler_quiescent;
 wire [63:0] e_admitted,e_served,e_completed,e_cancelled,e_expired,expired_h50,expired_age;
 wire [63:0] completed_h50_late,completed_age_late,late_started,event_source_tick,event_launch_tick;
 reg [63:0] segment;
 always @(posedge clk)if(!up)segment<=1;else if(book_valid&&book_ready)segment<=book_identity[63:0];
 g_queue_scheduler #(.SLOTS(SLOTS),.SW(SW)) scheduler(
  .clk(clk),.reset_n(por_n),.arm_valid(arm_pulse),.arm_session(session),
  .cfg_slots(active_slots),.age_limit_ticks(age_limit_ticks),.h50_records(h50_records),
  .now_tick(tick),.latest_rows(records),.latest_segment(segment),
  .s_valid(offer_valid),.s_ready(offer_ready),.s_desc(offer_desc),.s_source_tick(offer_source_tick),
  .write_valid(write_valid),.write_ready(write_ready),.write_slot(write_slot),.write_session(write_session),
  .write_generation(write_generation),.write_desc(write_desc),
  .commit_valid(e_commit_valid),.commit_ready(e_commit_ready),.commit_slot(e_commit_slot),
  .commit_session(commit_token[159:96]),.commit_generation(commit_token[95:32]),.commit_ok(commit_ok),
  .npu_valid(npu_valid),.npu_ready(npu_accept_ready),.npu_slot(npu_slot),.npu_session(npu_session),
  .npu_generation(npu_generation),.npu_desc(npu_desc),
  .done_valid(e_done_valid),.done_ready(e_done_ready),.done_slot(completed_token[SW-1:0]),
  .done_session(completed_token[159:96]),.done_generation(completed_token[95:32]),.done_result(completed_result),
  .event_valid(event_valid),.event_ready(event_ready),.event_desc(event_desc),.event_has_slot(event_has_slot),
  .event_slot(event_slot),.event_session(event_session),.event_generation(event_generation),
  .event_reason(event_reason),.event_result(event_result),.event_source_tick(event_source_tick),.event_launch_tick(event_launch_tick),
  .active(),.fault(e_fault),.protocol_errors(e_errors),.occupied(occupied),.queue_peak(queue_peak),
  .admitted(e_admitted),.served(e_served),.completed(e_completed),.cancelled(e_cancelled),.expired(e_expired),
  .expired_h50(expired_h50),.expired_age(expired_age),.completed_h50_late(completed_h50_late),
  .completed_age_late(completed_age_late),.late_started(late_started),.quiescent(scheduler_quiescent));
 wire output_valid,output_ready,output_reply;wire [511:0] output_data;wire bridge_fault;
 wire [63:0] written_bytes,read_bytes,control_written_bytes;
 reg [63:0] visibility_retries;
 reg output_pending,output_complete,read_retry;reg [7:0] retry_count;
 wire clear_request=npu_valid&&!sentinel_ready&&!sentinel_pending;
 assign output_valid=clear_request||read_retry||(npu_done_valid&&!output_pending);
 assign npu_done_ready=output_ready&&!output_pending&&!clear_request;
 assign e_done_valid=output_complete;
 g_payload_bridge memory(.por_n(por_n),.clk125(clk),.clk200(clk200),
  .f_valid(mem_valid),.f_ready(mem_ready),.frame_credit(frame_credit),.f_addr(mem_addr),.f_data(mem_data),.f_last(mem_last),
  .fence_valid(fence_valid),.fence_ready(fence_ready),.fence_token(fence_token),
  .fence_done(fence_done),.fence_done_token(fence_done_token),
  .output_valid(output_valid),.output_ready(output_ready),.output_addr(output_base[31:0]),
  .output_write(clear_request),.output_wdata({416'd0,32'h7fc00000,32'h7fc00000,32'h7fc00000}),
  .output_reply(output_reply),.output_data(output_data),
  .b_valid(b_valid),.b_ready(b_ready),.b_write(b_write),.b_fast_write(b_fast_write),.b_addr(b_addr),.b_data(b_data),
  .b_reply(b_reply),.b_rdata(b_rdata),.fault(bridge_fault),.written_bytes(written_bytes),.read_bytes(read_bytes),.control_written_bytes(control_written_bytes),.fifo_peak(fifo_peak),.capture_cycles(capture_cycles),.last_capture_cycles(last_capture_cycles),.last_transfer_cycles(last_transfer_cycles),.verify_retries(verify_retries));
 function [31:0] float_key(input [31:0] x);begin
  // IEEE signed zeros compare equal; raw argmax keeps the first equal logit.
  float_key=x[30:0]==0?32'h80000000:(x[31]?~x:(x^32'h80000000));
 end endfunction
 function [31:0] argmax3(input [95:0] x);
  reg [31:0] best;reg [31:0] index;begin best=float_key(x[31:0]);index=0;
   if(float_key(x[63:32])>best)begin best=float_key(x[63:32]);index=1;end
   if(float_key(x[95:64])>best)index=2;
   argmax3=index;
  end
 endfunction
 reg local_fault;
 always @(posedge clk)begin
  if(!up)begin output_pending<=0;output_complete<=0;completed_token<=0;completed_result<=0;local_fault<=0;
   sentinel_pending<=0;sentinel_ready<=0;read_retry<=0;retry_count<=0;visibility_retries<=0;end
  else begin
   if(clear_request&&output_ready)sentinel_pending<=1;
   if(npu_valid&&npu_accept_ready)sentinel_ready<=0;
   if(read_retry&&output_ready)read_retry<=0;
   if(npu_done_valid&&npu_done_ready)begin
    if(npu_done_token[31:0]>=SLOTS) local_fault<=1;
    completed_token<=npu_done_token;output_pending<=1;retry_count<=0;
   end
   if(output_reply)begin
    if(sentinel_pending)begin sentinel_pending<=0;sentinel_ready<=1;end
    else if(!output_pending)local_fault<=1;
    else if(output_data[30:23]==8'hff||output_data[62:55]==8'hff||output_data[94:87]==8'hff)begin
     if(retry_count==255)local_fault<=1;
     else begin read_retry<=1;retry_count<=retry_count+1;visibility_retries<=visibility_retries+1;end
    end else begin completed_result<={argmax3(output_data[95:0]),output_data[95:0]};output_complete<=1;end
   end
   if(e_done_valid&&e_done_ready)begin output_pending<=0;output_complete<=0;end
  end
 end
 // The scheduler retains window identity/timestamps in its DDR-slot metadata
 // BRAM after the two scratch contexts have already been reused.
 reg result_pending;reg [383:0] result_hold;wire meta_fault=1'b0;
 assign result_valid=result_pending;assign result_data=result_hold;
 assign event_ready=result_pending&&result_ready;
 always @(posedge clk)begin
  if(!up)begin result_pending<=0;result_hold<=0;end
  else begin
   if(event_valid&&!result_pending)begin
    result_hold<={(event_reason==0?tick-event_launch_tick:64'd0),tick-event_source_tick,
                 event_desc[351:288],event_desc[223:160],event_result};
    result_hold[127:120]<=event_reason;result_pending<=1;
   end
   if(event_valid&&event_ready)result_pending<=0;
  end
 end
 assign fault=norm_fault||quarantine||e_fault||bridge_fault||local_fault||meta_fault||memory_fault||f_protocol_fault||(|e_errors)||(|bad_commits);
 always @*begin
  status_data=0;
  case(status_index)
   0:status_data={55'd0,meta_fault,local_fault,bridge_fault,quarantine,e_fault,norm_fault,configured,active,fault};
   1:status_data=records;2:status_data=opportunities;3:status_data=local_admitted;4:status_data=E_admitted;
   5:status_data=retired;6:status_data=reject_busy;7:status_data=reject_full;8:status_data=reject_epoch;
   9:status_data=gate_skipped;10:status_data=written_bytes;11:status_data=read_bytes;
   12:status_data=encoded_rows;13:status_data=encoded_features;14:status_data=f_errors;15:status_data=e_errors;
   16:status_data={53'd0,contexts,occupied,!scheduler_quiescent};17:status_data=tick;18:status_data=row_number;
   19:status_data=e_admitted;20:status_data=e_completed;21:status_data=e_cancelled;
   22:status_data={32'd0,norm_error};23:status_data=invalid_encoded;
   24:status_data=control_written_bytes;25:status_data=visibility_retries;
   26:status_data=scratch_released;27:status_data=queue_peak;28:status_data=occupied;
   29:status_data=e_expired;30:status_data=expired_h50;31:status_data=expired_age;
   36:status_data=fifo_peak;37:status_data=last_capture_cycles;38:status_data=last_transfer_cycles;39:status_data=capture_cycles;40:status_data=verify_retries;
   32:status_data=completed_h50_late;33:status_data=completed_age_late;34:status_data=records;35:status_data=late_started;
  endcase
 end
endmodule
