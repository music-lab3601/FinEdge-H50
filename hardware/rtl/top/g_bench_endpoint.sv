// Board wrapper for B's frozen frontend. No inference or normalization.
module g_bench_endpoint(
 input wire por_n,c_clk,d_clk,fault_event,
 input wire [31:0] reg_awaddr,reg_awdata,input wire reg_awvalid,output wire reg_awready,
 input wire [31:0] reg_araddr,input wire reg_arvalid,output wire reg_arready,
 output wire [31:0] reg_rdata,output wire reg_rvalid,input wire reg_rready,
 output wire [31:0] awwaddr,output wire [511:0] awwdata,output wire [71:0] awwinfo,output wire awwvalid,input wire awwready,
 output wire [31:0] araddr,arinfo,output wire arvalid,input wire arready,
 input wire [511:0] rdata,input wire [31:0] rinfo,input wire rvalid,output wire rready,
 input wire PL_PHY_RX_CLK,input wire [3:0] PL_PHY_RXD,input wire PL_PHY_RX_CTRL,
 output wire PL_PHY_TX_CLK,output wire [3:0] PL_PHY_TXD,output wire PL_PHY_TX_CTRL,
 output wire PL_PHY_MDC,PL_PHY_RESET,inout wire PL_PHY_MDIO
);
 wire clk125,clk125_90,locked;
 fm_pl_clocks clocks(.clk_in1(d_clk),.reset(!por_n),.clk_out1(clk125),.clk_out2(clk125_90),.locked(locked));
 reg clock_started;
 always @(posedge d_clk or negedge por_n)if(!por_n)clock_started<=0;else if(locked)clock_started<=1;
 wire session_por=por_n&&clock_started;
 (* ASYNC_REG="TRUE" *) reg [1:0] stop_sync;
 always @(posedge clk125 or negedge session_por)
  if(!session_por)stop_sync<=0;else stop_sync<={stop_sync[0],fault_event||!locked};
 wire predict_configured,predict_book_ready,predict_fault,predict_control_fault;
 wire [63:0] book_accept_tick;
 wire udp_result_valid,udp_result_ready;wire [7:0] udp_result_addr,udp_result_byte;
 wire tensor_valid,tensor_ready,tensor_write,tensor_reply,tensor_fault,tensor_fast_write;
 wire [31:0] tensor_addr;wire [511:0] tensor_data,tensor_rdata;
 wire select_pw=reg_awaddr[15:12]==4'h2,select_pr=reg_araddr[15:12]==4'h2;
 wire old_awready,old_arready,old_rvalid,new_awready,new_arready,new_rvalid;
 wire [31:0] old_rdata,new_rdata;
 assign reg_awready=select_pw?new_awready:old_awready;
 assign reg_arready=select_pr?new_arready:old_arready;
 assign reg_rvalid=old_rvalid||new_rvalid;assign reg_rdata=old_rvalid?old_rdata:new_rdata;
 (* ASYNC_REG="TRUE" *) reg [1:0] predict_control_sync,tensor_fault_sync;
 always @(posedge clk125 or negedge session_por)
  if(!session_por)begin predict_control_sync<=0;tensor_fault_sync<=0;end
  else begin predict_control_sync<={predict_control_sync[0],predict_control_fault};tensor_fault_sync<={tensor_fault_sync[0],tensor_fault};end
 wire enable,capture_mode,memory_initialized,finish_valid,control_fault;
 wire [63:0] session,measure_first,measure_last,expected_last;
 wire [31:0] order_base,host_ip;
 wire [11:0] monitor_addr;wire [63:0] monitor_data;
 wire book_valid,book_frame_valid;wire [2559:0] book_data;wire [255:0] book_identity;wire [3327:0] book_frame;
 wire cap_ready,cap_empty,cap_busy,cap_pop_fault,cap_read,cap_pop;wire [7:0] cap_available;
 wire [6:0] cap_word;wire [31:0] cap_data;wire [63:0] captured_rows;
 wire in_range=book_identity[191:128]>=measure_first&&book_identity[191:128]<=measure_last;
 wire sink_ready=(!capture_mode||!in_range||cap_ready)&&(!predict_configured||predict_book_ready);
 wire initialized,front_fault,finished;wire [31:0] front_error;
 wire [63:0] raw_committed,eligible_count,core_cycle;
 wire mdio_o,mdio_oe;
 assign PL_PHY_MDIO=mdio_oe?mdio_o:1'bz;
 g_board_control control(.por_n(session_por),.c_clk(c_clk),.b_clk(clk125),
  .awaddr(reg_awaddr),.awdata(reg_awdata),.awvalid(reg_awvalid&&!select_pw),.awready(old_awready),
  .araddr(reg_araddr),.arvalid(reg_arvalid&&!select_pr),.arready(old_arready),.rdata(old_rdata),.rvalid(old_rvalid),.rready(reg_rready),
  .enable(enable),.session(session),.measure_first(measure_first),.measure_last(measure_last),
  .order_base(order_base),.host_ip(host_ip),.capture_mode(capture_mode),.memory_initialized(memory_initialized),
  .finish_valid(finish_valid),.expected_last(expected_last),.monitor_addr(monitor_addr),.monitor_data(monitor_data),
  .capture_empty(cap_empty),.capture_available(cap_available),.capture_read(cap_read),.capture_data(cap_data),
  .capture_word(cap_word),.capture_pop(cap_pop),.capture_pop_fault(cap_pop_fault),
  .phy_reset_n(PL_PHY_RESET),.phy_mdc(PL_PHY_MDC),.phy_mdio_o(mdio_o),.phy_mdio_oe(mdio_oe),.phy_mdio_i(PL_PHY_MDIO),.fault(control_fault));
 // Stage-only full-book capture removed from the production datapath.
 assign cap_ready=1;assign cap_empty=1;assign cap_available=0;assign cap_data=0;
 assign cap_busy=0;assign cap_pop_fault=0;assign captured_rows=0;
 g_network_frontend #(.FIFO_DEPTH(1024),.LEVEL_BANK_BITS(4)) frontend(.por_n(session_por),.clk125(clk125),.clk125_90(clk125_90),.clk200(d_clk),
  .enable(enable),.memory_initialized(memory_initialized),.external_stop(control_fault||stop_sync[1]||predict_fault||predict_control_sync[1]),
  .sink_ready(sink_ready),.session(session),.day_ns(64'd1765756800000000000),
  .start_ns(64'd1765809300000000000),.end_ns(64'd1765832100000000000),
  .measure_first(measure_first),.measure_last(measure_last),.order_base(order_base),.host_ip(host_ip),
  .instrument_id(32'd38),.publisher_id(16'd2),.status_count(8'd4),
  .status_ts({256'b0,64'd1765809000000054809,64'd1765789200000100453,64'd1765785892862405601,64'd1765785892862389018}),
  .status_allowed(8'b00001000),.finish_valid(finish_valid),.expected_last(expected_last),
  .rgmii_rx_clk(PL_PHY_RX_CLK),.rgmii_rxd(PL_PHY_RXD),.rgmii_rx_ctl(PL_PHY_RX_CTRL),
  .rgmii_tx_clk(PL_PHY_TX_CLK),.rgmii_txd(PL_PHY_TXD),.rgmii_tx_ctl(PL_PHY_TX_CTRL),
  .result_valid(udp_result_valid),.result_ready(udp_result_ready),.result_addr(udp_result_addr),.result_byte(udp_result_byte),
  .tensor_valid(tensor_valid),.tensor_ready(tensor_ready),.tensor_write(tensor_write),.tensor_fast_write(tensor_fast_write),.tensor_addr(tensor_addr),.tensor_data(tensor_data),
  .tensor_reply(tensor_reply),.tensor_fault(tensor_fault),.tensor_rdata(tensor_rdata),.book_accept_tick(book_accept_tick),
  .awwaddr(awwaddr),.awwdata(awwdata),.awwinfo(awwinfo),.awwvalid(awwvalid),.awwready(awwready),
  .araddr(araddr),.arinfo(arinfo),.arvalid(arvalid),.arready(arready),.rdata(rdata),.rinfo(rinfo),.rvalid(rvalid),.rready(rready),
  .monitor_addr(monitor_addr),.monitor_data(monitor_data),.book_valid(book_valid),.book_data(book_data),.book_identity(book_identity),
  .book_frame_valid(book_frame_valid),.book_frame(book_frame),.initialized(initialized),.fault(front_fault),.finished(finished),.error(front_error),
  .raw_committed_out(raw_committed),.eligible_count_out(eligible_count),.cycle_out(core_cycle),
  .sample_valid(),.sample_ordinal(),.sample_accept_cycle(),.sample_commit_cycle());
 wire arm_predict;wire [2047:0] slot_bases,slot_caps;wire [5:0] active_slots;wire [63:0] age_limit_ticks;wire [15:0] h50_records;wire [63:0] output_base;
 wire nreq_valid,nreq_ready,ndone_valid,ndone_ready,presult_valid,presult_ready;
 wire gp_result_valid,gp_result_ready,net_result_valid,net_result_ready;wire [383:0] gp_result_data,net_result_data;
 wire [255:0] nreq;wire [159:0] ndone_token;wire [383:0] presult;
 wire [7:0] predict_status_index;wire [63:0] predict_status;
 g_predict_control predict_control(.por_n(session_por),.c_clk(c_clk),.b_clk(clk125),
  .awaddr(reg_awaddr),.awdata(reg_awdata),.awvalid(reg_awvalid&&select_pw),.awready(new_awready),
  .araddr(reg_araddr),.arvalid(reg_arvalid&&select_pr),.arready(new_arready),.rdata(new_rdata),.rvalid(new_rvalid),.rready(reg_rready),
  .configured(predict_configured),.arm_pulse(arm_predict),.slot_bases(slot_bases),.slot_caps(slot_caps),.output_base(output_base),
  .active_slots(active_slots),.age_limit_ticks(age_limit_ticks),.h50_records(h50_records),
  .npu_valid(nreq_valid),.npu_ready(nreq_ready),.npu_request(nreq),.done_valid(ndone_valid),.done_ready(ndone_ready),.done_token(ndone_token),
  .result_valid(gp_result_valid),.result_ready(gp_result_ready),.result_data(gp_result_data),.status_data(predict_status),.status_index(predict_status_index),.datapath_fault(predict_fault),.fault(predict_control_fault));
 g_predict_path #(.ROM0("repo://hardware/rom/fixed_lane0.hex"),.ROM1("repo://hardware/rom/fixed_lane1.hex"),.ROM2("repo://hardware/rom/fixed_lane2.hex"),.ROM3("repo://hardware/rom/fixed_lane3.hex")) prediction(.por_n(session_por),.clk(clk125),.clk200(d_clk),.arm_pulse(arm_predict),
  .configured(predict_configured),.session(session),.slot_bases(slot_bases),.slot_caps(slot_caps),.output_base(output_base),
  .active_slots(active_slots),.age_limit_ticks(age_limit_ticks),.h50_records(h50_records),
  .book_valid(book_valid),.book_ready(predict_book_ready),.book_data(book_data),.book_identity(book_identity),.book_accept_tick(book_accept_tick),
  .npu_request_valid(nreq_valid),.npu_request_ready(nreq_ready),.npu_request(nreq),.npu_done_valid(ndone_valid),.npu_done_ready(ndone_ready),.npu_done_token(ndone_token),
  .result_valid(presult_valid),.result_ready(presult_ready),.result_data(presult),.status_index(predict_status_index),.status_data(predict_status),
  .b_valid(tensor_valid),.b_ready(tensor_ready),.b_write(tensor_write),.b_fast_write(tensor_fast_write),.b_addr(tensor_addr),.b_data(tensor_data),
  .b_reply(tensor_reply),.b_rdata(tensor_rdata),.memory_fault(tensor_fault_sync[1]),.fault(predict_fault));

 g_result_fanout result_fanout(.clk(clk125),.por_n(session_por),.s_valid(presult_valid),.s_ready(presult_ready),.s_data(presult),
  .gp_valid(gp_result_valid),.gp_ready(gp_result_ready),.gp_data(gp_result_data),
  .net_valid(net_result_valid),.net_ready(net_result_ready),.net_data(net_result_data));
 g_result_packet result_packet(.clk125(clk125),.clk200(d_clk),.por_n(session_por),.s_valid(net_result_valid),.s_ready(net_result_ready),.s_data(net_result_data),.session(session),
  .result_valid(udp_result_valid),.result_ready(udp_result_ready),.result_addr(udp_result_addr),.result_byte(udp_result_byte));

endmodule
