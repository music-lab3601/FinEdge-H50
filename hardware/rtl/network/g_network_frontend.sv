// Buildable integration boundary: real PL network + B core, no CPU book/NPU.
// G supplies clock generation, safe configuration/monitor CDC and real DDR port.
// User_Ddr is clk200; the book/DDR ordered request-response CDC is included.
module g_network_frontend #(parameter FIFO_DEPTH=256,ORDER_BITS=17,LEVEL_BITS=14,LEVEL_BANK_BITS=5)(
 input wire por_n,clk125,clk125_90,clk200,enable,memory_initialized,external_stop,sink_ready,
 input wire [63:0] session,day_ns,start_ns,end_ns,measure_first,measure_last,
 input wire [31:0] order_base,host_ip,instrument_id,input wire [15:0] publisher_id,
 input wire [7:0] status_count,input wire [511:0] status_ts,input wire [7:0] status_allowed,
 input wire finish_valid,input wire [63:0] expected_last,
 input wire rgmii_rx_clk,input wire [3:0] rgmii_rxd,input wire rgmii_rx_ctl,
 output wire rgmii_tx_clk,output wire [3:0] rgmii_txd,output wire rgmii_tx_ctl,
 input wire result_valid,output wire result_ready,output wire [7:0] result_addr,input wire [7:0] result_byte,
 input wire tensor_valid,tensor_write,tensor_fast_write,output wire tensor_ready,input wire [31:0] tensor_addr,input wire [511:0] tensor_data,
 output wire tensor_reply,tensor_fault,output wire [511:0] tensor_rdata,output wire [63:0] book_accept_tick,
 output wire [31:0] awwaddr,output wire [511:0] awwdata,output wire [71:0] awwinfo,output wire awwvalid,input wire awwready,
 output wire [31:0] araddr,arinfo,output wire arvalid,input wire arready,
 input wire [511:0] rdata,input wire [31:0] rinfo,input wire rvalid,output wire rready,
 input wire [11:0] monitor_addr,output wire [63:0] monitor_data,
 output wire book_valid,output wire [2559:0] book_data,output wire [255:0] book_identity,
 output wire book_frame_valid,output wire [3327:0] book_frame,
 output wire initialized,fault,finished,output wire [31:0] error,
 output wire [63:0] raw_committed_out,eligible_count_out,cycle_out,
 output wire sample_valid,output wire [63:0] sample_ordinal,sample_accept_cycle,sample_commit_cycle
);
 wire up125,up200;
 b1_reset_sync reset125(clk125,por_n,up125),reset200(clk200,por_n,up200);
 (* ASYNC_REG="TRUE" *) reg [1:0] enable200,fault125;
 wire net_fault,mfault,cdc_fault;
 always @(posedge clk200 or negedge up200)if(!up200)enable200<=0;else enable200<={enable200[0],enable};
 always @(posedge clk125 or negedge up125)if(!up125)fault125<=0;else fault125<={fault125[0],net_fault||mfault||cdc_fault};
 wire nv,nr,nl,cv,cr;wire [63:0] nd;wire [7:0] nk;wire [72:0] cd;
 wire core_initialized;
 wire host_cfg_ready125,host_cfg_valid200;wire [31:0] host_cfg_payload200;
 reg host_cfg_sent125,host_cfg_done125,host_cfg_loaded200;reg [31:0] host_ip200;
 assign initialized=core_initialized&&host_cfg_done125;
 always @(posedge clk125 or negedge up125)begin
  if(!up125)begin host_cfg_sent125<=0;host_cfg_done125<=0;end
  else begin
   if(enable&&!host_cfg_sent125&&host_cfg_ready125)host_cfg_sent125<=1;
   if(host_cfg_sent125&&host_cfg_ready125)host_cfg_done125<=1;
  end
 end
 b1_mailbox #(32) host_config_cdc(por_n,clk125,enable&&!host_cfg_sent125,host_cfg_ready125,host_ip,
  clk200,host_cfg_valid200,1'b1,host_cfg_payload200);
 always @(posedge clk200 or negedge up200)begin
  if(!up200)begin host_cfg_loaded200<=0;host_ip200<=0;end
  else if(host_cfg_valid200)begin host_cfg_loaded200<=1;host_ip200<=host_cfg_payload200;end
 end
 wire [1:0] speed;
 reg [15:0] report_counter;reg report_pending;wire report_ready,report_valid;
 reg [511:0] report_hold;wire [511:0] report_payload;
 always @(posedge clk125 or negedge up125)begin
  if(!up125)begin report_counter<=0;report_pending<=0;report_hold<=0;end
  else begin
   report_counter<=report_counter+1;
   if(report_pending&&report_ready)report_pending<=0;
   if(report_counter==0&&!report_pending)begin
    report_hold<={session,cycle_out,{32'b0,error},64'b0,64'b0,eligible_count_out,raw_committed_out,{29'b0,enable,initialized,fault,32'h31414d46}};
    report_pending<=1;
   end
  end
 end
 b1_mailbox #(512) ack_cdc(por_n,clk125,report_pending,report_ready,report_hold,clk200,report_valid,1'b1,report_payload);
 reg [511:0] ack_latest;
 always @(posedge clk200 or negedge up200)if(!up200)ack_latest<=0;else if(report_valid)ack_latest<=report_payload;

 fm_network network(.clk125(clk125),.clk125_90(clk125_90),.rst125(!up125),.clk200(clk200),.rst200(!up200),
  .enable(enable200[1]&&host_cfg_loaded200),.host_ip(host_ip200),.ack_payload(ack_latest),
  .result_valid(result_valid),.result_ready(result_ready),.result_addr(result_addr),.result_byte(result_byte),
  .rgmii_rx_clk(rgmii_rx_clk),.rgmii_rxd(rgmii_rxd),.rgmii_rx_ctl(rgmii_rx_ctl),
  .rgmii_tx_clk(rgmii_tx_clk),.rgmii_txd(rgmii_txd),.rgmii_tx_ctl(rgmii_tx_ctl),
  .m_valid(nv),.m_ready(nr),.m_data(nd),.m_keep(nk),.m_last(nl),.fault(net_fault),.speed(speed));
 // One-entry bundled mailbox is deliberately conservative in capacity, not a
 // claim of line rate. MAC/packet FIFOs absorb only bounded bursts.
 b1_mailbox #(73) payload_cdc(por_n,clk200,nv,nr,{nl,nk,nd},clk125,cv,cr,cd);
 wire mv,mr,mw,mreply;wire [31:0] ma,me;wire [511:0] md,mrd;
 wire dv,dr,dw,dreply;wire [31:0] da;wire [511:0] dd,drd;
 g_bench_core #(.ORDER_BITS(ORDER_BITS),.LEVEL_BITS(LEVEL_BITS),.LEVEL_BANK_BITS(LEVEL_BANK_BITS),.FIFO_DEPTH(FIFO_DEPTH)) core(
  .clk(clk125),.rst(!up125),.enable(enable&&host_cfg_done125),.memory_initialized(memory_initialized),.external_fault(fault125[1]||external_stop),
  .session(session),.day_ns(day_ns),.start_ns(start_ns),.end_ns(end_ns),.order_base(order_base),
  .instrument_id(instrument_id),.publisher_id(publisher_id),.status_count(status_count),.status_ts(status_ts),.status_allowed(status_allowed),
  .measure_first(measure_first),.measure_last(measure_last),.finish_valid(finish_valid),.expected_last(expected_last),
  .s_valid(cv),.s_ready(cr),.s_data(cd[63:0]),.s_keep(cd[71:64]),.s_last(cd[72]),
  .book_valid(book_valid),.book_ready(sink_ready),.book_data(book_data),.book_identity(book_identity),
  .mem_valid(mv),.mem_ready(mr),.mem_write(mw),.mem_addr(ma),.mem_wdata(md),.mem_rsp_valid(mreply),.mem_rdata(mrd),.mem_error(fault125[1]),
  .monitor_addr(monitor_addr),.monitor_data(monitor_data),.initialized(core_initialized),.fault(fault),.error(error),.finished(finished),
  .sample_valid(sample_valid),.sample_ordinal(sample_ordinal),.sample_accept_cycle(sample_accept_cycle),.sample_commit_cycle(sample_commit_cycle),
  .raw_committed_out(raw_committed_out),.eligible_count_out(eligible_count_out),.cycle_out(cycle_out),.book_accept_tick(book_accept_tick),
  .sample_mutated(),.sample_eligible(),.book_frame_valid(book_frame_valid),.book_frame(book_frame));
 fm_mem_cdc memory_cdc(.por_n(por_n),.book_clk(clk125),.ddr_clk(clk200),
  .s_valid(mv),.s_ready(mr),.s_write(mw),.s_addr(ma),.s_data(md),.s_reply(mreply),.s_rdata(mrd),
  .m_valid(dv),.m_ready(dr),.m_write(dw),.m_addr(da),.m_data(dd),.m_reply(dreply),.m_rdata(drd),.fault(cdc_fault));
 assign tensor_fault=mfault;
 g_dual_vendor_memory memory(.clk(clk200),.rst(!up200),.a_valid(dv),.a_ready(dr),.a_write(dw),.a_addr(da),.a_data(dd),
  .a_reply(dreply),.a_rdata(drd),.b_valid(tensor_valid),.b_ready(tensor_ready),.b_write(tensor_write),.b_fast_write(tensor_fast_write),.b_addr(tensor_addr),.b_data(tensor_data),
  .b_reply(tensor_reply),.b_rdata(tensor_rdata),.fault(mfault),.error(me),
  .awwaddr(awwaddr),.awwdata(awwdata),.awwinfo(awwinfo),.awwvalid(awwvalid),.awwready(awwready),
  .araddr(araddr),.arinfo(arinfo),.arvalid(arvalid),.arready(arready),.rdata(rdata),.rinfo(rinfo),.rvalid(rvalid),.rready(rready));
endmodule
