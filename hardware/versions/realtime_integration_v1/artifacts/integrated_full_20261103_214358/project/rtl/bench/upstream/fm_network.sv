// Pinned verilog-ethernet MAC + ARP + IPv4/UDP. No ARM network stack.
module fm_network #(
 parameter [47:0] LOCAL_MAC=48'h020000000049,
 parameter [31:0] LOCAL_IP=32'hc0a87e32,
 parameter [15:0] LOCAL_PORT=16'd20001
)(input wire clk125,clk125_90,rst125,clk200,rst200,
 input wire enable,input wire [31:0] host_ip,input wire [511:0] ack_payload,
 input wire result_valid,output wire result_ready,output wire [7:0] result_addr,input wire [7:0] result_byte,
 input wire rgmii_rx_clk,input wire [3:0] rgmii_rxd,input wire rgmii_rx_ctl,
 output wire rgmii_tx_clk,output wire [3:0] rgmii_txd,output wire rgmii_tx_ctl,
 output wire m_valid,input wire m_ready,output wire [63:0] m_data,output wire [7:0] m_keep,output wire m_last,
 output reg fault,output wire [1:0] speed);
 localparam DATA_WIDTH=8,KEEP_WIDTH=1;
 wire [7:0] mac_rx_data,mac_tx_data;
 wire mac_rx_valid,mac_rx_ready,mac_rx_last,mac_rx_user,mac_tx_valid,mac_tx_ready,mac_tx_last,mac_tx_user;
 wire rx_overflow,rx_bad,rx_fcs,rx_drop;

wire  rx_eth_hdr_valid;
wire [47:0] rx_eth_dest_mac;
wire [47:0] rx_eth_src_mac;
wire [15:0] rx_eth_type;
wire [DATA_WIDTH-1:0] rx_eth_payload_axis_tdata;
wire [KEEP_WIDTH-1:0] rx_eth_payload_axis_tkeep;
wire  rx_eth_payload_axis_tvalid;
wire  rx_eth_payload_axis_tlast;
wire  rx_eth_payload_axis_tuser;
wire rx_eth_hdr_ready,rx_eth_payload_axis_tready;
wire  tx_eth_hdr_valid;
wire [47:0] tx_eth_dest_mac;
wire [47:0] tx_eth_src_mac;
wire [15:0] tx_eth_type;
wire [DATA_WIDTH-1:0] tx_eth_payload_axis_tdata;
wire [KEEP_WIDTH-1:0] tx_eth_payload_axis_tkeep;
wire  tx_eth_payload_axis_tvalid;
wire  tx_eth_payload_axis_tlast;
wire  tx_eth_payload_axis_tuser;
wire tx_eth_hdr_ready,tx_eth_payload_axis_tready;
eth_mac_1g_rgmii_fifo #( .TARGET("XILINX"),.IODDR_STYLE("IODDR"),.CLOCK_INPUT_STYLE("BUFR"),.AXIS_DATA_WIDTH(8),.RX_FIFO_DEPTH(16384)) mac(
 .gtx_clk(clk125),
 .gtx_clk90(clk125_90),
 .gtx_rst(rst125),
 .logic_clk(clk200),
 .logic_rst(rst200),
 .tx_axis_tdata(mac_tx_data),
 .tx_axis_tkeep(1'b1),
 .tx_axis_tvalid(mac_tx_valid),
 .tx_axis_tready(mac_tx_ready),
 .tx_axis_tlast(mac_tx_last),
 .tx_axis_tuser(mac_tx_user),
 .rx_axis_tdata(mac_rx_data),
 .rx_axis_tkeep(),
 .rx_axis_tvalid(mac_rx_valid),
 .rx_axis_tready(mac_rx_ready),
 .rx_axis_tlast(mac_rx_last),
 .rx_axis_tuser(mac_rx_user),
 .rgmii_rx_clk(rgmii_rx_clk),
 .rgmii_rxd(rgmii_rxd),
 .rgmii_rx_ctl(rgmii_rx_ctl),
 .rgmii_tx_clk(rgmii_tx_clk),
 .rgmii_txd(rgmii_txd),
 .rgmii_tx_ctl(rgmii_tx_ctl),
 .tx_error_underflow(),
 .tx_fifo_overflow(),
 .tx_fifo_bad_frame(),
 .tx_fifo_good_frame(),
 .rx_error_bad_frame(rx_bad),
 .rx_error_bad_fcs(rx_fcs),
 .rx_fifo_overflow(rx_overflow),
 .rx_fifo_bad_frame(rx_drop),
 .rx_fifo_good_frame(),
 .speed(speed),
 .cfg_ifg(8'd12),
 .cfg_tx_enable(1'b1),
 .cfg_rx_enable(1'b1)
);

eth_axis_rx ethernet_rx(
 .clk(clk200),
 .rst(rst200),
 .s_axis_tdata(mac_rx_data),
 .s_axis_tkeep('0),
 .s_axis_tvalid(mac_rx_valid),
 .s_axis_tready(mac_rx_ready),
 .s_axis_tlast(mac_rx_last),
 .s_axis_tuser(mac_rx_user),
 .m_eth_hdr_valid(rx_eth_hdr_valid),
 .m_eth_hdr_ready(rx_eth_hdr_ready),
 .m_eth_dest_mac(rx_eth_dest_mac),
 .m_eth_src_mac(rx_eth_src_mac),
 .m_eth_type(rx_eth_type),
 .m_eth_payload_axis_tdata(rx_eth_payload_axis_tdata),
 .m_eth_payload_axis_tkeep(rx_eth_payload_axis_tkeep),
 .m_eth_payload_axis_tvalid(rx_eth_payload_axis_tvalid),
 .m_eth_payload_axis_tready(rx_eth_payload_axis_tready),
 .m_eth_payload_axis_tlast(rx_eth_payload_axis_tlast),
 .m_eth_payload_axis_tuser(rx_eth_payload_axis_tuser),
 .busy(),
 .error_header_early_termination()
);

eth_axis_tx ethernet_tx(
 .clk(clk200),
 .rst(rst200),
 .s_eth_hdr_valid(tx_eth_hdr_valid),
 .s_eth_hdr_ready(tx_eth_hdr_ready),
 .s_eth_dest_mac(tx_eth_dest_mac),
 .s_eth_src_mac(tx_eth_src_mac),
 .s_eth_type(tx_eth_type),
 .s_eth_payload_axis_tdata(tx_eth_payload_axis_tdata),
 .s_eth_payload_axis_tkeep(tx_eth_payload_axis_tkeep),
 .s_eth_payload_axis_tvalid(tx_eth_payload_axis_tvalid),
 .s_eth_payload_axis_tready(tx_eth_payload_axis_tready),
 .s_eth_payload_axis_tlast(tx_eth_payload_axis_tlast),
 .s_eth_payload_axis_tuser(tx_eth_payload_axis_tuser),
 .m_axis_tdata(mac_tx_data),
 .m_axis_tkeep(),
 .m_axis_tvalid(mac_tx_valid),
 .m_axis_tready(mac_tx_ready),
 .m_axis_tlast(mac_tx_last),
 .m_axis_tuser(mac_tx_user),
 .busy()
);

wire udp_hdr_valid,udp_hdr_ready,udp_valid,udp_ready,udp_last,udp_user;
 reg [1:0] send_state;reg [17:0] ack_timer;reg ack_due;reg [511:0] tx_payload;reg [7:0] tx_index,tx_length;
 reg send_result,byte_ready;reg [7:0] tx_byte;
 wire tx_udp_hdr_ready,tx_udp_data_ready;
 assign result_addr=tx_index;
 assign result_ready=send_state==2&&send_result&&byte_ready&&tx_udp_data_ready&&tx_index==191;
 always @(posedge clk200)begin
  if(rst200)begin send_state<=0;ack_timer<=0;ack_due<=0;send_result<=0;byte_ready<=0;tx_byte<=0;end
  else begin
   ack_timer<=ack_timer+1;
   if(enable&&ack_timer==0)ack_due<=1;
   case(send_state)
    0:if(result_valid)begin send_result<=1;tx_length<=192;tx_index<=0;send_state<=1;end
      else if(ack_due)begin send_result<=0;tx_payload<=ack_payload;tx_length<=64;tx_index<=0;ack_due<=0;send_state<=1;end
    1:if(tx_udp_hdr_ready)begin send_state<=2;byte_ready<=0;end
    2:if(!byte_ready)begin tx_byte<=send_result?result_byte:tx_payload[tx_index*8+:8];byte_ready<=1;end
      else if(tx_udp_data_ready)begin byte_ready<=0;if(tx_index==tx_length-1)send_state<=0;else tx_index<=tx_index+1;end
   endcase
  end
 end
 wire [7:0] udp_data;wire [31:0] udp_ip;wire [15:0] udp_port;wire [12:0] udp_fragment;
 reg accepted,in_packet;wire allowed=udp_ip==LOCAL_IP&&udp_port==LOCAL_PORT&&udp_fragment==0;
 assign udp_hdr_ready=!in_packet;
 always @(posedge clk200)begin
  if(rst200)begin accepted<=0;in_packet<=0;end
  else begin
   if(udp_hdr_valid&&udp_hdr_ready)begin accepted<=allowed;in_packet<=1;end
   if(udp_valid&&udp_ready&&udp_last)in_packet<=0;
  end
 end
 wire frame_ready;
 assign udp_ready=in_packet?(accepted?frame_ready:1'b1):1'b0;
 wire [7:0] stored_data;wire stored_valid,stored_ready,stored_last,stored_user,frame_overflow,frame_bad;
 axis_fifo #(.DEPTH(16384),.DATA_WIDTH(8),.KEEP_ENABLE(0),.KEEP_WIDTH(1),.ID_ENABLE(0),.DEST_ENABLE(0),
 .USER_ENABLE(1),.USER_WIDTH(1),.FRAME_FIFO(1),.DROP_OVERSIZE_FRAME(1),.DROP_BAD_FRAME(1),.DROP_WHEN_FULL(0)) packet_admission(
 .clk(clk200),.rst(rst200),.s_axis_tdata(udp_data),.s_axis_tkeep(1'b1),.s_axis_tvalid(udp_valid&&in_packet&&accepted),.s_axis_tready(frame_ready),.s_axis_tlast(udp_last),.s_axis_tid(1'b0),.s_axis_tdest(1'b0),.s_axis_tuser(udp_user),
 .m_axis_tdata(stored_data),.m_axis_tvalid(stored_valid),.m_axis_tready(stored_ready),.m_axis_tlast(stored_last),.m_axis_tuser(stored_user),.pause_req(1'b0),.status_overflow(frame_overflow),.status_bad_frame(frame_bad));
 axis_adapter #(.S_DATA_WIDTH(8),.S_KEEP_ENABLE(0),.S_KEEP_WIDTH(1),.M_DATA_WIDTH(64),.M_KEEP_ENABLE(1),.M_KEEP_WIDTH(8),.ID_ENABLE(0),.DEST_ENABLE(0),.USER_ENABLE(1),.USER_WIDTH(1)) pack64(
 .clk(clk200),.rst(rst200),.s_axis_tdata(stored_data),.s_axis_tkeep(1'b1),.s_axis_tvalid(stored_valid),.s_axis_tready(stored_ready),.s_axis_tlast(stored_last),.s_axis_tid(1'b0),.s_axis_tdest(1'b0),.s_axis_tuser(stored_user),
 .m_axis_tdata(m_data),.m_axis_tkeep(m_keep),.m_axis_tvalid(m_valid),.m_axis_tready(m_ready),.m_axis_tlast(m_last));
 always @(posedge clk200)if(rst200)fault<=0;else if(enable&&(rx_overflow||rx_bad||rx_fcs||rx_drop||frame_overflow||frame_bad))fault<=1;

udp_complete udp(
 .clk(clk200),
 .rst(rst200),
 .s_eth_hdr_valid(rx_eth_hdr_valid),
 .s_eth_hdr_ready(rx_eth_hdr_ready),
 .s_eth_dest_mac(rx_eth_dest_mac),
 .s_eth_src_mac(rx_eth_src_mac),
 .s_eth_type(rx_eth_type),
 .s_eth_payload_axis_tdata(rx_eth_payload_axis_tdata),
 .s_eth_payload_axis_tvalid(rx_eth_payload_axis_tvalid),
 .s_eth_payload_axis_tready(rx_eth_payload_axis_tready),
 .s_eth_payload_axis_tlast(rx_eth_payload_axis_tlast),
 .s_eth_payload_axis_tuser(rx_eth_payload_axis_tuser),
 .m_eth_hdr_valid(tx_eth_hdr_valid),
 .m_eth_hdr_ready(tx_eth_hdr_ready),
 .m_eth_dest_mac(tx_eth_dest_mac),
 .m_eth_src_mac(tx_eth_src_mac),
 .m_eth_type(tx_eth_type),
 .m_eth_payload_axis_tdata(tx_eth_payload_axis_tdata),
 .m_eth_payload_axis_tvalid(tx_eth_payload_axis_tvalid),
 .m_eth_payload_axis_tready(tx_eth_payload_axis_tready),
 .m_eth_payload_axis_tlast(tx_eth_payload_axis_tlast),
 .m_eth_payload_axis_tuser(tx_eth_payload_axis_tuser),
 .s_ip_hdr_valid('0),
 .s_ip_hdr_ready(),
 .s_ip_dscp('0),
 .s_ip_ecn('0),
 .s_ip_length('0),
 .s_ip_ttl('0),
 .s_ip_protocol('0),
 .s_ip_source_ip('0),
 .s_ip_dest_ip('0),
 .s_ip_payload_axis_tdata('0),
 .s_ip_payload_axis_tvalid('0),
 .s_ip_payload_axis_tready(),
 .s_ip_payload_axis_tlast('0),
 .s_ip_payload_axis_tuser('0),
 .m_ip_hdr_valid(),
 .m_ip_hdr_ready(1'b1),
 .m_ip_eth_dest_mac(),
 .m_ip_eth_src_mac(),
 .m_ip_eth_type(),
 .m_ip_version(),
 .m_ip_ihl(),
 .m_ip_dscp(),
 .m_ip_ecn(),
 .m_ip_length(),
 .m_ip_identification(),
 .m_ip_flags(),
 .m_ip_fragment_offset(),
 .m_ip_ttl(),
 .m_ip_protocol(),
 .m_ip_header_checksum(),
 .m_ip_source_ip(),
 .m_ip_dest_ip(),
 .m_ip_payload_axis_tdata(),
 .m_ip_payload_axis_tvalid(),
 .m_ip_payload_axis_tready(1'b1),
 .m_ip_payload_axis_tlast(),
 .m_ip_payload_axis_tuser(),
 .s_udp_hdr_valid(send_state==1),
 .s_udp_hdr_ready(tx_udp_hdr_ready),
 .s_udp_ip_dscp('0),
 .s_udp_ip_ecn('0),
 .s_udp_ip_ttl(8'd64),
 .s_udp_ip_source_ip(LOCAL_IP),
 .s_udp_ip_dest_ip(host_ip),
 .s_udp_source_port(LOCAL_PORT),
 .s_udp_dest_port(16'd20002),
 .s_udp_length({8'b0,tx_length}+16'd8),
 .s_udp_checksum(16'd0),
 .s_udp_payload_axis_tdata(tx_byte),
 .s_udp_payload_axis_tvalid(send_state==2&&byte_ready),
 .s_udp_payload_axis_tready(tx_udp_data_ready),
 .s_udp_payload_axis_tlast(tx_index==tx_length-1),
 .s_udp_payload_axis_tuser(1'b0),
 .m_udp_hdr_valid(udp_hdr_valid),
 .m_udp_hdr_ready(udp_hdr_ready),
 .m_udp_eth_dest_mac(),
 .m_udp_eth_src_mac(),
 .m_udp_eth_type(),
 .m_udp_ip_version(),
 .m_udp_ip_ihl(),
 .m_udp_ip_dscp(),
 .m_udp_ip_ecn(),
 .m_udp_ip_length(),
 .m_udp_ip_identification(),
 .m_udp_ip_flags(),
 .m_udp_ip_fragment_offset(udp_fragment),
 .m_udp_ip_ttl(),
 .m_udp_ip_protocol(),
 .m_udp_ip_header_checksum(),
 .m_udp_ip_source_ip(),
 .m_udp_ip_dest_ip(udp_ip),
 .m_udp_source_port(),
 .m_udp_dest_port(udp_port),
 .m_udp_length(),
 .m_udp_checksum(),
 .m_udp_payload_axis_tdata(udp_data),
 .m_udp_payload_axis_tvalid(udp_valid),
 .m_udp_payload_axis_tready(udp_ready),
 .m_udp_payload_axis_tlast(udp_last),
 .m_udp_payload_axis_tuser(udp_user),
 .ip_rx_busy(),
 .ip_tx_busy(),
 .udp_rx_busy(),
 .udp_tx_busy(),
 .ip_rx_error_header_early_termination(),
 .ip_rx_error_payload_early_termination(),
 .ip_rx_error_invalid_header(),
 .ip_rx_error_invalid_checksum(),
 .ip_tx_error_payload_early_termination(),
 .ip_tx_error_arp_failed(),
 .udp_rx_error_header_early_termination(),
 .udp_rx_error_payload_early_termination(),
 .udp_tx_error_payload_early_termination(),
 .local_mac(LOCAL_MAC),
 .local_ip(LOCAL_IP),
 .gateway_ip(host_ip),
 .subnet_mask(32'hffffff00),
 .clear_arp_cache(1'b0)
);

endmodule