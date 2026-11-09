`timescale 1ns/1ps
// FMR2 version2,192B little endian. Snapshot in200MHz, send through real PL
// MAC/UDP. The host uses its own QPC arrival time and source schedule, so full
// network+frontend+NPU+return latency needs no PC/FPGA clock synchronization.
module g_result_packet(input wire clk125,clk200,por_n,
 input wire s_valid,output wire s_ready,input wire [383:0] s_data,input wire [63:0] session,
 output wire result_valid,input wire result_ready,input wire [7:0] result_addr,output wire [7:0] result_byte);
 wire [447:0] payload;
 b1_mailbox #(448) queue(por_n,clk125,s_valid,s_ready,{session,s_data},clk200,result_valid,result_ready,payload);
 wire [1535:0] packet={
  960'd0,
  32'd125000000,32'd0,
  payload[383:320], //NPU request through output ready ticks
  payload[319:256], //decoded event through logits ready ticks
  payload[127:0],   //logits0..2 and class/terminal reason
  payload[255:192], //raw ordinal
  payload[191:128], //window id
  payload[447:384], //session
  32'd2,32'h32524d46};
 assign result_byte=result_addr<192?packet[result_addr*8+:8]:8'd0;
endmodule
