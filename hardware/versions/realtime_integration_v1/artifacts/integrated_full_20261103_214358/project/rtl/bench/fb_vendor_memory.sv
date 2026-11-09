// The actual vendor port adapter retained from the current workspace.
// Writes are read-compared by the snapshotted beat-port worker before replying.
module fb_vendor_memory(
 input wire clk,rst,
 input wire mem_valid,output wire mem_ready,input wire mem_write,
 input wire [31:0] mem_addr,input wire [511:0] mem_wdata,
 output wire mem_reply,output wire [511:0] mem_rdata,output wire fault,output wire [31:0] error,
 output wire [31:0] awwaddr,output wire [511:0] awwdata,output wire [71:0] awwinfo,
 output wire awwvalid,input wire awwready,
 output wire [31:0] araddr,arinfo,output wire arvalid,input wire arready,
 input wire [511:0] rdata,input wire [31:0] rinfo,input wire rvalid,output wire rready
);
 fm_mem_arbiter adapter(.clk(clk),.rst(rst),
  .a_valid(mem_valid),.a_ready(mem_ready),.a_write(mem_write),.a_addr(mem_addr),.a_data(mem_wdata),.a_reply(mem_reply),.a_rdata(mem_rdata),
  .b_valid(1'b0),.b_ready(),.b_verify(1'b0),.b_addr(32'b0),.b_data(512'b0),.b_reply(),.fault(fault),.error(error),
  .awwaddr(awwaddr),.awwdata(awwdata),.awwinfo(awwinfo),.awwvalid(awwvalid),.awwready(awwready),
  .araddr(araddr),.arinfo(arinfo),.arvalid(arvalid),.arready(arready),.rdata(rdata),.rinfo(rinfo),.rvalid(rvalid),.rready(rready));
endmodule
