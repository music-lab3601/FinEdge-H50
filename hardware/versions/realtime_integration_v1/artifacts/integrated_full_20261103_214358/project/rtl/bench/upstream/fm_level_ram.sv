// Explicit simple-dual-port BRAM. Initialization and ordinary writes share
// one address/data port; no inferred third memory port or LUTRAM fallback.
module fm_level_ram #(parameter ADDR_BITS=9)(
 input wire clk,we,input wire [ADDR_BITS-1:0] wa,input wire [128:0] wd,
 input wire re,input wire [ADDR_BITS-1:0] ra,output wire [128:0] rd
);
`ifdef SYNTHESIS
 wire [143:0] data;
 xpm_memory_sdpram #(.ADDR_WIDTH_A(ADDR_BITS),.ADDR_WIDTH_B(ADDR_BITS),
 .MEMORY_SIZE((1<<ADDR_BITS)*144),.MEMORY_PRIMITIVE("block"),.CLOCKING_MODE("common_clock"),
 .WRITE_DATA_WIDTH_A(144),.READ_DATA_WIDTH_B(144),.BYTE_WRITE_WIDTH_A(144),
 .READ_LATENCY_B(1),.READ_RESET_VALUE_B("0"),.WRITE_MODE_B("no_change"),.USE_MEM_INIT(0)) ram(
 .clka(clk),.clkb(clk),.ena(we),.wea(we),.addra(wa),.dina({15'b0,wd}),
 .enb(re),.addrb(ra),.doutb(data),.regceb(1'b1),.rstb(1'b0),.sleep(1'b0),.injectsbiterra(1'b0),.injectdbiterra(1'b0));
 assign rd=data[128:0];
`else
 reg [128:0] memory[0:(1<<ADDR_BITS)-1];reg [128:0] data;
 always @(posedge clk)begin if(we)memory[wa]<=wd;if(re)data<=memory[ra];end
 assign rd=data;
`endif
endmodule
