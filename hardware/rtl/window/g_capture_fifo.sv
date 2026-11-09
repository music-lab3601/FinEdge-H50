// Functional capture only: 128 x 512-byte rows in dual-clock 32-bit BRAM.
// Publish a row only after all 104 payload and 24 padding words are written.
module g_capture_fifo #(parameter ADDR_BITS=7)(
 input wire por_n,w_clk,s_valid,output wire s_ready,input wire [3327:0] s_frame,
 output wire writer_busy,output reg [63:0] captured_rows,
 input wire r_clk,read_enable,input wire [6:0] read_word,output reg [31:0] read_data,
 input wire pop,output wire empty,output wire [ADDR_BITS:0] available,output reg pop_fault
);
 localparam ROW_WORD_BITS=7;
 (* ram_style="block" *) reg [31:0] ram[0:(1<<(ADDR_BITS+ROW_WORD_BITS))-1];
 reg [ADDR_BITS:0] wbin,wgray,rbin,rgray;
 (* ASYNC_REG="TRUE" *) reg [ADDR_BITS:0] rg1,rg2,wg1,wg2;
 wire w_up,r_up;
 b1_reset_sync reset_w(w_clk,por_n,w_up),reset_r(r_clk,por_n,r_up);
 wire full=wgray=={~rg2[ADDR_BITS:ADDR_BITS-1],rg2[ADDR_BITS-2:0]};
 reg writing;reg [6:0] word_index;reg [3327:0] frame;
 wire [ADDR_BITS:0] next_w=wbin+1'b1,next_r=rbin+1'b1;
 assign writer_busy=writing;
 assign s_ready=w_up&&!writing&&!full;
 assign empty=rgray==wg2;
 function [ADDR_BITS:0] gray_to_binary(input [ADDR_BITS:0] g);
  integer k;begin gray_to_binary[ADDR_BITS]=g[ADDR_BITS];
   for(k=ADDR_BITS-1;k>=0;k=k-1)gray_to_binary[k]=gray_to_binary[k+1]^g[k];
  end
 endfunction
 assign available=gray_to_binary(wg2)-rbin;
 always @(posedge w_clk or negedge w_up)begin
  if(!w_up)begin wbin<=0;wgray<=0;rg1<=0;rg2<=0;writing<=0;word_index<=0;captured_rows<=0;frame<=0;end
  else begin
   rg1<=rgray;rg2<=rg1;
   if(s_valid&&s_ready)begin frame<=s_frame;word_index<=0;writing<=1;end
   if(writing)begin
    frame<=frame>>32;
    if(word_index==127)begin writing<=0;wbin<=next_w;wgray<=(next_w>>1)^next_w;captured_rows<=captured_rows+1;end
    else word_index<=word_index+1;
   end
  end
 end
 // BRAM write process has no asynchronous reset in its sensitivity list.
 always @(posedge w_clk)if(w_up&&writing)
  ram[{wbin[ADDR_BITS-1:0],word_index}]<=word_index<104?frame[31:0]:0;
 always @(posedge r_clk or negedge r_up)begin
  if(!r_up)begin rbin<=0;rgray<=0;wg1<=0;wg2<=0;pop_fault<=0;end
  else begin
   wg1<=wgray;wg2<=wg1;
   if(pop)begin
    if(empty)pop_fault<=1;
    else begin rbin<=next_r;rgray<=(next_r>>1)^next_r;end
   end
  end
 end
 always @(posedge r_clk)if(read_enable&&!empty)read_data<=ram[{rbin[ADDR_BITS-1:0],read_word}];
endmodule
