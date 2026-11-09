`timescale 1ns/1ps
// All-integer inverse preprocessing. Thresholds incorporate the exact frozen
// IEEE reference rounding, but NO IEEE operators exist in this datapath.
// Price: compare (2*p-ask1-bid1)*2^78 to threshold*(ask1+bid1).
// Quantity: compare uint64 to its exact first-code integer threshold.
module fp_fixed_scalar #(
 parameter integer PRICE=1,
 parameter ROM_FILE="fixed_lane0.hex"
)(
 input wire clk,rst,s_valid,output wire s_ready,input wire [3:0] s_group,
 input wire signed [33:0] s_delta,input wire [32:0] s_total,input wire [63:0] s_qty,
 output reg m_valid,input wire m_ready,output reg [7:0] m_code
);
 `include "fixed_parameters.vh"
 localparam RW=PRICE?(FP_TW+1):FP_QW;
 localparam IDLE=0,REQUEST=1,MULTIPLY=2,COMPARE=3;
 (* rom_style="block" *) reg [RW-1:0] threshold[0:2559];
 initial $readmemh(ROM_FILE,threshold);
 reg [1:0] state;
 reg [7:0] lo,hi;reg [3:0] group;
 reg [RW-1:0] threshold_q;
 wire [7:0] mid=({1'b0,lo}+{1'b0,hi})>>1;
 reg signed [FP_CW-1:0] lhs;
 reg [32:0] total;reg [63:0] qty;
 wire hit;
 generate if(PRICE)begin:price_path
  // 69x34 signed product; DSP inference, followed by a registered 112-bit compare.
  // Sign extension is explicit: the sum is unsigned33, delta is signed34.
  (* use_dsp="yes" *) reg signed [FP_TW+34-1:0] product;
  reg inclusive;
  always @(posedge clk)if(state==MULTIPLY)begin
   product<=$signed(threshold_q[FP_TW:1])*$signed({1'b0,total});
   inclusive<=threshold_q[0];
  end
  assign hit=inclusive?(lhs>=$signed(product)):(lhs>$signed(product));
 end else begin:quantity_path
  reg [FP_QW-1:0] bound;
  always @(posedge clk)if(state==MULTIPLY)bound<=threshold_q;
  assign hit=qty>=bound;
 end endgenerate
 always @(posedge clk)if(state==REQUEST&&lo!=hi)threshold_q<=threshold[{group,mid}];
 assign s_ready=!rst&&state==IDLE&&(!m_valid||m_ready);
 always @(posedge clk)begin
  if(rst)begin state<=IDLE;m_valid<=0;m_code<=0;lo<=0;hi<=255;group<=0;lhs<=0;total<=0;qty<=0;end
  else begin
   if(m_valid&&m_ready)m_valid<=0;
   if(s_valid&&s_ready)begin
    // Cast BEFORE the shift, preventing a 34-bit expression from truncating.
    lhs<=$signed({{(FP_CW-34){s_delta[33]}},s_delta})<<<FP_FRAC;
    total<=s_total;qty<=s_qty;group<=s_group;lo<=0;hi<=255;state<=REQUEST;
   end
   if(state==REQUEST)begin
    if(lo==hi)begin m_code<=lo-8'd128;m_valid<=1;state<=IDLE;end
    else state<=MULTIPLY;
   end else if(state==MULTIPLY)state<=COMPARE;
   else if(state==COMPARE)begin
    if(hit)lo<=mid+1'b1;else hi<=mid;
    state<=REQUEST;
   end
  end
 end
endmodule
