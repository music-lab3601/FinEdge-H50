`timescale 1ns/1ps
// Four lanes: ask price, ask quantity, bid price, bid quantity. Each accepted
// integer book is encoded once; output remains 40 signed codes in feature order.
module fp_fixed_record #(
 parameter ROM0="fixed_lane0.hex",ROM1="fixed_lane1.hex",
           ROM2="fixed_lane2.hex",ROM3="fixed_lane3.hex"
)(
 input wire clk,rst,s_valid,output wire s_ready,input wire [2559:0] s_book,
 input wire [255:0] s_identity,
 output reg m_valid,input wire m_ready,output reg [319:0] m_data,
 output reg [255:0] m_identity,output reg fault,output reg [31:0] error
);
 localparam IDLE=0,LAUNCH=1,COLLECT=2;
 reg [1:0] state;reg [3:0] group;reg [2559:0] book;reg [32:0] total;
 wire [3:0] ready,valid;wire [7:0] code[0:3];
 wire collect=state==COLLECT&&(&valid);
 wire [19:0] price_bad;
 generate for(genvar p=0;p<20;p=p+1)begin:check_price
  assign price_bad[p]=|s_book[p*128+32+:32];
 end endgenerate
 assign s_ready=!rst&&!fault&&state==IDLE&&(!m_valid||m_ready);
 generate for(genvar l=0;l<4;l=l+1)begin:lane
  wire [63:0] v=book[(group*4+l)*64+:64];
  wire signed [33:0] delta=$signed({1'b0,v[31:0],1'b0})-$signed({1'b0,total});
  fp_fixed_scalar #(.PRICE(!(l%2)),.ROM_FILE(l==0?ROM0:l==1?ROM1:l==2?ROM2:ROM3)) scalar(
   .clk(clk),.rst(rst),.s_valid(state==LAUNCH&&(&ready)),.s_ready(ready[l]),.s_group(group),
   .s_delta(delta),.s_total(total),.s_qty(v),.m_valid(valid[l]),.m_ready(collect),.m_code(code[l]));
 end endgenerate
 integer k;
 always @(posedge clk)begin
  if(rst)begin state<=IDLE;group<=0;book<=0;total<=0;m_valid<=0;m_data<=0;m_identity<=0;fault<=0;error<=0;end
  else if(!fault)begin
   if(m_valid&&m_ready)m_valid<=0;
   if(s_valid&&s_ready)begin
    if(s_book[0+:64]==0||s_book[128+:64]==0||(|price_bad))begin fault<=1;error<=1;end
    else begin
     book<=s_book;total<={1'b0,s_book[0+:32]}+{1'b0,s_book[128+:32]};
     m_identity<=s_identity;group<=0;state<=LAUNCH;
    end
   end
   if(state==LAUNCH&&(&ready))state<=COLLECT;
   if(collect)begin
    for(k=0;k<4;k=k+1)m_data[(group*4+k)*8+:8]<=code[k];
    if(group==9)begin state<=IDLE;m_valid<=1;end
    else begin group<=group+1'b1;state<=LAUNCH;end
   end
  end
 end
endmodule
