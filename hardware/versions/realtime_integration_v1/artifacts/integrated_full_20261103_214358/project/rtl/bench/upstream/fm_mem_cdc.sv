// Single ordered DDR transaction across the independent book/DDR clocks.
module fm_mem_cdc(input wire por_n,book_clk,ddr_clk,
 input wire s_valid,output wire s_ready,input wire s_write,input wire [31:0] s_addr,input wire [511:0] s_data,
 output wire s_reply,output wire [511:0] s_rdata,
 output wire m_valid,input wire m_ready,output wire m_write,output wire [31:0] m_addr,output wire [511:0] m_data,
 input wire m_reply,input wire [511:0] m_rdata,output reg fault);
 wire [544:0] req;wire response_ready;
 b1_mailbox #(545) request(por_n,book_clk,s_valid,s_ready,{s_write,s_addr,s_data},ddr_clk,m_valid,m_ready,req);
 assign {m_write,m_addr,m_data}=req;
 b1_mailbox #(512) response(por_n,ddr_clk,m_reply,response_ready,m_rdata,book_clk,s_reply,1'b1,s_rdata);
 always @(posedge ddr_clk or negedge por_n)if(!por_n)fault<=0;else if(m_reply&&!response_ready)fault<=1;
endmodule
