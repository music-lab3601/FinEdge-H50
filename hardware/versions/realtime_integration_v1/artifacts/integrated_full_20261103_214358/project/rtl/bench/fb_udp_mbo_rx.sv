// FME1 UDP payload: 32-byte little-endian header, then 1..24 untouched DBN MBOs.
// Header: magic,version16,count16,session64,packet_seq64,first_ordinal64.
// Input is only a complete, checksum/FCS-admitted UDP packet from the MAC stack.
module fb_udp_mbo_rx(
 input wire clk,rst,enable,input wire [63:0] session,
 input wire s_valid,output wire s_ready,input wire [63:0] s_data,
 input wire [7:0] s_keep,input wire s_last,
 output reg m_valid,input wire m_ready,output reg [447:0] m_data,output reg [63:0] m_ordinal,
 output wire packet_idle,output reg fault,output reg [31:0] error,output reg [63:0] packets,records
);
 reg [1:0] header;reg [2:0] word_index;reg [15:0] count,left;
 reg [63:0] next_packet,next_ordinal;reg [447:0] record;
 assign packet_idle=left==0&&header==0&&!m_valid;
 assign s_ready=enable&&!fault&&(!m_valid||m_ready);
 task fail(input [31:0] why);begin fault<=1;error<=why;m_valid<=0;end endtask
 always @(posedge clk)begin
  if(rst)begin header<=0;word_index<=0;left<=0;count<=0;next_packet<=1;next_ordinal<=1;m_valid<=0;fault<=0;error<=0;packets<=0;records<=0;end
  else begin
   if(m_valid&&m_ready)m_valid<=0;
   if(s_valid&&s_ready)begin
    if(s_keep!=8'hff)fail(1);
    else if(left==0)begin
     if(s_last)fail(2);
     else case(header)
      0:if(s_data[31:0]!=32'h31454d46||s_data[47:32]!=1||s_data[63:48]==0||s_data[63:48]>24)fail(3);
        else begin count<=s_data[63:48];header<=1;end
      1:if(s_data!=session)fail(4);else header<=2;
      2:if(s_data!=next_packet)fail(5);else header<=3;
      3:if(s_data!=next_ordinal)fail(6);else begin header<=0;left<=count;word_index<=0;end
     endcase
    end else begin
     record[word_index*64+:64]<=s_data;
     if(s_last!=(left==1&&word_index==6))fail(7);
     else if(word_index==6)begin
      m_data<={s_data,record[383:0]};m_ordinal<=next_ordinal;m_valid<=1;
      next_ordinal<=next_ordinal+1;records<=records+1;word_index<=0;left<=left-1;
      if(left==1)begin next_packet<=next_packet+1;packets<=packets+1;end
     end else word_index<=word_index+1;
    end
   end
  end
 end
endmodule
