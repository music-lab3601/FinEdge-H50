// Single-clock BRAM queue with a registered elastic head. Capacity includes head.
module fb_event_fifo #(parameter WIDTH=576,DEPTH=256)(
 input wire clk,rst,input wire s_valid,output wire s_ready,input wire [WIDTH-1:0] s_data,
 output reg m_valid,input wire m_ready,output reg [WIDTH-1:0] m_data,
 output wire [$clog2(DEPTH):0] occupancy
);
 localparam A=$clog2(DEPTH);
 (* ram_style="block" *) reg [WIDTH-1:0] ram[0:DEPTH-1];
 reg [A-1:0] wp,rp;reg [A:0] stored;
 assign occupancy=stored+m_valid;
 assign s_ready=occupancy<DEPTH;
 wire push=s_valid&&s_ready;
 wire prefetch=(!m_valid||m_ready)&&stored!=0;
 always @(posedge clk)begin
  if(rst)begin wp<=0;rp<=0;stored<=0;m_valid<=0;m_data<=0;end
  else begin
   if(push)begin ram[wp]<=s_data;wp<=(wp==DEPTH-1)?0:wp+1'b1;end
   if(prefetch)begin m_data<=ram[rp];rp<=(rp==DEPTH-1)?0:rp+1'b1;m_valid<=1;end
   else if(m_valid&&m_ready)m_valid<=0;
   case({push,prefetch})2'b10:stored<=stored+1'b1;2'b01:stored<=stored-1'b1;default:;endcase
  end
 end
endmodule
