// GP configuration + per-register atomic monitor mailbox. No raw data on GP.
module g_board_control #(parameter BASE=32'h400d0000)(
 input wire por_n,c_clk,b_clk,
 input wire [31:0] awaddr,awdata,input wire awvalid,output wire awready,
 input wire [31:0] araddr,input wire arvalid,output wire arready,
 output reg [31:0] rdata,output reg rvalid,input wire rready,
 output reg enable,output reg [63:0] session,measure_first,measure_last,
 output reg [31:0] order_base,host_ip,output reg capture_mode,memory_initialized,
 output wire finish_valid,output wire [63:0] expected_last,
 output wire [11:0] monitor_addr,input wire [63:0] monitor_data,
 input wire capture_empty,input wire [7:0] capture_available,
 output wire capture_read,input wire [31:0] capture_data,output wire [6:0] capture_word,
 output wire capture_pop,input wire capture_pop_fault,
 output reg phy_reset_n,phy_mdc,phy_mdio_o,phy_mdio_oe,input wire phy_mdio_i,
 output wire fault
);
 wire cu,bu;b1_reset_sync cr(c_clk,por_n,cu),br(b_clk,por_n,bu);
 reg [63:0] session_c,first_c,last_c;reg [31:0] order_c,host_c,flags_c;
 reg locked,c_fault,start_pending;wire start_ready,start_b;wire [287:0] config_b;
 b1_mailbox #(288) cfg(por_n,c_clk,start_pending,start_ready,
  {flags_c,last_c,first_c,host_c,order_c,session_c},b_clk,start_b,1'b1,config_b);
 (* ASYNC_REG="TRUE" *) reg [1:0] fault_sync;
 always @(posedge b_clk or negedge bu)begin
  if(!bu)begin enable<=0;session<=0;measure_first<=0;measure_last<=0;order_base<=0;
   host_ip<=32'hc0a87e01;capture_mode<=0;memory_initialized<=0;fault_sync<=0;end
  else begin
   fault_sync<={fault_sync[0],c_fault|capture_pop_fault};
   if(start_b)begin
    {memory_initialized,capture_mode}<={config_b[257],config_b[256]};
    {measure_last,measure_first,host_ip,order_base,session}<=config_b[255:0];enable<=1;
   end
  end
 end
 assign fault=fault_sync[1];
 reg finish_pending;wire finish_ready,finish_bit;
 b1_mailbox #(1) finish_box(por_n,c_clk,finish_pending,finish_ready,1'b1,b_clk,finish_valid,1'b1,finish_bit);
 assign expected_last=measure_last;
 reg snap_pending,snap_waiting;reg [11:0] snap_index;
 wire snap_ready,request_b;wire [11:0] request_index;
 b1_mailbox #(12) request_box(por_n,c_clk,snap_pending,snap_ready,snap_index,b_clk,request_b,1'b1,request_index);
 assign monitor_addr=request_index;
 reg reply_pending;reg [63:0] reply_hold;wire reply_ready,reply_c;wire [63:0] reply_data;
 b1_mailbox #(64) reply_box(por_n,b_clk,reply_pending,reply_ready,reply_hold,c_clk,reply_c,1'b1,reply_data);
 always @(posedge b_clk or negedge bu)begin
  if(!bu)begin reply_pending<=0;reply_hold<=0;end
  else begin
   if(reply_pending&&reply_ready)reply_pending<=0;
   if(request_b)begin reply_hold<=monitor_data;reply_pending<=1;end
  end
 end
 reg [63:0] snapshot;reg [31:0] snap_sequence;
 wire wh=awvalid&&awready&&awaddr[31:16]==BASE[31:16]&&awaddr[1:0]==0;
 wire rh=arvalid&&arready&&araddr[31:16]==BASE[31:16]&&araddr[1:0]==0;
 wire [15:0] wo=awaddr[15:0],ro=araddr[15:0];
 reg capture_read_pending;
 assign awready=cu;
 assign arready=cu&&!rvalid&&!capture_read_pending;
 assign capture_read=rh&&ro>=16'h1000&&ro<16'h1200&&!capture_empty;
 assign capture_word=araddr[8:2];
 assign capture_pop=wh&&wo==16'h094&&awdata==32'h504f5021;
 function [31:0] read_word(input [15:0] off);
  begin case(off)
   16'h000:read_word=32'h31424746; // FGB1
   16'h004:read_word=1;
   16'h008:read_word={28'b0,c_fault,capture_pop_fault,snap_waiting,locked};
   16'h090:read_word={15'b0,capture_available,8'b0,capture_empty};
   16'h0a0:read_word=snap_sequence;16'h0a4:read_word={31'b0,snap_waiting};
   16'h0a8:read_word=snapshot[31:0];16'h0ac:read_word=snapshot[63:32];
   16'h0fc:read_word={27'b0,phy_mdio_i,phy_mdio_oe,phy_mdio_o,phy_mdc,phy_reset_n};
   default:read_word=0;
  endcase end
 endfunction
 always @(posedge c_clk or negedge cu)begin
  if(!cu)begin
   session_c<=0;first_c<=0;last_c<=0;order_c<=0;host_c<=32'hc0a87e01;flags_c<=0;
   locked<=0;c_fault<=0;start_pending<=0;finish_pending<=0;snap_pending<=0;snap_waiting<=0;
   snap_index<=0;snapshot<=0;snap_sequence<=0;rvalid<=0;rdata<=0;capture_read_pending<=0;
   phy_reset_n<=0;phy_mdc<=0;phy_mdio_o<=1;phy_mdio_oe<=0;
  end else begin
   if(start_pending&&start_ready)start_pending<=0;
   if(finish_pending&&finish_ready)finish_pending<=0;
   if(snap_pending&&snap_ready)snap_pending<=0;
   if(reply_c)begin snapshot<=reply_data;snap_sequence<=snap_sequence+1;snap_waiting<=0;end
   if(rvalid&&rready)rvalid<=0;
   if(capture_read_pending)begin rdata<=capture_data;rvalid<=1;capture_read_pending<=0;end
   if(rh)begin
    if(capture_read)capture_read_pending<=1;
    else begin rdata<=read_word(ro);rvalid<=1;end
   end else if(arvalid&&arready)begin rdata<=0;rvalid<=1;end
   if(wh)begin
    case(wo)
     16'h010:if(!locked)session_c[31:0]<=awdata;else c_fault<=1;
     16'h014:if(!locked)session_c[63:32]<=awdata;else c_fault<=1;
     16'h018:if(!locked)order_c<=awdata;else c_fault<=1;
     16'h01c:if(!locked)host_c<=awdata;else c_fault<=1;
     16'h020:if(!locked)first_c[31:0]<=awdata;else c_fault<=1;
     16'h024:if(!locked)first_c[63:32]<=awdata;else c_fault<=1;
     16'h028:if(!locked)last_c[31:0]<=awdata;else c_fault<=1;
     16'h02c:if(!locked)last_c[63:32]<=awdata;else c_fault<=1;
     16'h030:if(!locked)flags_c<=awdata;else c_fault<=1;
     16'h080:if(!locked&&start_ready&&awdata==32'h53544152&&session_c!=0&&
                first_c!=0&&last_c>=first_c&&flags_c[1]&&order_c[5:0]==0)begin locked<=1;start_pending<=1;end
                else c_fault<=1;
     16'h084:if(locked&&!finish_pending&&finish_ready&&awdata==32'h454e4421)finish_pending<=1;else c_fault<=1;
     16'h088:if(!snap_waiting&&snap_ready&&!snap_pending)begin snap_index<=awdata[11:0];snap_pending<=1;snap_waiting<=1;end else c_fault<=1;
     16'h094:if(awdata!=32'h504f5021||capture_empty)c_fault<=1;
     16'h0fc:{phy_mdio_oe,phy_mdio_o,phy_mdc,phy_reset_n}<=awdata[3:0];
     default:c_fault<=1;
    endcase
   end
  end
 end
endmodule
