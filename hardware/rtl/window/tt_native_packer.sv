`timescale 1ns/1ps
// One-pass100-row reader. Three channel blocks are produced per four rows;
// DDR-side RAM reorders their75 beats into sequential NCHWc16 addresses.
module tt_native_packer(
 input wire clk,rst,input wire start_valid,output wire start_ready,input wire start_context,
 output wire read_enable,output wire read_context,output wire [6:0] read_row,
 input wire [319:0] read_data,
 output wire m_valid,input wire m_ready,output wire [511:0] m_data,output wire m_last,
 output wire [6:0] m_index,
 output reg [63:0] windows,read_cycles,blocked_cycles
);
 reg active,context_id,pending;reg [6:0] issued,pending_row;
 reg [511:0] a0,a1,a2,q0,q1,q2;reg [1:0] emit,remaining;reg [4:0] group_id;
 wire take=m_valid&&m_ready;
 wire can_read=remaining==0||(remaining==1&&m_ready)||issued[1:0]!=3;
 assign start_ready=!rst&&!active&&!pending&&remaining==0;
 assign read_enable=!rst&&active&&issued<100&&can_read;
 assign read_context=context_id;assign read_row=issued;
 assign m_valid=!rst&&remaining!=0;
 assign m_data=emit==0?q0:emit==1?q1:q2;
 assign m_index=(emit==0?7'd0:emit==1?7'd25:7'd50)+{2'd0,group_id};
 assign m_last=group_id==24&&emit==2;
 reg [511:0] c0,c1,c2;
 always @*begin
  c0=a0;c1=a1;c2=a2;
  c0[pending_row[1:0]*128+:128]=read_data[127:0];
  c1[pending_row[1:0]*128+:128]=read_data[255:128];
  c2[pending_row[1:0]*128+:128]={64'd0,read_data[319:256]};
 end
 always @(posedge clk)begin
  if(rst)begin active<=0;context_id<=0;pending<=0;issued<=0;pending_row<=0;a0<=0;a1<=0;a2<=0;
   q0<=0;q1<=0;q2<=0;emit<=0;remaining<=0;group_id<=0;windows<=0;read_cycles<=0;blocked_cycles<=0;end
  else begin
   if(start_valid&&start_ready)begin active<=1;context_id<=start_context;issued<=0;end
   if(m_valid&&!m_ready)blocked_cycles<=blocked_cycles+1;
   if(take)begin
    remaining<=remaining-1'b1;emit<=emit+1'b1;
    if(m_last)begin active<=0;windows<=windows+1;end
   end
   pending<=read_enable;
   if(read_enable)begin pending_row<=issued;issued<=issued+1;read_cycles<=read_cycles+1;end
   if(pending)begin
    if(pending_row[1:0]==3)begin
     // The producer has reserved75 FIFO positions; payload cannot be stalled
     // long enough to overwrite this three-beat output group.
     // synthesis translate_off
     if(remaining!=0&&!(remaining==1&&m_ready))$error("window FIFO credit violated");
     // synthesis translate_on
     q0<=c0;q1<=c1;q2<=c2;remaining<=3;emit<=0;group_id<=pending_row[6:2];a0<=0;a1<=0;a2<=0;
    end else begin a0<=c0;a1<=c1;a2<=c2;end
   end
  end
 end
endmodule
