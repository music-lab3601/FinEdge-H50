// One tagged transaction at the vendor port. Orders and window writes share
// the port fairly; order commits are read-compared before subsequent lookups.
module fm_mem_arbiter(
 input wire clk,rst,
 input wire a_valid,output wire a_ready,input wire a_write,input wire [31:0] a_addr,input wire [511:0] a_data,
 output reg a_reply,output reg [511:0] a_rdata,
 input wire b_valid,output wire b_ready,input wire b_verify,input wire [31:0] b_addr,input wire [511:0] b_data,
 output reg b_reply,output wire fault,output wire [31:0] error,
 output wire [31:0] awwaddr,output wire [511:0] awwdata,output wire [71:0] awwinfo,output wire awwvalid,input wire awwready,
 output wire [31:0] araddr,arinfo,output wire arvalid,input wire arready,
 input wire [511:0] rdata,input wire [31:0] rinfo,input wire rvalid,output wire rready
);
 reg [1:0] state;reg last_grant,selected;reg [63:0] token;reg [639:0] request;
 wire choose_b=b_valid&&(!a_valid||!last_grant);wire pr,pv;wire [639:0] response;wire [127:0] fault_data;
 wire port_fault;reg protocol_fault;
 assign fault=port_fault||protocol_fault;assign error=port_fault?fault_data[31:0]:protocol_fault?32'h100:0;
 assign a_ready=!rst&&!fault&&state==0&&!choose_b;assign b_ready=!rst&&!fault&&state==0&&choose_b;
 shared_input_beat_port_v1 #(.TIMEOUT(262144)) port_worker(
  .clk(clk),.por_n(!rst),.fault_event(1'b0),.req_valid(state==1),.req_ready(pr),.req_data(request),
  .rsp_valid(pv),.rsp_ready(1'b1),.rsp_data(response),.fault_valid(port_fault),.fault_ready(1'b0),.fault_data(fault_data),
  .awwaddr(awwaddr),.awwdata(awwdata),.awwinfo(awwinfo),.awwvalid(awwvalid),.awwready(awwready),
  .araddr(araddr),.arinfo(arinfo),.arvalid(arvalid),.arready(arready),.rdata(rdata),.rinfo(rinfo),.rvalid(rvalid),.rready(rready));
 always @(posedge clk)begin
  if(rst)begin state<=0;last_grant<=0;token<=1;a_reply<=0;b_reply<=0;protocol_fault<=0;end
  else begin
   a_reply<=0;b_reply<=0;
   if(!fault)case(state)
    0:if(a_valid||b_valid)begin selected<=choose_b;last_grant<=choose_b;
      request<={token,choose_b?(b_verify?32'd2:32'd1):(a_write?32'd2:32'd0),choose_b?b_addr:a_addr,choose_b?b_data:a_data};state<=1;
     end
    1:if(pr)state<=2;
    2:if(pv)begin
      if(response[639:576]!=token||response[575:544]!=request[575:544]||response[543:512]!=0)protocol_fault<=1;
      else begin if(selected)b_reply<=1;else begin a_reply<=1;a_rdata<=response[511:0];end token<=token+1;state<=0;end
     end
   endcase
  end
 end
endmodule
