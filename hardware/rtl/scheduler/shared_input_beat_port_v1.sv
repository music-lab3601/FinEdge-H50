// One operation at a time. op1 acknowledges acceptance; op2 read-compares the final metadata write.
// This proves observed contents, not the undocumented vendor write-drain contract.
module shared_input_beat_port_v1 #(parameter TIMEOUT=4096)(
 input wire clk,por_n,fault_event,
 input wire req_valid,output wire req_ready,input wire [639:0] req_data,
 output reg rsp_valid,input wire rsp_ready,output reg [639:0] rsp_data,
 output reg fault_valid,input wire fault_ready,output reg [127:0] fault_data,
 output reg [31:0] awwaddr,output reg [511:0] awwdata,output wire [71:0] awwinfo,
 output reg awwvalid,input wire awwready,
 output reg [31:0] araddr,output wire [31:0] arinfo,output reg arvalid,input wire arready,
 input wire [511:0] rdata,input wire [31:0] rinfo,input wire rvalid,output wire rready
);
 reg busy,waiting_r,quarantined,notify,reply_generated,verifying;
 reg [31:0] verify_cycles;
 reg [63:0] token;
 reg [31:0] operation,error,wait_cycles;
 reg [511:0] response_payload;
 assign req_ready=!quarantined && !busy && !rsp_valid && !notify && !fault_event && !rvalid;
 assign awwinfo={1'b0,64'hffffffffffffffff,6'b0,1'b1};
 assign arinfo=32'd1;
 assign rready=1'b1;
 wire ar_fire=arvalid && arready,w_fire=awwvalid && awwready;
 wire expected_r=busy && !quarantined && (operation==0 || verifying) && (waiting_r || ar_fire);
 reg fatal;
 task fail;
  input [31:0] why;
  begin
   if(!quarantined && !fatal)begin
    fatal=1;quarantined<=1;error<=why;
    fault_data<={token,operation,why};fault_valid<=1;
    // A generated ordinary reply is immutable. Report later faults ONLY on
    // the separate sticky event channel, never as a second reply for token.
    if(busy && !reply_generated)notify<=1;
   end
  end
 endtask
 always @(posedge clk or negedge por_n)begin
  if(!por_n)begin
   busy<=0;waiting_r<=0;quarantined<=0;notify<=0;token<=0;operation<=0;error<=0;verifying<=0;verify_cycles<=0;
   wait_cycles<=0;response_payload<=0;rsp_valid<=0;rsp_data<=0;reply_generated<=0;
   fault_valid<=0;fault_data<=0;
   awwaddr<=0;awwdata<=0;awwvalid<=0;araddr<=0;arvalid<=0;
  end else begin
   fatal=0;
   if(fault_valid && fault_ready)fault_valid<=0;
   if(rsp_valid && rsp_ready)begin rsp_valid<=0;if(!quarantined)busy<=0;end
   if(req_valid && req_ready)begin
    token<=req_data[639:576];operation<=req_data[575:544];error<=0;
    response_payload<=0;wait_cycles<=0;waiting_r<=0;busy<=1;reply_generated<=0;verifying<=0;verify_cycles<=0;
    if(quarantined)begin error<=4;notify<=1;end
    else if(req_data[575:544]>2 || req_data[517:512]!=0)begin error<=1;notify<=1;end
    else if(req_data[575:544]==0)begin araddr<=req_data[543:512];arvalid<=1;end
    else begin awwaddr<=req_data[543:512];awwdata<=req_data[511:0];awwvalid<=1;end
   end
   if(busy && !quarantined && !notify && !rsp_valid)begin
    if(ar_fire || rvalid || w_fire)wait_cycles<=0;
    else if(wait_cycles==TIMEOUT-1)fail(2);
    else wait_cycles<=wait_cycles+1;
   end
   if(ar_fire)begin arvalid<=0;waiting_r<=1;end
   if(w_fire)begin awwvalid<=0;if(!quarantined && !fault_event)begin
 if(operation==2)begin verifying<=1;verify_cycles<=0;araddr<=awwaddr;arvalid<=1;end
 else notify<=1;end end
   if(verifying && !quarantined && !notify && !rsp_valid)begin
    if(verify_cycles==TIMEOUT-1)fail(5);else verify_cycles<=verify_cycles+1;
   end
   if(rvalid)begin
    if(expected_r && !fault_event && !fatal)begin
     waiting_r<=0;
     if(verifying)begin
      if(rdata===awwdata)begin response_payload<=rdata;notify<=1;verifying<=0;end
      else begin araddr<=awwaddr;arvalid<=1;end
     end else begin response_payload<=rdata;notify<=1;end
    end
    else if(!quarantined)fail(3);
   end
   if(fault_event && !quarantined)fail(4);
   // Faults observed on the generation edge take priority over a pending
   // success. On the next edge exactly one error terminal reply is generated.
   if(notify && !rsp_valid && !reply_generated && !fatal)begin
    rsp_data<={token,operation,error,response_payload};rsp_valid<=1;notify<=0;reply_generated<=1;
   end
  end
 end
endmodule
