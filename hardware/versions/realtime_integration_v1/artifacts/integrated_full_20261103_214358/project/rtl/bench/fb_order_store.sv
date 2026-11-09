// Three 160-bit entries per owned DDR beat; key comparisons run in parallel.
// One mutation owns the cached beat until commit. Never reorder same-key writes.
module fb_order_store #(parameter BITS=17, parameter MAX_PROBE=256)(
 input wire clk,rst,clear,input wire [31:0] base,input wire [63:0] lookup_key,
 input wire lookup_valid,output wire lookup_ready,
 output reg found_valid,input wire found_ready,output reg found,
 output reg [31:0] old_price,old_size,output reg old_side,
 input wire commit_valid,output wire commit_ready,input wire commit_write,commit_delete,commit_side,
 input wire [31:0] commit_price,commit_size,
 output reg mem_valid,input wire mem_ready,output reg mem_write,
 output reg [31:0] mem_addr,output reg [511:0] mem_wdata,
 input wire mem_rsp_valid,input wire [511:0] mem_rdata,input wire mem_error,
 output reg busy,output reg fault,output reg [31:0] error,output reg [31:0] max_probe,output reg [63:0] probe_steps
);
 localparam IDLE=0,READ=1,WAIT_READ=2,CHECK=3,FOUND=4,COMMIT=5,WRITE=6,WAIT_WRITE=7;
 reg [3:0] state;reg [31:0] epoch;reg [BITS-1:0] bucket,free_bucket;
 reg [31:0] probes;reg [63:0] key;reg [511:0] cached,free_data;
 reg free_saved;reg [1:0] chosen,free_slot;
 reg [159:0] entry[0:2];integer j,hit,empty,available;reg unused;
 wire [63:0] hashed=lookup_key^(lookup_key>>17)^(lookup_key>>37);
 assign lookup_ready=state==IDLE&&!fault&&!clear;
 assign commit_ready=state==COMMIT&&!fault;
 task fail(input [31:0] why);begin fault<=1;error<=why;mem_valid<=0;found_valid<=0;end endtask
 always @(posedge clk)begin
  if(rst)begin state<=IDLE;epoch<=1;fault<=0;error<=0;mem_valid<=0;found_valid<=0;busy<=0;max_probe<=0;probe_steps<=0;end
  else if(!fault)begin
   if(mem_error)fail(1);
   if(clear)begin
    if(state!=IDLE||epoch==32'hffffffff)fail(2);
    else epoch<=epoch+1;
   end else case(state)
    IDLE:if(lookup_valid)begin key<=lookup_key;bucket<=hashed[BITS-1:0];probes<=0;free_saved<=0;busy<=1;state<=READ;end
    READ:begin mem_addr<=base+({{(32-BITS){1'b0}},bucket}<<6);mem_write<=0;mem_valid<=1;state<=WAIT_READ;end
    WAIT_READ:begin
     if(mem_valid&&mem_ready)mem_valid<=0;
     if(mem_rsp_valid)begin cached<=mem_rdata;state<=CHECK;end
    end
    CHECK:begin
     probe_steps<=probe_steps+1;if(probes+1>max_probe)max_probe<=probes+1;
     hit=-1;empty=-1;available=-1;unused=cached[511:480]!=epoch;
     for(j=0;j<3;j=j+1)begin
      entry[j]=cached[j*160+:160];
      if(!unused&&entry[j][129:128]==1&&entry[j][63:0]==key)hit=j;
      if(unused||entry[j][129:128]==0)empty=j;
      if(unused||entry[j][129:128]!=1)if(available<0)available=j;
     end
     if(hit>=0)begin chosen<=hit;found<=1;old_price<=entry[hit][95:64];old_size<=entry[hit][127:96];old_side<=entry[hit][130];state<=FOUND;end
     else if(empty>=0)begin
      found<=0;old_price<=0;old_size<=0;old_side<=0;
      if(free_saved)begin cached<=free_data;bucket<=free_bucket;chosen<=free_slot;end
      else begin chosen<=available;if(unused)cached<={epoch,480'b0};end
      state<=FOUND;
     end else if(probes==MAX_PROBE-1)fail(3);
     else begin
      if(!free_saved&&available>=0)begin free_saved<=1;free_slot<=available;free_data<=cached;free_bucket<=bucket;end
      probes<=probes+1;bucket<=bucket+1'b1;state<=READ;
     end
    end
    FOUND:begin found_valid<=1;if(found_valid&&found_ready)begin found_valid<=0;state<=COMMIT;end end
    COMMIT:if(commit_valid)begin
     if(!commit_write)begin busy<=0;state<=IDLE;end
     else begin
      cached[chosen*160+:160]<={29'b0,commit_side,(commit_delete?2'd2:2'd1),commit_size,commit_price,key};state<=WRITE;
     end
    end
    WRITE:begin mem_addr<=base+({{(32-BITS){1'b0}},bucket}<<6);mem_wdata<=cached;mem_write<=1;mem_valid<=1;state<=WAIT_WRITE;end
    WAIT_WRITE:begin if(mem_valid&&mem_ready)mem_valid<=0;if(mem_rsp_valid)begin busy<=0;state<=IDLE;end end
    default:fail(4);
   endcase
  end
 end
endmodule
