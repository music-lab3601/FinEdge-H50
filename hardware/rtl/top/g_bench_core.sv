// Production profile: old phase-only DDR counters/histogram endpoints return0.
// FME1-admitted UDP payload -> bounded raw FIFO -> exact book -> full top10 sink.
// External memory replies MUST mean read data / visible write completion.
module g_bench_core #(
 parameter ORDER_BITS=17,LEVEL_BITS=14,LEVEL_BANK_BITS=5,STATUS_MAX=8,
 parameter FIFO_DEPTH=256,END_TIMEOUT_CYCLES=125000000,CLOCK_HZ=125000000
)(
 input wire clk,rst,enable,memory_initialized,external_fault,
 input wire [63:0] session,day_ns,start_ns,end_ns,
 input wire [31:0] order_base,instrument_id,input wire [15:0] publisher_id,
 input wire [7:0] status_count,input wire [STATUS_MAX*64-1:0] status_ts,
 input wire [STATUS_MAX-1:0] status_allowed,
 input wire [63:0] measure_first,measure_last,
 input wire finish_valid,input wire [63:0] expected_last,
 input wire s_valid,output wire s_ready,input wire [63:0] s_data,input wire [7:0] s_keep,input wire s_last,
 output wire book_valid,input wire book_ready,output wire [2559:0] book_data,output wire [255:0] book_identity,
 output wire mem_valid,input wire mem_ready,output wire mem_write,output wire [31:0] mem_addr,
 output wire [511:0] mem_wdata,input wire mem_rsp_valid,input wire [511:0] mem_rdata,input wire mem_error,
 input wire [11:0] monitor_addr,output reg [63:0] monitor_data,
 output wire initialized,fault,output wire [31:0] error,output reg finished,
 output wire sample_valid,output wire [63:0] sample_ordinal,sample_accept_cycle,sample_commit_cycle,
 output wire sample_mutated,sample_eligible,
 output wire book_frame_valid,output wire [3327:0] book_frame,
 output wire [63:0] raw_committed_out,eligible_count_out,cycle_out,book_accept_tick
);
 localparam [63:0] CLOCK_HZ64=CLOCK_HZ;
 wire rvalid,rready,rx_fault,packet_idle;wire [447:0] rdata;wire [63:0] rordinal,packets,received;
 wire [31:0] rx_error;reg local_fault;reg [31:0] local_error;
 wire bvalid,bready,book_fault;wire [31:0] book_error;wire [63:0] accepted,mutations,eligible;
 wire done,done_mutated,done_eligible;wire [63:0] done_ordinal,done_ts;wire [4:0] phase;
 wire internal_book_valid;
 wire [31:0] max_probe;wire [63:0] probe_steps;
 wire [575:0] qdata;wire qvalid;wire [$clog2(FIFO_DEPTH):0] qcount;
 reg [63:0] cycle,committed,enqueued,active_accept,active_ordinal;
 assign book_accept_tick=active_accept;
 assign raw_committed_out=committed;assign eligible_count_out=eligible;assign cycle_out=cycle;
 reg [63:0] rx_arrival,stamped_ordinal;
 reg active,mem_pending;
 reg [63:0] measured_received,measured_committed,measured_rows,first_accept,last_commit;
 reg [63:0] queue_peak,backlog_peak,latency_sum,latency_max,oldest_age_max;
 reg [63:0] mem_reads,mem_writes,mem_wait,price_cycles,lookup_cycles,snapshot_cycles;
 reg [63:0] rx_stall,book_stall,output_stall;
 reg [63:0] measured_mem_reads,measured_mem_writes,measured_mem_wait;
 reg [63:0] measured_queue_peak,measured_backlog_peak,measured_oldest_age_max;
 reg [63:0] active_start,service_sum,service_max;
 reg [63:0] latency_hist[0:31];integer hb,latency_bin;
 reg [63:0] latency_delta,service_delta;
 reg [4:0] histogram_bin;reg metrics_v1,metrics_v2;
 reg [63:0] age_registered,pending_registered;reg measured_observation;
 reg [31:0] end_wait;reg ending;reg [63:0] declared_last;
 wire in_measure=rordinal>=measure_first&&rordinal<=measure_last;
 wire [63:0] arrival_cycle=stamped_ordinal==rordinal?rx_arrival:cycle;
 wire active_measure=active_ordinal>=measure_first&&active_ordinal<=measure_last;
 wire [63:0] age=active?cycle-active_accept:qvalid?cycle-qdata[575:512]:0;
 wire active_enable=enable&&memory_initialized&&!fault&&!finished;
 assign book_valid=internal_book_valid&&!fault;
 assign book_frame_valid=book_valid&&book_ready;
 fb_bookrow_r1 pack_book(.session_id(session),.stream_id(64'd0),.day_ns(day_ns),
  .local_accept_ticks(active_accept),.local_commit_ticks(cycle),.clock_hz(CLOCK_HZ64),
  .identity(book_identity),.values(book_data),.frame(book_frame));
 assign fault=local_fault||external_fault||rx_fault||book_fault;
 assign error=local_fault?local_error:external_fault?32'hb0000001:rx_fault?(32'hb1000000|rx_error):book_fault?(32'hb2000000|book_error):0;
 wire book_init;
 assign initialized=book_init&&memory_initialized&&!fault;
 fb_udp_mbo_rx decoder(.clk(clk),.rst(rst),.enable(active_enable),.session(session),
  .s_valid(s_valid),.s_ready(s_ready),.s_data(s_data),.s_keep(s_keep),.s_last(s_last),
  .m_valid(rvalid),.m_ready(rready),.m_data(rdata),.m_ordinal(rordinal),.packet_idle(packet_idle),
  .fault(rx_fault),.error(rx_error),.packets(packets),.records(received));
 fb_event_fifo #(.WIDTH(576),.DEPTH(FIFO_DEPTH)) queue(.clk(clk),.rst(rst),
  .s_valid(rvalid&&active_enable),.s_ready(rready),.s_data({arrival_cycle,rordinal,rdata}),
  .m_valid(qvalid),.m_ready(bready&&active_enable),.m_data(qdata),.occupancy(qcount));
 assign bvalid=qvalid&&active_enable;
 fb_book_core #(.ORDER_BITS(ORDER_BITS),.LEVEL_BITS(LEVEL_BITS),.LEVEL_BANK_BITS(LEVEL_BANK_BITS),.STATUS_MAX(STATUS_MAX)) book(
  .clk(clk),.rst(rst),.enable(active_enable),.order_base(order_base),.publisher_id(publisher_id),.instrument_id(instrument_id),
  .day_ns(day_ns),.start_ns(start_ns),.end_ns(end_ns),.status_count(status_count),.status_ts(status_ts),.status_allowed(status_allowed),
  .s_valid(bvalid),.s_ready(bready),.s_data(qdata[447:0]),.s_ordinal(qdata[511:448]),
  .m_valid(internal_book_valid),.m_ready(book_ready&&!fault),.m_book(book_data),.m_identity(book_identity),
  .mem_valid(mem_valid),.mem_ready(mem_ready),.mem_write(mem_write),.mem_addr(mem_addr),.mem_wdata(mem_wdata),
  .mem_rsp_valid(mem_rsp_valid),.mem_rdata(mem_rdata),.mem_error(mem_error),
  .fault(book_fault),.error(book_error),.raw_count(accepted),.mutation_count(mutations),.eligible_count(eligible),.initialized(book_init),
  .done_valid(done),.done_ordinal(done_ordinal),.done_ts(done_ts),.done_mutated(done_mutated),.done_eligible(done_eligible),
  .phase(phase),.lookup_max_probe(max_probe),.lookup_probe_steps(probe_steps));
 assign sample_valid=done&&!fault&&(done_ordinal>=measure_first&&done_ordinal<=measure_last)&&
  (done_ordinal==measure_first||done_ordinal==measure_last||done_ordinal[7:0]==0);
 assign sample_ordinal=done_ordinal;assign sample_accept_cycle=active_accept;assign sample_commit_cycle=cycle;
 assign sample_mutated=done_mutated;assign sample_eligible=done_eligible;
 task fail(input [31:0] why);begin local_fault<=1;local_error<=why;end endtask
 integer g;
 always @(posedge clk)begin
  if(rst)begin
   metrics_v1<=0;metrics_v2<=0;latency_delta<=0;service_delta<=0;histogram_bin<=0;
   age_registered<=0;pending_registered<=0;measured_observation<=0;
   cycle<=0;committed<=0;enqueued<=0;active<=0;active_accept<=0;active_ordinal<=0;rx_arrival<=0;stamped_ordinal<=0;
   measured_received<=0;measured_committed<=0;measured_rows<=0;first_accept<=0;last_commit<=0;
   queue_peak<=0;backlog_peak<=0;latency_sum<=0;latency_max<=0;oldest_age_max<=0;
   mem_reads<=0;mem_writes<=0;mem_wait<=0;mem_pending<=0;price_cycles<=0;lookup_cycles<=0;snapshot_cycles<=0;
   rx_stall<=0;book_stall<=0;output_stall<=0;measured_mem_reads<=0;measured_mem_writes<=0;measured_mem_wait<=0;
   measured_queue_peak<=0;measured_backlog_peak<=0;measured_oldest_age_max<=0;active_start<=0;service_sum<=0;service_max<=0;
   for(hb=0;hb<32;hb=hb+1)latency_hist[hb]<=0;
   end_wait<=0;ending<=0;declared_last<=0;local_fault<=0;local_error<=0;finished<=0;
  end else begin
   cycle<=cycle+1;metrics_v1<=0;metrics_v2<=0;
   if(enable&&!fault)begin
    if(!memory_initialized)fail(32'hb0000010);
    if(status_count==0||status_count>STATUS_MAX||order_base[5:0]!=0||
       {1'b0,order_base}+(33'd64<<ORDER_BITS)>33'h100000000||measure_first==0||measure_last<measure_first)
      fail(32'hb0000011);
    for(g=1;g<STATUS_MAX;g=g+1)if(g<status_count&&status_ts[g*64+:64]<=status_ts[(g-1)*64+:64])fail(32'hb0000012);
   end
   if(!fault)begin
    // Observation-only pipeline: subtract -> aggregate/bin -> histogram.
    // The source timestamps and book/local-sink handshake are unchanged.
    metrics_v1<=done&&active_measure;metrics_v2<=metrics_v1;
    if(metrics_v1)begin
     latency_sum<=latency_sum+latency_delta;
     if(latency_delta>latency_max)latency_max<=latency_delta;
     service_sum<=service_sum+service_delta;
     if(service_delta>service_max)service_max<=service_delta;
     latency_bin=0;
     for(hb=1;hb<32;hb=hb+1)if(latency_delta>=(64'd1<<hb))latency_bin=hb;
     histogram_bin<=latency_bin[4:0];
    end
    if(metrics_v2)latency_hist[histogram_bin]<=latency_hist[histogram_bin]+1;
    // Also separate subtraction from peak comparisons on age/backlog.
    age_registered<=age;pending_registered<=received-committed;
    measured_observation<=(active&&active_measure)||(qvalid&&qdata[511:448]>=measure_first&&qdata[511:448]<=measure_last);
    if(rvalid&&stamped_ordinal!=rordinal)begin
     rx_arrival<=cycle;stamped_ordinal<=rordinal;
     if(in_measure)begin measured_received<=measured_received+1;if(measured_received==0)first_accept<=cycle;end
    end
    if(rvalid&&rready&&active_enable)begin
     enqueued<=enqueued+1;
    end
    if(qcount>queue_peak)queue_peak<=qcount;
    if(pending_registered>backlog_peak)backlog_peak<=pending_registered;
    if(age_registered>oldest_age_max)oldest_age_max<=age_registered;
    if((active&&active_measure)||(qvalid&&qdata[511:448]>=measure_first&&qdata[511:448]<=measure_last))begin
     if(qcount>measured_queue_peak)measured_queue_peak<=qcount;

    end
    if(measured_observation)begin
     if(pending_registered>measured_backlog_peak)measured_backlog_peak<=pending_registered;
     if(age_registered>measured_oldest_age_max)measured_oldest_age_max<=age_registered;
    end
    if(bvalid&&bready)begin
     active<=1;active_accept<=qdata[575:512];active_start<=cycle;active_ordinal<=qdata[511:448];
    end
    if(done)begin
     if(!active||done_ordinal!=active_ordinal||done_ordinal!=committed+1)fail(32'hb0000020);
     committed<=committed+1;
     // A new raw may be accepted on the same edge as the prior completion pulse.
     if(!(bvalid&&bready))active<=0;
     if(active_measure)begin
      measured_committed<=measured_committed+1;measured_rows<=measured_rows+done_eligible;last_commit<=cycle;
      latency_delta<=cycle-active_accept;service_delta<=cycle-active_start;
     end
    end
    if(mem_valid&&mem_ready)begin
     mem_pending<=1;
     if(mem_write)mem_writes<=mem_writes+1;else mem_reads<=mem_reads+1;
     if(active_measure)begin
      if(mem_write)measured_mem_writes<=measured_mem_writes+1;else measured_mem_reads<=measured_mem_reads+1;
     end
     if(mem_addr[5:0]!=0||mem_addr<order_base||{1'b0,mem_addr}>={1'b0,order_base}+(33'd64<<ORDER_BITS))fail(32'hb0000030);
    end
    if(mem_rsp_valid)mem_pending<=0;
    if(mem_pending||(mem_valid&&!mem_ready))begin
     mem_wait<=mem_wait+1;if(active_measure)measured_mem_wait<=measured_mem_wait+1;
    end
    if(active_measure)begin
     if(phase==13)price_cycles<=price_cycles+1;
     if(phase==1||phase==2)lookup_cycles<=lookup_cycles+1;
     if(phase==9||phase==10) snapshot_cycles<=snapshot_cycles+1;
    end
    if(s_valid&&!s_ready)rx_stall<=rx_stall+1;
    if(qvalid&&!bready)book_stall<=book_stall+1;
    if(book_valid&&!book_ready)output_stall<=output_stall+1;
    if(finish_valid&&!ending)begin ending<=1;declared_last<=expected_last;end_wait<=0;end
    if(ending&&!finished)begin
     if(received>declared_last||committed>declared_last)fail(32'hb0000040);
     else if(received==declared_last&&committed==declared_last&&packet_idle&&qcount==0&&!active&&!metrics_v1&&!metrics_v2)finished<=1;
     else if(end_wait==END_TIMEOUT_CYCLES-1)fail(32'hb0000041);
     else end_wait<=end_wait+1;
    end
   end
  end
 end
 // Same-clock read view. G must use a snapshot mailbox or freeze before GP reads.
 always @*begin
  monitor_data=0;
  case(monitor_addr)
   12'h000:monitor_data=64'h314248434e454246; // FBENCHB1
   12'h008:monitor_data={error,27'b0,finished,ending,initialized,enable,fault};
   12'h010:monitor_data=cycle;12'h018:monitor_data=packets;
   12'h020:monitor_data=received;12'h028:monitor_data=enqueued;12'h030:monitor_data=accepted;12'h038:monitor_data=committed;
   12'h040:monitor_data=mutations;12'h048:monitor_data=eligible;
   12'h050:monitor_data=measured_received;12'h058:monitor_data=measured_committed;12'h060:monitor_data=measured_rows;
   12'h068:monitor_data=first_accept;12'h070:monitor_data=last_commit;
   12'h078:monitor_data=qcount;12'h080:monitor_data=queue_peak;12'h088:monitor_data=received-committed;
   12'h090:monitor_data=backlog_peak;12'h098:monitor_data=age;12'h0a0:monitor_data=oldest_age_max;
   12'h0a8:monitor_data=latency_sum;12'h0b0:monitor_data=latency_max;
   12'h0b8:monitor_data=0;12'h0c0:monitor_data=0;12'h0c8:monitor_data=0;
   12'h0d0:monitor_data=0;12'h0d8:monitor_data=0;12'h0e0:monitor_data=0;
   12'h0e8:monitor_data=0;12'h0f0:monitor_data=0;12'h0f8:monitor_data=0;
   12'h100:monitor_data=0;12'h108:monitor_data=0;
   12'h110:monitor_data=0;12'h118:monitor_data=0;12'h120:monitor_data=0;
   12'h128:monitor_data=0;
   12'h130:monitor_data=measured_queue_peak;12'h138:monitor_data=measured_backlog_peak;12'h140:monitor_data=measured_oldest_age_max;
   12'h148:monitor_data=service_sum;12'h150:monitor_data=service_max;
   default:monitor_data=0;
  endcase
 end
endmodule
