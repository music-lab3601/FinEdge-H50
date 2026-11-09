// One frozen physical-network source for ARM/PS and FPGA/PL. No book code.
#include "platform.hpp"
#include "finedge_contract.h"
#include <iostream>
#include <algorithm>
using namespace fe;
struct PacketPlan{U ordinal=0,count=0,offset=0,due_offset=0;};
#pragma pack(push,1)
struct Publication{U packet,ordinal,count,due,ready,submitted,done,overdue_events;};
#pragma pack(pop)
static_assert(sizeof(Publication)==64,"publication ABI");
// The TX loop publishes one count after a successful send. An independent
// observer writes immutable trace prefixes; no file I/O or UI feedback on TX.
class LivePublication {
 std::vector<Publication>&records;std::atomic<size_t>published{0};std::atomic<bool>stop{false};std::thread worker;std::string error;
 public:
 LivePublication(std::vector<Publication>&r,const std::filesystem::path&p,bool enabled):records(r){
  if(enabled)worker=std::thread([this,p]{try{
   pin(6);std::ofstream file(p,std::ios::binary);if(!file)throw std::runtime_error("Live publication open");size_t saved=0;
   for(;;){size_t n=published.load(std::memory_order_acquire);
    if(n>saved){file.write((const char*)(records.data()+saved),std::streamsize((n-saved)*sizeof(Publication)));file.flush();if(!file)throw std::runtime_error("Live publication write");saved=n;}
    if(stop.load(std::memory_order_acquire)&&saved==published.load(std::memory_order_acquire))break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
   }
  }catch(const std::exception&e){error=e.what();}});
 }
 void publish(size_t n){published.store(n,std::memory_order_release);}
 std::string finish(){stop.store(true,std::memory_order_release);if(worker.joinable())worker.join();return error;}
 ~LivePublication(){finish();}
};
int main(int argc,char**argv){
 Socket sock=INVALID;std::filesystem::path out;std::vector<Publication>trace;U submitted=0;std::string failure;bool created_output=false;
 try{
  Args a(argc,argv);a.only({"raw","first","last","session","first-packet","host","port","out","schedule","speed","rate","packetization","core","lead-ms","send-buffer-bytes","live-publications","duration-ns"});
  out=a.get("out");fresh_directory(out);created_output=true;net_init();
  U first=a.num("first"),last=a.num("last"),session=a.num("session"),packet=a.num("first-packet",1),speed=a.num("speed",1),rate=a.num("rate");
  if(!first||last<first||!session||!packet||last-first>2000000)throw std::runtime_error("Source bounds/session");
  std::string schedule=a.get("schedule","historical"),packing=a.get("packetization","same-ts");
  if((schedule!="historical"&&schedule!="rate")||(packing!="same-ts"&&packing!="single")||!speed||speed>1000||(schedule=="rate"&&(!rate||rate>10000000||packing!="single")))throw std::runtime_error("Explicit schedule/packetization");
  U n=last-first+1;std::vector<FinEdgeRawMbo>raw(static_cast<size_t>(n));
  std::ifstream input(a.get("raw"),std::ios::binary);if(!input)throw std::runtime_error("Source raw file");
  input.seekg(std::streamoff((first-1)*56));input.read((char*)raw.data(),std::streamsize(n*56));if(U(input.gcount())!=n*56)throw std::runtime_error("Truncated raw source");
  for(size_t i=1;i<raw.size();i++)if(raw[i].ts_event<raw[i-1].ts_event)throw std::runtime_error("Source event time regression");
  std::vector<PacketPlan>plan;plan.reserve(size_t(n));std::vector<uint8_t>blob;blob.reserve(size_t(n*88));
  for(U i=0;i<n;){U count=1;if(packing=="same-ts")while(count<24&&i+count<n&&raw[size_t(i+count)].ts_event==raw[size_t(i)].ts_event)count++;
   U offset=blob.size(),due=schedule=="historical"?(raw[size_t(i)].ts_event-raw[0].ts_event)/speed:i*1000000000ull/rate;
   FinEdgeFmeHeader h{FINEDGE_EVENT_MAGIC,1,uint16_t(count),session,packet+U(plan.size()),first+i};
   blob.resize(size_t(offset+32+56*count));std::memcpy(blob.data()+offset,&h,32);std::memcpy(blob.data()+offset+32,raw.data()+i,size_t(count*56));
   plan.push_back({first+i,count,offset,due});i+=count;
  }
  U duration=a.num("duration-ns",0);if(duration&&duration<plan.back().due_offset)throw std::runtime_error("Duration precedes final event");
  trace.resize(plan.size());LivePublication live(trace,out/"publication_live.bin",a.num("live-publications",0)!=0);
  sock=udp_socket();auto target=endpoint(a.get("host"),unsigned(a.num("port",20001)));
  int requested=int(a.num("send-buffer-bytes",262144));if(setsockopt(sock,SOL_SOCKET,SO_SNDBUF,(const char*)&requested,sizeof requested))throw std::runtime_error("SO_SNDBUF");
  if(connect(sock,(sockaddr*)&target,sizeof target))throw std::runtime_error("Source UDP connect");
  pin(unsigned(a.num("core",4)));U anchor=now_ns()+a.num("lead-ms",200)*1000000ull;
  write_text(out/"anchor.json","{\"anchor_ns\":"+std::to_string(anchor)+",\"session\":"+std::to_string(session)+",\"first_packet\":"+std::to_string(packet)+",\"first\":"+std::to_string(first)+",\"last\":"+std::to_string(last)+"}");
  size_t actual=0;
  for(size_t i=0;i<plan.size();i++){
   auto&p=plan[i];U due=anchor+p.due_offset;while(now_ns()<due)relax();
   U ready=now_ns(),before=now_ns();int bytes=int(32+p.count*56);
   int sent=int(send(sock,(const char*)blob.data()+p.offset,bytes,0));U after=now_ns();
   trace[i]={packet+U(i),p.ordinal,p.count,due,ready,before,after,0};actual=i+1;
   if(sent!=bytes){failure="Nonblocking UDP send failed; no retry/re-anchor";break;}
   submitted+=p.count;
   live.publish(i+1);
  }
  if(failure.empty()&&duration)while(now_ns()<anchor+duration)relax();
  U schedule_finished=now_ns();std::string live_error=live.finish();
  trace.resize(actual);
  U peak_overdue=0;size_t right=0;
  for(size_t i=0;i<trace.size();i++){
   while(right<plan.size()&&anchor+plan[right].due_offset<=trace[i].submitted)right++;
   U due_ordinal=right?plan[right-1].ordinal+plan[right-1].count:first;
   trace[i].overdue_events=due_ordinal>plan[i].ordinal?due_ordinal-plan[i].ordinal:0;
   peak_overdue=std::max(peak_overdue,trace[i].overdue_events);
  }
  std::ofstream log(out/"publication.bin",std::ios::binary);log.write((const char*)trace.data(),std::streamsize(trace.size()*sizeof(Publication)));if(!log)throw std::runtime_error("Publication write");log.close();
  std::string status=failure.empty()&&submitted==n?"passed":"failed";
  std::string result="{\"status\":"+quote(status)+",\"error\":"+quote(failure)+",\"events_submitted\":"+std::to_string(submitted)+",\"packets_submitted\":"+std::to_string(trace.size())+
   ",\"anchor_ns\":"+std::to_string(anchor)+",\"last_send_done_ns\":"+std::to_string(trace.empty()?0:trace.back().done)+",\"source_clock_waits_for_measured_consumer\":false,\"single_TX_thread\":true,\"payloads_prebuilt\":true,\"book_or_model\":false"+
   ",\"source_queue_peak_events\":"+std::to_string(peak_overdue)+",\"queue_metric\":\"planned-due backlog at API submission; no producer queue\",\"schedule\":"+quote(schedule)+",\"packetization\":"+quote(packing)+
   ",\"speed\":"+std::to_string(speed)+",\"rate_events_s\":"+std::to_string(rate)+",\"send_buffer_requested_bytes\":"+std::to_string(requested)+",\"send_timestamp_kind\":\"API_submission_not_NIC_hardware\",\"live_publication_error\":"+quote(live_error)+",\"live_publication_interval_ms\":20,\"requested_duration_ns\":"+std::to_string(duration)+",\"schedule_finished_ns\":"+std::to_string(schedule_finished)+"}";
  write_text(out/"source.json",result);std::cout<<result<<std::endl;close_socket(sock);return status=="passed"?0:2;
 }catch(const std::exception&e){if(sock!=INVALID)close_socket(sock);if(created_output)try{write_text(out/"startup_failure.json","{\"status\":\"failed\",\"error\":"+quote(e.what())+"}");}catch(...){}std::cerr<<e.what()<<std::endl;return 3;}
}
