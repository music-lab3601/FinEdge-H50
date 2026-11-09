#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include "frozen.h"
using U=uint64_t;
void fail(const char*x){throw std::runtime_error(x);}
#pragma pack(push,1)
struct Raw{uint8_t len,type;uint16_t publisher;uint32_t instrument;U ts,id;int64_t price;uint32_t size;uint8_t flags,channel;char action,side;U recv;int32_t delta;uint32_t sequence;};
struct Wdm{U id[4];uint32_t emit;uint8_t pad[28];float value[40];uint8_t tail[32];};
struct Trace{Wdm w;U raw[40];};
struct Row{U identity[4];U raw[40];float normalized[40];};
#pragma pack(pop)
static_assert(sizeof(Raw)==56&&sizeof(Row)==512,"offline row ABI");
struct Order{char side;U price,size;};
struct Book{
 std::unordered_map<U,Order> orders;std::map<U,U> bids,asks;bool initialized=false,previous=false;int gate=-1;U ordinal=0,row=0,segment=0,segcount=0,lastts=0;
 Book(){orders.reserve(262144);}
 void qty(char side,U p,int64_t d){auto&l=side=='B'?bids:asks;auto it=l.find(p);U old=it==l.end()?0:it->second;if(d<0&&old<(U)(-d))fail("negative depth");U v=old+d;if(!v){if(it!=l.end())l.erase(it);}else l[p]=v;}
 bool apply(const Raw&r,Trace&t){
  ordinal++;if(r.len!=14||r.publisher!=2||r.instrument!=38||r.ts<lastts)fail("raw identity/time");lastts=r.ts;
  int g=gate;while(g+1<GATE_N&&GATE_TS[g+1]<=r.ts)g++;if(g!=gate){previous=false;gate=g;}
  if(r.action=='T'||r.action=='F'||r.action=='N')return false;
  if(r.action=='R'){orders.clear();bids.clear();asks.clear();initialized=true;previous=false;return false;}
  if(!initialized||!(r.side=='A'||r.side=='B')||!r.id||!r.size||(r.flags&(4|16|64)))fail("raw action state");
  auto it=orders.find(r.id);
  if(r.action=='C'){
   if(it==orders.end()||it->second.side!=r.side||r.size>it->second.size)fail("cancel");
   qty(r.side,it->second.price,-(int64_t)r.size);it->second.size-=r.size;if(!it->second.size)orders.erase(it);
  }else{
   if(r.price<=0||r.price==INT64_MAX||r.price%100000)fail("price");U p=r.price/100000;
   if(r.action=='A'){if(it!=orders.end())fail("duplicate");}
   else if(r.action=='M'){if(it==orders.end()||it->second.side!=r.side)fail("modify");qty(r.side,it->second.price,-(int64_t)it->second.size);}
   else fail("action");orders[r.id]={r.side,p,r.size};qty(r.side,p,r.size);
  }
  U current=row++;
  bool ok=r.ts>=START_NS&&r.ts<END_NS&&gate>=0&&GATE_OK[gate]&&r.ts!=GATE_TS[gate]&&bids.size()>=10&&asks.size()>=10&&bids.rbegin()->first<asks.begin()->first;
  if(!ok){previous=false;return false;}
  if(!previous){segment++;segcount=0;}previous=true;segcount++;
  auto a=asks.begin();auto b=bids.rbegin();for(int i=0;i<10;i++,a++,b++){t.raw[4*i]=a->first;t.raw[4*i+1]=a->second;t.raw[4*i+2]=b->first;t.raw[4*i+3]=b->second;}
  for(int i=0;i<40;i+=2)if(!t.raw[i]||t.raw[i]>=(1ull<<51))fail("delivery price range");
  memset(&t.w,0,sizeof t.w);t.w.id[0]=segment;t.w.id[1]=current;t.w.id[2]=ordinal;t.w.id[3]=r.ts-DAY_NS-18000000000000ull;
  double mid=((double)t.raw[0]+(double)t.raw[2])/2.0;
  for(int i=0;i<40;i++){double x=(double)t.raw[i];if(i%2==0)x=(x-mid)/mid;t.w.value[i]=(float)((x-MEAN[i])/STDDEV[i]);if(!std::isfinite(t.w.value[i]))fail("normalized nonfinite");}
  t.w.emit=segcount>=100;return true;
 }
};

int main(int argc,char**argv){try{
  if(argc!=4)throw std::runtime_error("cache output count");
  const auto wanted=std::stoull(argv[3]);std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);
  if(!in||!out)throw std::runtime_error("File open");
  Raw r;Book b;uint64_t accepted=0;
  while(in.read(reinterpret_cast<char*>(&r),sizeof r)){
    Trace t{};if(!b.apply(r,t))continue;
    Row row;std::memcpy(row.identity,t.w.id,32);std::memcpy(row.raw,t.raw,320);std::memcpy(row.normalized,t.w.value,160);
    out.write(reinterpret_cast<char*>(&row),sizeof row);if(++accepted==wanted)break;
  }
  if(accepted!=wanted)throw std::runtime_error("Not enough accepted records");
  std::cout<<"{\"accepted\":"<<accepted<<",\"raw_consumed\":"<<b.ordinal<<"}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
