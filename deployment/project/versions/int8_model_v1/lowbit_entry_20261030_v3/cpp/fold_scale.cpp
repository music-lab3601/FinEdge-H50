#include <icraft-xir/core/network.h>
#include <icraft-xir/utils/json.hpp>
#include <fstream>
#include <cstring>
#include <iostream>
#include <cmath>
using namespace icraft::xir;
using Json=nlohmann::json;
float tf32(float value){uint32_t bits;std::memcpy(&bits,&value,4);bits=(bits+0xFFF+((bits>>13)&1u))&0xFFFFE000u;std::memcpy(&value,&bits,4);return value;}
int main(int argc,char**argv){try{
  if(argc!=6)throw std::runtime_error("source.json source.raw scales.f32 out.json out.raw");
  auto net=Network::CreateFromJsonFile(argv[1]);net.loadParamsFromFile(argv[2]);
  float scale[40];std::ifstream in(argv[3],std::ios::binary);in.read(reinterpret_cast<char*>(scale),sizeof scale);
  if(in.gcount()!=sizeof scale)throw std::runtime_error("40 scales required");
  Json changes=Json::array();
  for(auto id:{6,20}){
    auto p=net.getValueById(id).cast<Params>();auto t=p.tensorType();
    if(t->shape.size()!=4||t->shape[2]!=40||t->shape[3]!=32)throw std::runtime_error("Unexpected HWIO source weights");
    auto values=p.data<float>();auto count=t.numElements();double max_abs=0;size_t changed=0;
    for(size_t i=0;i<count;++i){float old=values[i],ideal=old*scale[(i/32)%40];float next=tf32(ideal);
      if(!std::isfinite(next))throw std::runtime_error("Fold overflow");max_abs=std::max(max_abs,double(std::abs(next-ideal)));changed+=old!=next;values[i]=next;
    }
    changes.push_back(Json{{"parameter_id",id},{"elements",count},{"changed",changed},{"max_abs_additional_tf32_rounding",max_abs}});
  }
  net.dumpJsonAndParams(std::filesystem::path(argv[4]),std::filesystem::path(argv[5]));
  std::cout<<Json{{"status","folded"},{"changes",changes},{"training",false}}.dump(2)<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
