// Independent read-only verification: only two input-channel weight tensors
// may differ; all biases and downstream quantized parameters must be identical.
#include <icraft-xir/core/network.h>
#include <icraft-xir/utils/json.hpp>
#include <fstream>
#include <cstring>
#include <set>
#include <iostream>
using namespace icraft::xir;
using Json=nlohmann::json;
float roundtf(float x){uint32_t b;std::memcpy(&b,&x,4);b=(b+0xFFF+((b>>13)&1))&0xFFFFE000;std::memcpy(&x,&b,4);return x;}
int main(int argc,char**argv){try{
  if(argc!=6)throw std::runtime_error("source.json source.raw folded.json folded.raw scales.f32");
  auto a=Network::CreateFromJsonFile(argv[1]);a.loadParamsFromFile(argv[2]);
  auto b=Network::CreateFromJsonFile(argv[3]);b.loadParamsFromFile(argv[4]);
  float s[40];std::ifstream f(argv[5],std::ios::binary);f.read(reinterpret_cast<char*>(s),sizeof s);
  if(f.gcount()!=sizeof s)throw std::runtime_error("Scales");
  Json rows=Json::array();std::set<int64_t> seen;size_t changed_tensors=0;
  for(auto p:a.allParams()){
    auto id=p->v_id;if(!seen.insert(id).second)continue;
    auto q=b.getValueById(id).cast<Params>();auto n=p.tensorType().bytes();
    if(n!=q.tensorType().bytes())throw std::runtime_error("Parameter byte size changed");
    auto left=p.data<char>(),right=q.data<char>();bool equal=std::memcmp(left.get(),right.get(),n)==0;
    if(id==6||id==20){
      auto t=p.tensorType();if(t->shape.size()!=4||t->shape[2]!=40||t->shape[3]!=32)throw std::runtime_error("HWIO source shape");
      auto x=p.data<float>(),y=q.data<float>();
      for(size_t i=0;i<t.numElements();++i){float e=roundtf(x[i]*s[(i/32)%40]);if(std::memcmp(&e,&y[i],4))throw std::runtime_error("Scale fold formula mismatch");}
    }else if(!equal)throw std::runtime_error("Unexpected changed downstream weight/bias");
    changed_tensors+=!equal;rows.push_back(Json{{"id",id},{"bytes",n},{"bit_equal",equal},{"formula_checked",id==6||id==20}});
  }
  if(changed_tensors!=2)throw std::runtime_error("Expected exactly two folded weight tensors");
  std::cout<<Json{{"status","passed"},{"parameters_checked",seen.size()},{"changed_tensors",changed_tensors},{"parameters",rows},{"board_accessed",false}}.dump(2)<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
