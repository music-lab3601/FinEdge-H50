// Host-only utility. No device URL, socket backend or board reset exists here.
#include <icraft-xir/core/network.h>
#include <icraft-xir/utils/json.hpp>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <filesystem>

using namespace icraft::xir;
using namespace icraft::xrt;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;

int main(int argc,char**argv){
  try{
    std::map<std::string,std::string> args;
    if((argc-1)%2)throw std::runtime_error("Arguments must be --key value pairs");
    for(int i=1;i<argc;i+=2)args[argv[i]]=argv[i+1];
    for(auto key:{"--json","--raw","--inputs","--out","--mode"})if(!args.count(key))throw std::runtime_error("Missing argument");
    const auto mode=args.at("--mode");
    if(mode!="forward"&&mode!="encode")throw std::runtime_error("Host mode is forward or encode");
    auto net=Network::CreateFromJsonFile(args.at("--json"));net.loadParamsFromFile(args.at("--raw"));
    auto view=net.view(0);
    if(view.inputs().size()!=1||view.outputs().size()!=1)throw std::runtime_error("One input/output required");
    auto boundary=view.inputs()[0].tensorType();
    TensorType target=boundary;
    std::vector<TensorType> chain;
    std::vector<int64_t> chain_ids;
    if(mode=="encode"){
      if(!args.count("--value"))throw std::runtime_error("Encode requires compiled value ID");
      auto id=std::stoll(args.at("--value"));bool found=false;
      for(auto op:net->ops)for(auto value:op->outputs)if(value->v_id==id){target=value.tensorType();found=true;}
      if(!found)throw std::runtime_error("Unknown compiled value ID");
      if(args.count("--prefix-values")){
        std::stringstream list(args.at("--prefix-values"));std::string token;
        while(std::getline(list,token,',')){
          auto vid=std::stoll(token);chain_ids.push_back(vid);chain.push_back(net.getValueById(vid).tensorType());
        }
        if(chain_ids.empty()||chain_ids.back()!=id)throw std::runtime_error("Prefix must finish at target value");
      }else{chain.push_back(target);chain_ids.push_back(id);}
    }
    auto shape=target->shape;
    // The codec only accepts a logical unpadded [1,40,100] boundary.
    // Device-reordered/padded tensors require a separate explicitly tested ABI.
    if(shape.size()!=3||shape[0]!=1||shape[1]!=40||shape[2]!=100)throw std::runtime_error("Unsupported codec target shape");
    auto fp=TensorType(FloatType::FP32(),{1,40,100},target->layout);
    Tensor input(fp);input.mallocOn(HostDevice::MemRegion());
    auto converter=Backend::Create<HostBackend>(view,HostDevice::Default());
    Session session;
    if(mode=="forward"){session=Session::Create<HostBackend>(view,{HostDevice::Default()});session.apply();}
    std::filesystem::create_directories(args.at("--out"));
    auto base=std::filesystem::path(args.at("--out"));
    std::ifstream data(args.at("--inputs"),std::ios::binary);
    if(!data)throw std::runtime_error("Input open failed");
    std::ofstream output(base/(mode=="forward"?"logits.f32":"encoded.sqb"),std::ios::binary);
    std::ofstream decoded,hardware,reconstructed;
    if(mode=="encode"){decoded.open(base/"decoded.f32",std::ios::binary);hardware.open(base/"encoded.hqb",std::ios::binary);reconstructed.open(base/"sdk_dequantized.f32",std::ios::binary);}
    std::ofstream timings(base/"timings.jsonl");
    std::array<float,4000> values;size_t count=0,sqb_size=0,hqb_size=0;
    for(;;){
      data.read(reinterpret_cast<char*>(values.data()),sizeof(values));
      if(data.gcount()==0)break;
      if(data.gcount()!=sizeof(values))throw std::runtime_error("Truncated input tensor");
      for(auto x:values)if(!std::isfinite(x))throw std::runtime_error("Nonfinite input");
      auto start=Clock::now();input.write(0,reinterpret_cast<char*>(values.data()),sizeof(values));
      Tensor cast;
      if(mode=="encode"){
        cast=input;TensorType previous=fp;
        for(auto type:chain){cast=icraft::hostbackend::utils::Cast(cast,previous,type,converter);previous=type;}
      }else cast=icraft::hostbackend::utils::Cast(input,fp,target,converter);
      if(mode=="encode"){
        std::ostringstream a(std::ios::out|std::ios::binary),b(std::ios::out|std::ios::binary),c(std::ios::out|std::ios::binary);
        cast.dump(a,"SQB");cast.dump(b,"SFB");cast.dump(c,"HQB");
        auto aa=a.str(),bb=b.str(),cc=c.str();
        if(count&& (aa.size()!=sqb_size||cc.size()!=hqb_size))throw std::runtime_error("Variable encoded size");
        sqb_size=aa.size();hqb_size=cc.size();output.write(aa.data(),aa.size());decoded.write(bb.data(),bb.size());hardware.write(cc.data(),cc.size());
        auto back=icraft::hostbackend::utils::Cast(cast,target,fp,converter);
        std::ostringstream dd(std::ios::out|std::ios::binary);back.dump(dd,"SFB");auto d=dd.str();reconstructed.write(d.data(),d.size());
      }else{
        auto results=session.forward({cast});
        if(results.size()!=1)throw std::runtime_error("Wrong output count");
        if(!results[0].waitForReady(std::chrono::milliseconds(120000),std::chrono::milliseconds(1)))throw std::runtime_error("Host output timeout");
        std::ostringstream bytes(std::ios::out|std::ios::binary);results[0].dump(bytes,"SFB");auto text=bytes.str();
        if(text.size()!=3*sizeof(float))throw std::runtime_error("Wrong logits byte count");
        output.write(text.data(),text.size());
      }
      auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
      timings<<Json{{"index",count},{"host_elapsed_ns",ns}}.dump()<<"\n";
      ++count;if(count%64==0)std::cerr<<"Host processed "<<count<<"\n";
    }
    Json chain_json=Json::array();
    for(size_t i=0;i<chain.size();++i){
      Json entry={{"value_id",chain_ids[i]},{"dtype_bytes",chain[i].bytes()}};
      auto scale=chain[i].getScale();
      if(scale.has_value()){
        auto s=scale.value();Json a=Json::array(),bits=Json::array();
        for(auto value:s->origin_scale){uint32_t b;std::memcpy(&b,&value,4);a.push_back(value);bits.push_back(b);}
        entry["scale_axis"]=s->axis;entry["scales"]=a;entry["scale_float32_bits"]=bits;
      }
      chain_json.push_back(entry);
    }
    std::ofstream report(base/"summary.json");
    report<<Json{{"status","completed"},{"backend","Host"},{"board_accessed",false},{"count",count},
       {"mode",mode},{"sqb_bytes_per_tensor",sqb_size},{"hqb_bytes_per_tensor",hqb_size},
       {"target_dtype_bytes",target.bytes()},{"cast_chain",chain_json}}.dump(2)<<"\n";
    return 0;
  }catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}
}
