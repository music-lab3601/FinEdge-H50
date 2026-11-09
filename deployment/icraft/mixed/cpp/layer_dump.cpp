// Host-only intermediate diagnostics. No device/socket backend.
#include <icraft-xir/core/network.h>
#include <icraft-xir/utils/json.hpp>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <map>
#include <array>
#include <cmath>
#include <iostream>
using namespace icraft::xir;
using namespace icraft::xrt;
using Json=nlohmann::json;
int main(int argc,char**argv){try{
  if(argc!=5)throw std::runtime_error("json raw inputs.f32 outdir");
  auto net=Network::CreateFromJsonFile(argv[1]);net.loadParamsFromFile(argv[2]);auto view=net.view(0);
  auto boundary=view.inputs()[0].tensorType();
  auto fp=TensorType(FloatType::FP32(),{1,40,100},boundary->layout);
  auto converter=Backend::Create<HostBackend>(view,HostDevice::Default());
  auto session=Session::Create<HostBackend>(view,{HostDevice::Default()});session.apply();
  Tensor input(fp);input.mallocOn(HostDevice::MemRegion());
  std::filesystem::path out(argv[4]);std::filesystem::create_directories(out);
  std::ofstream meta(out/"tensors.jsonl");size_t sample=0;
  session.setPostCallBack([&](const Session&,const Operation&op,const Backend&,std::vector<Tensor>&outputs){
    for(size_t i=0;i<outputs.size();++i){
      auto t=outputs[i];auto dtype=t.dtype();
      auto fp_type=TensorType(FloatType::FP32(),dtype->shape,dtype->layout,dtype->merged_distrs);
      auto decoded=icraft::hostbackend::utils::Cast(t,dtype,fp_type,converter);
      auto name="s"+std::to_string(sample)+"_op"+std::to_string(op->op_id)+"_"+std::to_string(i)+".f32";
      std::ofstream f(out/name,std::ios::binary);decoded.dump(f,"SFB");
      Json shape=Json::array();for(auto d:dtype->shape)shape.push_back(d);
      meta<<Json{{"sample",sample},{"op_id",op->op_id},{"output",i},{"type",std::string(op->typeKey())},{"shape",shape},{"file",name}}.dump()<<"\n";
    }
  });
  std::ifstream data(argv[3],std::ios::binary);std::array<float,4000> x;
  while(data.read(reinterpret_cast<char*>(x.data()),sizeof(x))){
    for(auto v:x)if(!std::isfinite(v))throw std::runtime_error("Nonfinite");
    input.write(0,reinterpret_cast<char*>(x.data()),sizeof(x));
    auto cast=icraft::hostbackend::utils::Cast(input,fp,boundary,converter);session.forward({cast});++sample;
  }
  if(data.gcount()!=0)throw std::runtime_error("Truncated input");
  std::ofstream result(out/"result.json");result<<Json{{"status","completed"},{"samples",sample},{"board_accessed",false}}.dump(2);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
