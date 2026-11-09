// Direct integer bytes enter Tensor::write. No Float32 Cast adapter in forward mode.
#include <icraft-xir/core/network.h>
#include <icraft-xir/utils/json.hpp>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <map>
#include <vector>
using namespace icraft::xir;
using namespace icraft::xrt;
using Json=nlohmann::json;
int main(int argc,char**argv){try{
  std::map<std::string,std::string> a;for(int i=1;i+1<argc;i+=2)a[argv[i]]=argv[i+1];
  for(auto k:{"--json","--raw","--inputs","--out","--mode"})if(!a.count(k))throw std::runtime_error("Missing argument");
  auto mode=a.at("--mode");if(mode!="forward"&&mode!="encode")throw std::runtime_error("Host only mode forward or encode");
  auto net=Network::CreateFromJsonFile(a.at("--json"));net.loadParamsFromFile(a.at("--raw"));auto view=net.view(0);
  if(view.inputs().size()!=1)throw std::runtime_error("One input required");
  auto dtype=view.inputs()[0].tensorType();
  if((dtype->shape.size()!=3&&dtype->shape.size()!=5)||dtype->shape[0]!=1||dtype.numElements()<4000||dtype.numElements()>6400)throw std::runtime_error("Unsupported entry shape");
  auto fp=TensorType(FloatType::FP32(),dtype->shape,dtype->layout,dtype->merged_distrs);
  auto converter=Backend::Create<HostBackend>(view,HostDevice::Default());
  Tensor input(mode=="encode"?fp:dtype);input.mallocOn(HostDevice::MemRegion());
  Session session;if(mode=="forward"){session=Session::Create<HostBackend>(view,{HostDevice::Default()});session.apply();}
  std::filesystem::path out(a.at("--out"));std::filesystem::create_directories(out);
  std::ofstream binary(out/(mode=="forward"?"logits.f32":"sdk_encoded.bin"),std::ios::binary),decoded;
  if(mode=="encode")decoded.open(out/"sdk_decoded.f32",std::ios::binary);
  std::ifstream f(a.at("--inputs"),std::ios::binary);size_t bytes=mode=="encode"?fp.bytes():dtype.bytes(),count=0;
  std::vector<char> buffer(bytes);
  while(f.read(buffer.data(),buffer.size())){
    input.write(0,buffer.data(),buffer.size());
    if(mode=="encode"){
      auto q=icraft::hostbackend::utils::Cast(input,fp,dtype,converter);
      std::vector<char> raw(dtype.bytes());q.read(raw.data(),0,raw.size());binary.write(raw.data(),raw.size());
      auto back=icraft::hostbackend::utils::Cast(q,dtype,fp,converter);
      raw.resize(fp.bytes());back.read(raw.data(),0,raw.size());decoded.write(raw.data(),raw.size());
    }else{
      auto y=session.forward({input});if(y.size()!=1)throw std::runtime_error("Wrong output count");
      std::ostringstream stream(std::ios::out|std::ios::binary);y[0].dump(stream,"SFB");auto data=stream.str();
      if(data.size()!=12)throw std::runtime_error("Not three FP32 logits");binary.write(data.data(),data.size());
    }++count;
  }
  if(f.gcount()!=0)throw std::runtime_error("Truncated input");
  Json scales=Json::array(),zeros=Json::array();auto scale=dtype.getScale();auto z=dtype.getZeroPoints();
  if(scale.has_value())for(auto s:scale.value()->origin_scale)scales.push_back(s);
  if(z.has_value())for(auto p:z.value())zeros.push_back(p);
  std::ofstream summary(out/"summary.json");summary<<Json{{"status","completed"},{"mode",mode},{"backend","Host"},
    {"input_bytes",dtype.bytes()},{"source_bytes_per_tensor",bytes},{"count",count},{"scales",scales},{"zero_points",zeros},
    {"direct_integer_write",mode=="forward"},{"runtime_fp32_input_adapter",false},{"board_accessed",false}}.dump(2);
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
