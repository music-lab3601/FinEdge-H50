"""Freeze the current Book implementation into a standalone offline extractor."""
import shutil,subprocess
from common2 import ROOT,BASE,PROJECT,SDK,PYTHON,read,save,sha,env

def main():
    source=PROJECT/'versions/network_fast_v1/native/fast.cpp'
    text=source.read_text(encoding='utf-8')
    book=text[text.index('struct Order{'):text.index('void wait_due(')]
    prefix='''#include <cstdint>
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
'''
    suffix='''
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
  std::cout<<"{\\"accepted\\":"<<accepted<<",\\"raw_consumed\\":"<<b.ordinal<<"}\\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}
'''
    (ROOT/'cpp/fresh_book.cpp').write_text(prefix+book+suffix,encoding='utf-8')
    frozen=PROJECT/'versions/network_fast_v1/native/frozen.h';shutil.copyfile(frozen,ROOT/'cpp/frozen.h')
    save(ROOT/'contracts/offline_book_sources.json',dict(source=str(source),source_sha256=sha(source),frozen_sha256=sha(frozen),
        purpose='offline fresh-holdout reconstruction; not a real-time benchmark',book_class_copied_without_logic_changes=True))
    cmake=BASE/'.venv/Scripts/cmake.exe';build=ROOT/'build';toolchain=PROJECT/'.tools/vs_buildtools'
    cmds=[[str(cmake),'-S',str(ROOT/'cpp'),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64',
           '-DCMAKE_GENERATOR_INSTANCE='+str(toolchain),'-DCMAKE_PREFIX_PATH='+str(SDK)],
          [str(cmake),'--build',str(build),'--config','Release','--parallel','1']]
    build_log=ROOT/'artifacts/build.log'
    if build_log.exists():build_log=ROOT/'artifacts/build_repaired.log'
    with build_log.open('wb') as f:
        for cmd in cmds:
            p=subprocess.run(cmd,cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT)
            if p.returncode:raise RuntimeError('See artifacts/build.log')
    print('Diagnostics and offline extractor built')
if __name__=='__main__':main()
