#pragma once
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include <cerrno>
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>

namespace fe {
using U=uint64_t;
#ifdef _WIN32
using Socket=SOCKET;
constexpr Socket INVALID=INVALID_SOCKET;
inline void net_init(){WSADATA w;if(WSAStartup(MAKEWORD(2,2),&w))throw std::runtime_error("Winsock startup");}
inline void close_socket(Socket s){closesocket(s);}
inline bool would_block(){int e=WSAGetLastError();return e==WSAEWOULDBLOCK;}
inline U now_ns(){static const U freq=[](){LARGE_INTEGER f;QueryPerformanceFrequency(&f);return U(f.QuadPart);}();LARGE_INTEGER t;QueryPerformanceCounter(&t);U v=t.QuadPart;return v/freq*1000000000ull+v%freq*1000000000ull/freq;}
inline void pin(unsigned core){if(core>=64||!SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1ull<<core)))throw std::runtime_error("Thread affinity unavailable");}
inline void relax(){YieldProcessor();}
#else
using Socket=int;
constexpr Socket INVALID=-1;
inline void net_init(){}
inline void close_socket(Socket s){close(s);}
inline bool would_block(){return errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR;}
inline U now_ns(){timespec t{};clock_gettime(CLOCK_MONOTONIC_RAW,&t);return U(t.tv_sec)*1000000000ull+U(t.tv_nsec);}
inline void pin(unsigned core){cpu_set_t set;CPU_ZERO(&set);if(core>=CPU_SETSIZE)throw std::runtime_error("CPU index");CPU_SET(core,&set);if(pthread_setaffinity_np(pthread_self(),sizeof set,&set))throw std::runtime_error("Thread affinity unavailable");}
inline void relax(){
#if defined(__aarch64__)
 __asm__ volatile("yield" ::: "memory");
#else
 std::this_thread::yield();
#endif
}
#endif
inline std::string quote(const std::string&s){std::string r="\"";for(unsigned char c:s){if(c=='\"'||c=='\\'){r+='\\';r+=char(c);}else if(c=='\n')r+="\\n";else if(c=='\r')r+="\\r";else if(c>=32)r+=char(c);}return r+'\"';}
struct Args {
 std::map<std::string,std::string> values;
 Args(int n,char**v){for(int i=1;i<n;i+=2){if(i+1>=n||std::string(v[i]).rfind("--",0)!=0)throw std::runtime_error("Expected --name value");if(!values.emplace(v[i]+2,v[i+1]).second)throw std::runtime_error("Duplicate option");}}
 std::string get(const std::string&k,const std::string&d="")const{auto p=values.find(k);return p==values.end()?d:p->second;}
 U num(const std::string&k,U d=0)const{auto s=get(k);if(s.empty())return d;if(s[0]=='-')throw std::runtime_error("Negative option "+k);size_t n=0;auto v=std::stoull(s,&n);if(n!=s.size())throw std::runtime_error("Invalid number "+k);return v;}
 void only(std::initializer_list<const char*> names)const{for(auto&p:values){bool found=false;for(auto n:names)found|=p.first==n;if(!found)throw std::runtime_error("Unknown option --"+p.first);}}
};
inline sockaddr_in endpoint(const std::string&host,unsigned port){sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(uint16_t(port));if(!port||port>65535||inet_pton(AF_INET,host.c_str(),&a.sin_addr)!=1)throw std::runtime_error("IPv4 endpoint");return a;}
inline Socket udp_socket(){Socket s=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);if(s==INVALID)throw std::runtime_error("UDP socket");
#ifdef _WIN32
 u_long nb=1;if(ioctlsocket(s,FIONBIO,&nb))throw std::runtime_error("Nonblocking UDP");
#else
 int f=fcntl(s,F_GETFL);if(f<0||fcntl(s,F_SETFL,f|O_NONBLOCK)<0)throw std::runtime_error("Nonblocking UDP");
#endif
 return s;}
inline int receive_buffer(Socket s,int requested){if(setsockopt(s,SOL_SOCKET,SO_RCVBUF,(const char*)&requested,sizeof requested))throw std::runtime_error("SO_RCVBUF");int actual=0;
#ifdef _WIN32
 int n=sizeof actual;
#else
 socklen_t n=sizeof actual;
#endif
 if(getsockopt(s,SOL_SOCKET,SO_RCVBUF,(char*)&actual,&n))throw std::runtime_error("Get SO_RCVBUF");return actual;}
inline void write_text(const std::filesystem::path&p,const std::string&text){std::ofstream f(p,std::ios::binary);if(!f||!(f<<text))throw std::runtime_error("Write "+p.string());}
inline void fresh_directory(const std::filesystem::path&p){if(p.empty()||!std::filesystem::create_directories(p))throw std::runtime_error("New output directory required");}
} // namespace fe
