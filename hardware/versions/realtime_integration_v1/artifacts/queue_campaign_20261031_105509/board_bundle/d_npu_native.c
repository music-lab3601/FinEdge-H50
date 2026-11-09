/* G model qualification. Register/DMA sequences match the qualified
 * native/shared_fast_mmio.c; no FPGA producer, register scan or reset. */
#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
static volatile uint32_t *gp;
static unsigned char *ps;
static uint32_t ps_addr;
static inline void fence(void){__asm__ volatile("dsb sy" ::: "memory");}
static uint64_t ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
int dq_open(uint32_t physical){
 if(physical<0x30100000u||physical>0x38100000u-16384||(physical&4095))return -1;
 int fd=open("/dev/mem",O_RDWR|O_SYNC);if(fd<0)return -errno;
 void *p=mmap(0,0x100000,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0x40000000);close(fd);if(p==MAP_FAILED)return -errno;gp=p;
 fd=open("/dev/udmabuf0",O_RDWR|O_SYNC);if(fd<0)return -errno;
 p=mmap(0,134217728,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);if(p==MAP_FAILED)return -errno;
 ps=(unsigned char*)p+(physical-0x30100000u);ps_addr=physical;return 0;
}
uint32_t dq_count(void){fence();return gp[0x40c64/4];}
uint32_t dq_reg(uint32_t off){fence();return gp[off/4];}
static int transfer(uint32_t src,uint32_t dst,uint32_t bytes,uint32_t direction){
 if(!bytes||bytes>8192||(bytes&63)||(src&63)||(dst&63))return -3;
 gp[0x40518/4]=src;gp[0x4051c/4]=src+bytes-64;
 gp[0x40520/4]=dst;gp[0x40524/4]=dst+bytes-64;
 gp[0x40514/4]=direction;fence();gp[0x40504/4]=1;fence();gp[0x40504/4]=0;fence();
 uint64_t end=ns()+1000000000;
 for(uint32_t n=0;;n++){uint32_t s=gp[0x40584/4];if(s&10)return -2;if(!(s&5)){fence();return 0;}if(!(n&1023)&&ns()>end)return -1;}
}
int dq_input(uint32_t destination,const void *data,uint32_t bytes){
 if(!ps||!bytes||bytes>16000||(bytes&63)||(destination&63)||destination>0xffffffffu-bytes)return -3;
 for(uint32_t off=0;off<bytes;off+=8192){uint32_t n=bytes-off;if(n>8192)n=8192;
  memcpy(ps,(const unsigned char*)data+off,n);fence();int r=transfer(ps_addr,destination+off,n,2);if(r)return r;}
 return 0;
}
int dq_output(uint32_t source,void *data){
 if(!ps||(source&63))return -3;int r=transfer(source,ps_addr+8192,64,1);if(r)return r;
 fence();memcpy(data,ps+8192,12);return 0;
}
int dq_launch(uint32_t master,uint32_t index,uint32_t state,uint32_t layers,uint32_t *counts){
 if(master!=1||index>127||!layers||layers>1024)return -3;
 volatile uint32_t *wait=gp+(0x40000+((6<<7)+index+10)*4)/4;
 volatile uint32_t *start=gp+(0x40000+((master<<11)+9)*4)/4;
 uint32_t base=dq_count();counts[0]=base;*wait=state;fence();*start=1;fence();
 uint64_t end=ns()+1000000000;
 for(uint32_t n=0;;n++){uint32_t current=dq_count();uint32_t delta=current-base;
  if(delta>=layers&&delta<0x80000000u){counts[1]=current;return delta==layers?0:-4;}
  if(!(n&1023)&&ns()>end)return -1;}
}
