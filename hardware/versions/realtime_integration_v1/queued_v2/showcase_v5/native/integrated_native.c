/* G production hot loop. Original SDK initialization stays in Python; no
 * Python callback, tensor copy, full input readback or output DMA per window. */
#include "d_npu_native.c"
#include <pthread.h>
#include <sched.h>
#include <stdio.h>

uint32_t sf_read(uint32_t off){return dq_reg(off);}
void sf_write(uint32_t off,uint32_t value){gp[off/4]=value;fence();}
int sf_open(void){
 if(gp)return 0;int fd=open("/dev/mem",O_RDWR|O_SYNC);if(fd<0)return -errno;
 void *p=mmap(0,0x100000,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0x40000000);close(fd);
 if(p==MAP_FAILED)return -errno;gp=p;return 0;
}
int gi_snapshot(uint32_t base,uint32_t index,uint64_t *value){
 uint32_t seq=sf_read(base+0x30);sf_write(base+0x30,index);uint64_t end=ns()+1000000000ull;
 while(sf_read(base+0x30)==seq)if(ns()>end)return -1;
 *value=(uint64_t)sf_read(base+0x34)|((uint64_t)sf_read(base+0x38)<<32);return 0;
}
int gi_book_snapshot(uint32_t index,uint64_t *value){
 uint32_t seq=sf_read(0xd00a0);sf_write(0xd0088,index);uint64_t end=ns()+1000000000ull;
 while(sf_read(0xd00a0)==seq)if(ns()>end)return -1;
 *value=(uint64_t)sf_read(0xd00a8)|((uint64_t)sf_read(0xd00ac)<<32);return 0;
}
typedef struct {
 uint64_t window,ordinal,slot,generation,request_ns,npu_ns,board_ticks,service_ticks;
 uint32_t logits[3],class_index,count_before,count_after;
} gi_trace;
_Static_assert(sizeof(gi_trace)==88,"G trace ABI");
typedef struct {
 uint64_t window,ordinal,generation,slot;
 unsigned char input[4800];float logits[3];uint32_t class_index;
} gi_audit;
_Static_assert(sizeof(gi_audit)==4848,"G audit ABI");
static gi_audit *audit;
static uint32_t audit_bases[32],audit_capacity,audit_count;
void gi_audit_set(const uint32_t *bases,uint32_t slots,void *buffer,uint32_t count){
 if(slots>32)slots=32;memcpy(audit_bases,bases,slots*4);audit=buffer;audit_capacity=count;audit_count=0;
}
int gi_read_bytes(uint32_t address,void *buffer,uint32_t size){
 unsigned char *dst=buffer;
 for(uint32_t off=0;off<size;off+=8192){uint32_t n=size-off;if(n>8192)n=8192;
  int rc=transfer(address+off,ps_addr,n,1);if(rc)return rc;fence();memcpy(dst+off,ps,n);
 }return 0;
}
uint32_t gi_audit_count(void){return audit_count;}
/* Per-slot recipe: instruction[3],sync,layers,MMU mode/table/config. */
static uint32_t last_instruction=UINT32_MAX,last_instruction_size;
static uint32_t last_mmu_mode=UINT32_MAX,last_mmu_table=UINT32_MAX,last_mmu_config=UINT32_MAX;
static uint64_t last_launch_wait_ns;
int gi_dispatch(const uint32_t *r,uint32_t *counts){
 if(r[1]||!r[0]||(r[0]&127)||!r[2]||(r[2]&127)||r[4]!=30||r[5]>1)return -20;
 /* VPE decoded state must also be invalidated when translation changes. */
 int mapping_changed=last_mmu_mode!=r[5]||last_mmu_table!=r[6]||last_mmu_config!=r[7];
 if(mapping_changed){
  sf_write(0x40840,r[5]);
  if(r[5]){sf_write(0x40844,r[6]);sf_write(0x40848,r[7]);sf_write(0x4084c,15);sf_write(0x4084c,0);}
  last_mmu_mode=r[5];last_mmu_table=r[6];last_mmu_config=r[7];
 }
 if(mapping_changed||last_instruction!=r[0]||last_instruction_size!=r[2]){
  sf_write(0x42000,0);sf_write(0x42028,1);sf_write(0x42018,r[0]);sf_write(0x42020,r[2]);sf_write(0x44028,0);
  last_instruction=r[0];last_instruction_size=r[2];
 }
 uint64_t begin=ns();int rc=dq_launch(1,r[3],0,r[4],counts);
 last_launch_wait_ns=ns()-begin;return rc;
}
/* GUI-only 5Hz monitor. The native worker remains the sole GP reader. */
static uint64_t *gui_monitor;
void gi_monitor_set(uint64_t *values){gui_monitor=values;}
uint64_t gi_trace_count(const uint64_t *statistics){return __atomic_load_n(statistics+2,__ATOMIC_ACQUIRE);}
int gi_monitor_copy(uint64_t *values){
 if(!gui_monitor)return -1;
 uint64_t seq=__atomic_load_n(gui_monitor,__ATOMIC_ACQUIRE);if(seq&1)return 1;
 for(unsigned i=0;i<12;i++)values[i]=__atomic_load_n(gui_monitor+i,__ATOMIC_RELAXED);
 __atomic_thread_fence(__ATOMIC_ACQUIRE);
 return seq==__atomic_load_n(gui_monitor,__ATOMIC_ACQUIRE)?0:1;
}
static int gi_monitor_sample(void){
 static const uint32_t ids[8]={1,2,3,5,7,29,28,27};
 uint64_t values[8];
 for(unsigned i=0;i<8;i++)if(gi_snapshot(0xd2000,ids[i],values+i))return -22;
 uint64_t seq=__atomic_load_n(gui_monitor,__ATOMIC_RELAXED);
 __atomic_store_n(gui_monitor,seq+1,__ATOMIC_RELEASE);
 __atomic_store_n(gui_monitor+1,ns(),__ATOMIC_RELAXED);
 for(unsigned i=0;i<8;i++)__atomic_store_n(gui_monitor+i+2,values[i],__ATOMIC_RELAXED);
 __atomic_store_n(gui_monitor,seq+2,__ATOMIC_RELEASE);return 0;
}
int gi_run(const uint32_t *recipes,gi_trace *trace,uint32_t capacity,
           uint64_t first,uint64_t last,volatile uint32_t *stop,uint64_t timeout_ms,
           uint64_t *statistics,uint32_t active_slots){
 if(active_slots==0||active_slots>32)return -21;
 cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(2,&mask);
 if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask))return -10;
 uint64_t end=ns()+timeout_ms*1000000ull,all_launch=0,all_results=0,saved=0;
 uint64_t prior_generation=0,request_stamp=0,npu_elapsed=0;
 uint32_t pending=0,req[8]={0},counts[2]={0},words[12]={0};
 int rc=0;uint64_t next_monitor=0;
 for(;;){
  uint32_t flags=sf_read(0xd2008);
  if(*stop&&!pending&&!(flags&3)){
   /* One GP reader owns all runtime MMIO, including finite drain. Python only
    * requests FINISH; it must not race a second CPU's status reads against the
    * native request/result loop through the register response mux. */
   uint64_t bflags=0,books=0,rows=0,encoded=0,admitted=0,retired=0,contexts_slots=0;
   if(gi_book_snapshot(8,&bflags)){rc=-17;break;}
   if(bflags&1){rc=-18;break;}
   if(bflags&16){
    if(gi_book_snapshot(0x48,&books)||gi_snapshot(0xd2000,1,&rows)||
       gi_snapshot(0xd2000,12,&encoded)||gi_snapshot(0xd2000,3,&admitted)||
       gi_snapshot(0xd2000,5,&retired)||gi_snapshot(0xd2000,16,&contexts_slots)){rc=-19;break;}
    if(rows==books&&encoded==rows&&admitted==retired&&contexts_slots==0)break;
   }
  }
  if(flags&16){rc=-11;break;}
  if(flags&2){
   for(unsigned i=0;i<12;i++)words[i]=sf_read(0xd2180+i*4);
   uint64_t window=(uint64_t)words[4]|((uint64_t)words[5]<<32);
   uint64_t ordinal=(uint64_t)words[6]|((uint64_t)words[7]<<32);
   uint32_t reason=words[3]>>24;
   int matched=pending&&window==((uint64_t)req[5]|((uint64_t)req[6]<<32));
   if((!reason&&!matched)||reason>4||(words[3]&255)>2){rc=-12;break;}
   if(matched)for(unsigned i=0;i<audit_count;i++)if(audit[i].generation==prior_generation){
    audit[i].ordinal=ordinal;memcpy(audit[i].logits,words,12);audit[i].class_index=words[3];
   }
   if(ordinal>=first&&ordinal<=last){
    if(saved>=capacity){rc=-13;break;}
    gi_trace *t=&trace[saved++];memset(t,0,sizeof(*t));
    t->window=window;t->ordinal=ordinal;t->slot=matched?req[0]:UINT32_MAX;t->generation=matched?((uint64_t)req[1]|((uint64_t)req[2]<<32)):0;
    t->request_ns=matched?request_stamp:0;t->npu_ns=matched?npu_elapsed:0;
    t->board_ticks=(uint64_t)words[8]|((uint64_t)words[9]<<32);
    t->service_ticks=(uint64_t)words[10]|((uint64_t)words[11]<<32);
    memcpy(t->logits,words,12);t->class_index=words[3];t->count_before=counts[0];t->count_after=counts[1];
    __atomic_store_n(statistics+2,saved,__ATOMIC_RELEASE);
   }
   sf_write(0xd202c,0x54414b45u);if(matched)pending=0;all_results++;
  }else if((flags&1)&&!pending){
   for(unsigned i=0;i<8;i++)req[i]=sf_read(0xd2100+i*4);
   uint64_t generation=(uint64_t)req[1]|((uint64_t)req[2]<<32);
   if(req[0]>=active_slots||generation<=prior_generation||req[7]!=30){rc=-14;break;}
   prior_generation=generation;const uint32_t *r=recipes+req[0]*8;
   if(r[4]!=30){rc=-15;break;}
   if(audit_count<audit_capacity){
    gi_audit *a=&audit[audit_count];memset(a,0,sizeof(*a));
    a->window=(uint64_t)req[5]|((uint64_t)req[6]<<32);a->generation=generation;a->slot=req[0];
    rc=transfer(audit_bases[req[0]],ps_addr,4800,1);if(rc)break;
    fence();memcpy(a->input,ps,4800);audit_count++;
   }
   request_stamp=ns();rc=gi_dispatch(r,counts);npu_elapsed=last_launch_wait_ns;
   if(rc)break;
   sf_write(0xd2028,0x444f4e45u);pending=1;all_launch++;
  }
  uint64_t now=ns();
  if(now>end){rc=-16;break;}
  if(gui_monitor&&!pending&&now>=next_monitor){
   rc=gi_monitor_sample();if(rc)break;next_monitor=now+200000000ull;
  }
 }
 statistics[0]=all_launch;statistics[1]=all_results;
 __atomic_store_n(statistics+2,saved,__ATOMIC_RELEASE);statistics[3]=pending;
 return rc;
}

/* Clause-22 management, configuration only. No market data uses MDIO. */
static void mdio_wait(void){uint64_t end=ns()+1000;while(ns()<end){}}
static void mdio_bit(uint32_t oe,uint32_t data,uint32_t clk){
 sf_write(0xd00fc,1|(clk<<1)|(data<<2)|(oe<<3));mdio_wait();
}
static void mdio_out(uint32_t b){mdio_bit(1,b,0);mdio_bit(1,b,1);}
/* Sample the stable low phase before raising MDC. On the actual YT8521,
 * sampling after the rising edge skips the first data bit (0x011a->0x0235).
 * Preserve both Z/0 turnaround cycles; the second low-phase sample is ACK. */
static uint32_t mdio_in(void){mdio_bit(0,1,0);uint32_t v=(sf_read(0xd00fc)>>4)&1;mdio_bit(0,1,1);return v;}
static void mdio_bits(uint32_t v,int n){for(int i=n-1;i>=0;i--)mdio_out((v>>i)&1);}
static void mdio_header(uint32_t phy,uint32_t reg,uint32_t op){
 for(int i=0;i<32;i++){mdio_out(1);}
 mdio_bits(1,2);mdio_bits(op,2);mdio_bits(phy,5);mdio_bits(reg,5);
}
int mf_mdio_read(uint32_t phy,uint32_t reg){
 if(phy>31||reg>31)return -1;
 mdio_header(phy,reg,2);mdio_in();uint32_t noack=mdio_in(),v=0;
 for(int i=0;i<16;i++){v=(v<<1)|mdio_in();}
 mdio_bit(0,1,0);
 return noack?-2:(int)v;
}
int mf_mdio_write(uint32_t phy,uint32_t reg,uint32_t data){
 if(phy>31||reg>31||data>65535)return -1;
 mdio_header(phy,reg,1);mdio_bits(2,2);mdio_bits(data,16);mdio_bit(0,1,0);return 0;
}
int mf_phy_reset(void){sf_write(0xd00fc,0);usleep(10000);sf_write(0xd00fc,1);usleep(100000);return 0;}
