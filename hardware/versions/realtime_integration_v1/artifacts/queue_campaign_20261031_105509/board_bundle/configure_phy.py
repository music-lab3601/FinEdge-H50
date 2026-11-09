"""YT8521 RX 750ps extra skew matched to routed BUFR/IDDR capture timing."""
from pathlib import Path
import argparse,ctypes,json,time

def configure(directory):
    lib=ctypes.CDLL(str(directory/'libintegrated.so'))
    lib.sf_read.argtypes=[ctypes.c_uint32];lib.sf_read.restype=ctypes.c_uint32
    lib.mf_mdio_read.argtypes=[ctypes.c_uint32]*2
    lib.mf_mdio_write.argtypes=[ctypes.c_uint32]*3
    if lib.sf_open():raise RuntimeError('MMIO')
    if lib.sf_read(0xd0000)!=0x31424746:raise RuntimeError('PL-market firmware not loaded')
    if lib.sf_read(0xd0008)&15:raise RuntimeError('Configure PHY before enabling producer')
    lib.mf_phy_reset();found=[]
    for phy in range(32):
        a,b=lib.mf_mdio_read(phy,2),lib.mf_mdio_read(phy,3)
        if a>=0 and b>=0 and (a,b) not in ((0,0),(65535,65535)):found.append((phy,a,b))
    # This board responds at its strapped address 3 and at address zero.
    # Prefer the sole nonzero address, only when zero reports the SAME PHY ID.
    candidates=[item for item in found if item[0]!=0]
    if len(candidates)==1 and all(item[1:]==candidates[0][1:] for item in found):
        phy,a,b=candidates[0]
    elif len(found)==1:phy,a,b=found[0]
    else:raise RuntimeError('Ambiguous PL PHY addresses: '+repr(found))
    if ((a<<16)|b)&0xfffffff0 != 0x00000110:raise RuntimeError('Unexpected YT8521 identity '+hex((a<<16)|b))
    def write(reg,value):
        if lib.mf_mdio_write(phy,reg,value):raise RuntimeError('MDIO write')
    def read(reg):
        v=lib.mf_mdio_read(phy,reg)
        if v<0:raise RuntimeError('MDIO read')
        return v
    def ext_read(reg):write(0x1e,reg);return read(0x1f)
    def ext_write(reg,value):write(0x1e,reg);write(0x1f,value)
    before={hex(r):ext_read(r) for r in (0xa000,0xa001,0xa003,0x27,0xc)}
    ext_write(0xa000,before['0xa000']&~2) # copper register space
    ext_write(0xa001,before['0xa001']&~0x107) # copper RGMII; no extra 1.9ns RX delay
    ext_write(0xa003,(before['0xa003']&~0x7cff)|(5<<10)) # RX 5*150ps; TX zero, no TX inversion
    ext_write(0x27,ext_read(0x27)&~0x8000);ext_write(0xc,ext_read(0xc)&~0x1000)
    pause_before=read(4)
    write(4,pause_before&~0x0c00)
    pause_after=read(4)
    if pause_after&0x0c00:raise RuntimeError('PL PAUSE advertisement not disabled')
    write(0,0x1200) # enable/restart autonegotiation
    until=time.monotonic()+10;status=0
    while time.monotonic()<until:
        read(1);status=read(1)
        if status&4:break
        time.sleep(.1)
    vendor=read(0x11);after={hex(r):ext_read(r) for r in (0xa000,0xa001,0xa003,0x27,0xc)}
    if after['0xa003']&0x7cff !=5<<10 or after['0xa001']&0x107:raise RuntimeError('PHY skew/mode readback differs')
    result=dict(status='link_up' if status&4 else 'configured_no_link',phy_address=phy,phy_id=hex((a<<16)|b),
                observed_addresses=found,bmsr=hex(status),vendor_status=hex(vendor),before=before,after=after,
                link_mbps={0:10,1:100,2:1000}.get((vendor>>14)&3),full_duplex=bool(vendor&0x2000),
                pause_before=hex(pause_before),pause_after=hex(pause_after),pause_advertised=bool(pause_after&0x0c00),
                RX_delay_ps=750,TX_delay='MAC 90 degree, PHY TX delay disabled',
                RX_capture='Routed FPGA insertion plus 750ps PHY skew; external delay profile 2.25..4.25 ns from preceding DDR launch edge',
                reference='Linux v6.6 drivers/net/phy/motorcomm.c')
    (directory/'phy.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    configure(ap.parse_args().directory)
