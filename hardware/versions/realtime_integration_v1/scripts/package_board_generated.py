"""Package a completed routed candidate with its real timing/DRC evidence."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,gzip,time
V=Path(__file__).resolve().parents[1];P=V.parents[1];HW=P/'hardware/wujing_fpai'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--cdc-recipe',type=Path,help='Explicit G coverage correction stored under this build; original frozen recipe remains intact')
    a=ap.parse_args();out=a.build.resolve()
    recipe=a.cdc_recipe.resolve() if a.cdc_recipe else out/'network_cdc_frozen.tcl'
    if not recipe.is_relative_to(out):raise ValueError('CDC recipe must be frozen under this build')
    r={**json.loads((out/'result.json').read_text()),**json.loads((out/'execution_result.json').read_text())};project=Path(r['project']);runs=project.parent/'fpai_demo_vivado.runs/impl_1'
    if r['status']!='build_finished_requires_validation':raise RuntimeError('A completed candidate build is required')
    audit=out/'physical_audit'
    if audit.exists():audit=out/('physical_audit_'+time.strftime('%Y%m%d_%H%M%S'))
    audit.mkdir()
    # Keep the exact final constraints with the audit. Late-added upstream MAC
    # synchronizer rules alter analysis/placement metadata only, not logic or
    # routing; ordinary data paths must still close on the real routed design.
    cdc=recipe.read_text().replace(
        '[file normalize [file join [file dirname [info script]] ../vendor/verilog-ethernet]]',
        '{'+(V/'board_sources/vendor/verilog-ethernet').as_posix()+'}')
    (audit/'applied_cdc.tcl').write_text(cdc)
    tcl=f'''open_checkpoint {{{(out/'market_routed.dcp').as_posix()}}}
source {{{(audit/'applied_cdc.tcl').as_posix()}}}
# 750ps additional PHY RX delay. The physical FPGA clock/data routes are kept;
# change the external device configuration with its matching launch-edge model.
# Previous 1.95ns PHY skew caused real RX hold violations on these routed pins.
# 4ns half-cycle minus 0.75ns PHY skew, with +/-1ns external uncertainty.
set_input_delay -clock market_rgmii_rx -max 4.25 [get_ports {{PL_PHY_RXD[*] PL_PHY_RX_CTRL}}]
set_input_delay -clock market_rgmii_rx -min 2.25 [get_ports {{PL_PHY_RXD[*] PL_PHY_RX_CTRL}}]
set_input_delay -clock market_rgmii_rx -clock_fall -add_delay -max 4.25 [get_ports {{PL_PHY_RXD[*] PL_PHY_RX_CTRL}}]
set_input_delay -clock market_rgmii_rx -clock_fall -add_delay -min 2.25 [get_ports {{PL_PHY_RXD[*] PL_PHY_RX_CTRL}}]
set f [open {{{(audit/'slack.txt').as_posix()}}} w]
foreach type {{max min}} {{
 set p [get_timing_paths -delay_type $type -max_paths 1]
 puts $f "$type [get_property SLACK $p] [get_property STARTPOINT_PIN $p] [get_property ENDPOINT_PIN $p]"
}}
close $f
report_route_status -file {{{(audit/'route.txt').as_posix()}}}
report_timing -max_paths 20 -file {{{(audit/'critical_paths.txt').as_posix()}}}
report_pulse_width -all_violators -limit 100 -file {{{(audit/'pulse_width.txt').as_posix()}}}
set bc [get_cells -quiet -hier -filter {{NAME =~ *frontend/core/cycle_reg*}}]
set bp [get_pins -quiet -of_objects $bc -filter {{REF_PIN_NAME == C}}]
set bk [get_clocks -quiet -of_objects $bp]
if {{[llength $bk]!=1}} {{error "G book clock coverage differs"}}
set period [get_property PERIOD $bk]
if {{abs($period-8.0)>0.0001}} {{error "G expected 125MHz book clock"}}
set bf [open {{{(audit/'book_clock.txt').as_posix()}}} w]
puts $bf "125000000 Hz; PERIOD $period ns; $bk"
close $bf
report_clocks -file {{{(audit/'clocks.txt').as_posix()}}}
check_timing -override_defaults {{no_clock unconstrained_internal_endpoints}} -file {{{(audit/'clock_coverage.txt').as_posix()}}}
report_exceptions -ignored -file {{{(audit/'ignored_exceptions.txt').as_posix()}}}
exit
'''
    (audit/'audit.tcl').write_text(tcl)
    env = dict(os.environ)
    procise = Path(env['FMSH_PROCISE_PATH'])
    vivado = Path(env['FINEDGE_VIVADO'])
    env['PATH'] = str(procise/'dll') + os.pathsep + str(procise/'bin') + os.pathsep + env['PATH']
    with (audit/'console.txt').open('wb') as f:
        proc=subprocess.run([str(vivado),'-mode','batch','-source',str(audit/'audit.tcl')],cwd=audit,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
    if proc.returncode:raise RuntimeError('Audit tool failed')
    slack={row.split()[0]:float(row.split()[1]) for row in (audit/'slack.txt').read_text().splitlines()}
    if slack['max']<0 or slack['min']<0:raise RuntimeError('Ordinary setup/hold timing has not closed: '+repr(slack))
    if re.search(r'(?im)^\s*\|.*\|\s*Error\s*\|',(out/'drc.txt').read_text()):raise RuntimeError('DRC errors remain')
    route=(audit/'route.txt').read_text()
    errors=re.search(r'nets with routing errors\.*\s*:\s*(\d+)',route)
    if not errors or int(errors[1]):raise RuntimeError('Routing errors or missing coverage')
    coverage=(audit/'clock_coverage.txt').read_text()
    for pattern in (r'There are (\d+) register/latch pins with no clock\.',r'There are (\d+) pins that are not constrained for maximum delay\.'):
        count=re.search(pattern,coverage)
        if not count or int(count[1]):raise RuntimeError('Active clock/timing coverage is incomplete')
    pulse=(audit/'pulse_width.txt').read_text()
    rows=[line.strip() for line in pulse.splitlines() if re.match(r'^(Min|Max)\s+',line)]
    allowed={
        'U0_ps_ai_peripheral/ZGolo/u_AI_Mate_top/clkrst_wrapper_u0/AI_CORE_CLK/inst/clkout1_buf/I':(1.409,1.000,-0.409),
        'U0_ps_ai_peripheral/ZGolo/u_AI_Mate_top/clkrst_wrapper_u0/AI_CORE_CLK/inst/plle2_adv_inst/CLKOUT0':(1.071,1.000,-0.071)}
    if len(rows)!=2:raise RuntimeError('Unexpected pulse-width/minimum-period violations: '+repr(rows))
    observed=set()
    for row in rows:
        parts=row.split();pin=parts[-1]
        if not row.startswith('Min Period ') or pin not in allowed:
            raise RuntimeError('New pulse-width/minimum-period violation: '+row)
        actual=tuple(float(x) for x in parts[-5:-2])
        if actual!=allowed[pin]:raise RuntimeError('Known vendor minimum-period exception changed: '+row)
        observed.add(pin)
    if observed!=set(allowed):raise RuntimeError('Vendor exception coverage')
    bit=runs/'ai7030_edif_top_disable_icap.bit'
    if not bit.is_file():raise RuntimeError('Vendor disable_icap bit is required')
    dest=out/'boot';dest.mkdir()
    manifest=json.loads((P/'artifacts/fpga_reference_20261021/boot_candidate_UNVERIFIED_20261021/manifest.json').read_text())
    for row in manifest['inputs']:
        if row['name'].endswith('.bit'):continue
        src=HW/'vendor/BOOT_Gen'/row['name']
        if sha(src)!=row['sha256']:raise RuntimeError('Bootloader identity changed')
        shutil.copy2(src,dest/src.name)
    shutil.copy2(bit,dest/bit.name)
    bif='the_ROM_image:\n{\n'+''.join(' '+tag+(dest/name).as_posix()+'\n' for tag,name in (
        ('[bootloader, destination_device = ps]','FSBL251210_ddr400_demo.out'),('[destination_device = pl]',bit.name),
        ('[destination_cpu = apu_0, destination_device = ps]','bl31.elf'),('[destination_cpu = apu_0, destination_device = ps]','u-boot')))+' [fsbl_config]apu_x32\n}\n'
    (dest/'market.bif').write_text(bif)
    (dest/'package.tcl').write_text('if {[catch {create_boot_image {'+(dest/'market.bif').as_posix()+'} {'+(dest/'BOOT.bin').as_posix()+'}} e]} {puts stderr $e;exit 1}\nputs MARKET_BOOT_PACKAGED\nexit\n')
    env.update(PROCISE_DIR=str(procise),TCL_LIBRARY=str(procise/'tcl8.4'),ICTIME_HOME=str(procise),APP_DIR=str(procise),FMSH_DB=str(procise/'db'))
    with (dest/'console.txt').open('wb') as f:proc=subprocess.run([str(procise/'bin/procise.exe'),str(dest/'package.tcl')],cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    if proc.returncode or 'MARKET_BOOT_PACKAGED' not in (dest/'console.txt').read_text():raise RuntimeError('BOOT packaging failed')
    boot=dest/'BOOT.bin';(dest/'BOOT.bin.gz').write_bytes(gzip.compress(boot.read_bytes(),compresslevel=6))
    release=dict(status='packaged_pending_board_validation',slack_ns=slack,boot_sha256=sha(boot),
                 bit_sha256=sha(bit),compressed_sha256=sha(dest/'BOOT.bin.gz'),
                 hardware_acceptance=False,new_PHY_timing_assumptions_validated_on_board=False,
                 vendor_general_DDR_drain_proven=False,audit=str(audit),source_build=str(out))
    release['vendor_only_minimum_period_exceptions']=sorted(observed)
    release['final_CDC_constraints_sha256']=sha(audit/'applied_cdc.tcl')
    release['final_CDC_recipe']=str(recipe)
    release['CDC_scope']='Held-data mailboxes, Gray counters, and upstream MAC/FIFO first-stage synchronizers only; no new whole-clock waiver'
    release['PHY_RX_extra_delay_ps']=750
    release['PHY_RX_external_input_delay_ns']={'min':2.25,'max':4.25,'reference':'preceding DDR launch edge'}
    release['PHY_RX_profile_scope']='Candidate external +/-1ns bounds; 750ps PHY skew required at deployment, actual packet validation and PVT signoff separate'
    (dest/'release.json').write_text(json.dumps(release,indent=2));print(json.dumps(release))
if __name__=='__main__':main()
