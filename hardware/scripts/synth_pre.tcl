# Keep the vendor hook unchanged and preserve its original error details.
set vendor_hook {vendor://fpga/ip_patch/process_control/synthesis_pre.tcl}
if {[catch {source $vendor_hook} detail options]} {
    puts stderr "FINEDGE_VENDOR_SYNTH_PRE_FAILED $detail"
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    return -options $options $detail
}
