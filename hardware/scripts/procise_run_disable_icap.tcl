if {![info exists ::env(FINEDGE_BITSTREAM)]} {error "Set FINEDGE_BITSTREAM to the copied bitstream before postprocessing"}
set_device JFMQL30TAI676H
increment_bit_cfg_v3 $::env(FINEDGE_BITSTREAM) -disable_icap
