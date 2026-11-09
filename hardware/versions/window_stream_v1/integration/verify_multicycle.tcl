# Only expected-RAM -> verification control has a two-cycle contract.
# RPREF reads expected RAM at T0; RSEND may issue at T1; bus_pending gates any
# RWAIT payload_reply until T2 at the earliest. Expected data stays unchanged
# throughout the request/response. Incoming DDR data and payload write data
# retain their normal single-cycle setup/hold requirements.
set bridges [get_cells -hier -quiet -filter {ORIG_REF_NAME =~ g_payload_bridge* || REF_NAME =~ g_payload_bridge*}]
if {[llength $bridges]!=1} {error "Expected one whole-window payload engine"}
foreach box $bridges {
 set ram [get_cells -hier -quiet -filter "NAME =~ ${box}/expected_reg* && REF_NAME =~ RAMB*"]
 set src [get_pins -quiet -of_objects $ram -filter {REF_PIN_NAME == CLKARDCLK || REF_PIN_NAME == CLKBWRCLK}]
 set dst [get_cells -hier -quiet -filter "(NAME =~ ${box}/state_reg* || NAME =~ ${box}/index_reg* || NAME =~ ${box}/retries_reg*) && REF_NAME =~ FD*"]
 if {![llength $src] || [llength $dst]<3} {error "Missing expected-RAM/verification-control endpoints"}
 set clocks [get_clocks -quiet -of_objects $src]
 if {[llength $clocks]!=1 || abs([get_property PERIOD $clocks]-5.0)>0.0001} {error "Verification expected RAM must use 200MHz"}
 set ends [get_pins -quiet -of_objects $dst -filter {REF_PIN_NAME == C}]
 set end_clocks [get_clocks -quiet -of_objects $ends]
 if {[llength $end_clocks]!=1 || $end_clocks!=$clocks} {error "Verification control must share the RAM clock"}
 set_multicycle_path -setup 2 -from $src -to $dst
 set_multicycle_path -hold 1 -from $src -to $dst
 puts "WT_VERIFY_MCP2 RAM=[llength $ram] source_clocks=[llength $src] control_registers=[llength $dst] setup=2 hold=1"
}
