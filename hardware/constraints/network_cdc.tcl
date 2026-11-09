# Apply after synthesis of the complete G-owned integration design.
# This is a scoped constraint recipe, not a claim that coverage was signed off.
foreach box [get_cells -hier -filter {ORIG_REF_NAME =~ b1_mailbox* || REF_NAME =~ b1_mailbox*}] {
 set first [get_cells -quiet -regexp "${box}/(ack_sync_reg|request_sync_reg)\\\[0\\\]"]
 if {[llength $first]} {set_false_path -to [get_pins -of_objects $first -filter {REF_PIN_NAME == D}]}
 # phys_opt may clone data registers as *_reg[n]_rep / *_replica.
 # Match sequential endpoints including those replicas, never only exact[n].
 set hold [get_cells -quiet -hier -filter "NAME =~ ${box}/hold_reg* && (REF_NAME =~ FD* || REF_NAME =~ LD*)"]
 set capture [get_cells -quiet -hier -filter "NAME =~ ${box}/m_data_reg* && (REF_NAME =~ FD* || REF_NAME =~ LD*)"]
 if {[llength $hold] && [llength $capture]} {
  set bundled_budget_ns 5.0
  # G monitor request: a 125MHz destination captures held data only after two
  # request synchronizer edges (>=16ns). Bound the data route to one 8ns
  # destination period, leaving another period for setup/CDC margin.
  # This applies only to the acknowledged immutable 12-bit request payload.
  if {[string match {*control/request_box} $box]} {set bundled_budget_ns 8.0}
  set_max_delay -datapath_only $bundled_budget_ns -from $hold -to $capture
  set_false_path -hold -from $hold -to $capture
 }
}
set first [get_cells -hier -quiet -regexp {.*(enable200|fault125)_reg\[0\]$}]
if {[llength $first]} {set_false_path -to [get_pins -of_objects $first -filter {REF_PIN_NAME == D}]}
set async_regs [get_cells -quiet -hier -filter {ASYNC_REG == TRUE}]
set async_pins [get_pins -quiet -of_objects $async_regs -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
if {[llength $async_pins]} {set_false_path -to $async_pins}
set eth {repo://third_party/verilog-ethernet}
source $eth/lib/axis/syn/vivado/axis_async_fifo.tcl
source $eth/syn/vivado/rgmii_phy_if.tcl
proc fb_source_mac_cdc {path} {
 set fd [open $path r];set rules [read $fd];close $fd
 regsub -all {\$inst/([[:alnum:]_/]+)\[([0-9*]+)\](/C)?} $rules {[format {%s/\1[\2]\3} $inst]} rules
 uplevel 1 $rules
}
fb_source_mac_cdc $eth/syn/vivado/eth_mac_fifo.tcl
fb_source_mac_cdc $eth/syn/vivado/eth_mac_1g_rgmii.tcl
fb_source_mac_cdc $eth/syn/vivado/eth_mac.tcl
# G must check matched first stages / held-data endpoints and MAC reports.
# Do not add set_clock_groups -asynchronous to bypass these datapath bounds.

# G capture Gray pointers: bound source-to-first-stage skew and delay.
foreach pair {{wgray wg1} {rgray rg1}} {
 lassign $pair src dst
 set a [get_cells -quiet -hier -filter "NAME =~ *capture/${src}_reg*"]
 set b [get_cells -quiet -hier -filter "NAME =~ *capture/${dst}_reg*"]
 if {[llength $a] && [llength $b]} {
  set_max_delay -datapath_only 8.0 -from $a -to $b
  set_bus_skew 8.0 -from $a -to $b
  set_false_path -hold -from $a -to $b
 }
}
set first [get_cells -hier -quiet -regexp {.*fault_sync_reg\[0\]$}]
if {[llength $first]} {set_false_path -to [get_pins -of_objects $first -filter {REF_PIN_NAME == D}]}
# Only the first metastability stage may be asynchronous. The synchronized
# output and all book state remain fully timed in the 125 MHz domain.
set first [get_cells -hier -quiet -regexp {.*stop_sync_reg\[0\]$}]
if {[llength $first]} {set_false_path -to [get_pins -of_objects $first -filter {REF_PIN_NAME == D}]}

set first [get_cells -hier -quiet -regexp {.*(predict_control_sync|tensor_fault_sync|overflow_sync|datapath_fault_sync)_reg\[0\]$}]
if {[llength $first]!=3} {error "G integrated fault synchronizer coverage mismatch (window FIFO eliminates overflow_sync)"}
set_false_path -to [get_pins -of_objects $first -filter {REF_PIN_NAME == D}]

# Exact Gray-pointer crossing constraints; second synchronizer stages remain
# normally timed. There is no whole-clock-group exception.
set fifos [get_cells -hier -quiet -filter {ORIG_REF_NAME =~ wt_async_fifo* || REF_NAME =~ wt_async_fifo*}]
if {[llength $fifos]!=3} {error "Expected exactly three whole-window CDC FIFOs"}
foreach box $fifos {
 foreach pair {{wg wg1} {rg rg1}} {
  lassign $pair a b
  set src [get_cells -hier -quiet -filter "NAME =~ ${box}/${a}_reg* && REF_NAME =~ FD*"]
  set dst [get_cells -hier -quiet -filter "NAME =~ ${box}/${b}_reg* && REF_NAME =~ FD*"]
  if {![llength $src] || ![llength $dst]} {error "Missing Gray FIFO endpoints $box $a $b"}
  set_max_delay -datapath_only 5.0 -from $src -to $dst
  set_bus_skew 5.0 -from $src -to $dst
  set_false_path -hold -from $src -to $dst
  puts "WT_FIFO_CDC $box $a -> $b source=[llength $src] target=[llength $dst] budget=5ns"
 }
 # The small command/completion LUTRAMs hold immutable entries. Their read
 # register loads only after the two-stage Gray empty check. Bound the exact
 # RAM-to-prefetch path rather than exempting either clock domain.
 if {![string match {*payload_fifo} $box]} {
  set ram [get_cells -hier -quiet -filter "NAME =~ ${box}/mem_reg* && (REF_NAME =~ RAM* || REF_NAME =~ FD*)"]
  # Use the decomposed RAMD32/RAMS32 clock pins. The enclosing RAM32M WCLK
  # port is not a valid timing startpoint in the Vivado 2018.3 model.
  set launch [get_pins -quiet -of_objects $ram -filter {REF_PIN_NAME == CLK || REF_PIN_NAME == C}]
  set data_out [get_pins -quiet -of_objects $ram -filter {DIRECTION == OUT}]
  set reader [get_cells -hier -quiet -filter "NAME =~ ${box}/rb_reg* && REF_NAME =~ FD*"]
  set read_launch [get_pins -quiet -of_objects $reader -filter {REF_PIN_NAME == C}]
  set capture [get_cells -hier -quiet -filter "NAME =~ ${box}/r_data_reg* && REF_NAME =~ FD*"]
  if {![llength $ram] || ![llength $launch] || ![llength $read_launch] || ![llength $data_out] || ![llength $capture]} {error "Missing FIFO metadata RAM clock/output/capture $box"}
  # Vivado exposes RAMD32 CLK->O paths as well as read-address->O paths.
  # A newly written word is readable only after wg1, wg2 and the registered
  # fetch condition. At 125MHz this gives >=16ns before capture. The writer
  # cannot reuse the slot until the read pointer returns through rg1/rg2.
  # Bound write publication to 5ns and exempt only its asynchronous hold
  # relationship. Read-address paths retain their ordinary hold checks.
  set_max_delay -datapath_only 5.0 -from $launch -through $data_out -to $capture
  set_false_path -hold -from $launch -through $data_out -to $capture
  set_max_delay -datapath_only 5.0 -from $read_launch -through $data_out -to $capture
  puts "WT_FIFO_DATA_CDC $box write_clocks=[llength $launch] outputs=[llength $data_out] capture=[llength $capture] budget=5ns"
 }
}

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
