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
  set launch [get_pins -quiet -of_objects $ram -filter {REF_PIN_NAME == WCLK || REF_PIN_NAME == CLK || REF_PIN_NAME == C}]
  set data_out [get_pins -quiet -of_objects $ram -filter {DIRECTION == OUT}]
  set reader [get_cells -hier -quiet -filter "NAME =~ ${box}/rb_reg* && REF_NAME =~ FD*"]
  set read_launch [get_pins -quiet -of_objects $reader -filter {REF_PIN_NAME == C}]
  set capture [get_cells -hier -quiet -filter "NAME =~ ${box}/r_data_reg* && REF_NAME =~ FD*"]
  if {![llength $ram] || ![llength $launch] || ![llength $read_launch] || ![llength $data_out] || ![llength $capture]} {error "Missing FIFO metadata RAM clock/output/capture $box"}
  # LUTRAM asynchronous read paths can launch from the read-address register.
  # Bound every actual path through its output, without inventing a clock-to-
  # output arc. Normal read-domain setup/hold remains timed.
  # In Vivado's distributed-RAM model WCLK is a write timing endpoint, not
  # a read-data launch point. Read address -> mux -> capture is timed here;
  # source-domain writes are normally timed and Gray publication guards CDC.
  set_max_delay -datapath_only 5.0 -from $read_launch -through $data_out -to $capture
  puts "WT_FIFO_DATA_CDC $box outputs=[llength $data_out] capture=[llength $capture] budget=5ns"
 }
}
