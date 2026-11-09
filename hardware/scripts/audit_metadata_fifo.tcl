# Run after applying network_cdc.tcl to the final design.
set report [open $finedge_fifo_audit_file w]
puts $report "fifo\twrite_clock_pins\tcapture_registers\trequirement_ns\tslack_ns\thold_exception\tstartpoint\tendpoint"
set checked 0
foreach box [get_cells -hier -filter {ORIG_REF_NAME =~ wt_async_fifo* || REF_NAME =~ wt_async_fifo*}] {
    if {[string match {*payload_fifo} $box]} {continue}
    set ram [get_cells -hier -filter "NAME =~ ${box}/mem_reg* && (REF_NAME =~ RAM* || REF_NAME =~ FD*)"]
    set launch [get_pins -of_objects $ram -filter {REF_PIN_NAME == CLK || REF_PIN_NAME == C}]
    set outputs [get_pins -of_objects $ram -filter {DIRECTION == OUT}]
    set capture [get_cells -hier -filter "NAME =~ ${box}/r_data_reg* && REF_NAME =~ FD*"]
    set path [get_timing_paths -delay_type max -from $launch -through $outputs -to $capture -max_paths 1]
    if {[llength $path] != 1} {error "Missing metadata RAM publication path $box"}
    set requirement [get_property REQUIREMENT $path]
    if {abs($requirement - 5.0) > 0.0001} {error "Metadata RAM publication requires a 5ns datapath bound"}
    if {[get_property SLACK $path] < 0} {error "Metadata RAM publication exceeds 5ns"}
    # Explicit -from/-through queries also return false paths. Inspect the
    # exception property to verify the asynchronous hold relationship.
    set hold [get_timing_paths -delay_type min -from $launch -through $outputs -to $capture -max_paths 1]
    if {[llength $hold] != 1 || [get_property EXCEPTION $hold] != "False Path"} {
        error "Metadata RAM asynchronous hold exception is missing"
    }
    puts $report [join [list $box [llength $launch] [llength $capture] $requirement \
        [get_property SLACK $path] [get_property EXCEPTION $hold] \
        [get_property STARTPOINT_PIN $path] [get_property ENDPOINT_PIN $path]] "\t"]
    incr checked
}
close $report
if {$checked != 2} {error "Expected exactly two metadata RAM publication checks"}
puts FINEDGE_METADATA_FIFO_AUDIT_PASSED
