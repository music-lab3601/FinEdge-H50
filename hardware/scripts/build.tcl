# Materialized by finedge.py fpga. Sources and outputs belong to this build.
set build {__BUILD__}
set project {__PROJECT__}
set source [file join $build hardware]
set_param general.maxThreads 4
open_project $project
set_property include_dirs [file join $source rom] [get_filesets sources_1]
set_property STEPS.OPT_DESIGN.TCL.POST [file join $source scripts opt_post.tcl] [get_runs impl_1]
set_property STEPS.WRITE_BITSTREAM.TCL.PRE [file join $build project ip_patch process_control write_bitstream_pre.tcl] [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Default [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
update_compile_order -fileset sources_1
reset_run synth_1
launch_runs synth_1 -jobs 2 -scripts_only
close_project
exec $::env(FINEDGE_PYTHON) -B -m hardware.scripts.run_integrated_vivado_step --directory $build --step synth_1 >@ stdout 2>@ stderr
open_project $project
wait_on_run synth_1
if {![string match "*synth_design Complete*" [get_property STATUS [get_runs synth_1]]]} {error "H50 synthesis failed"}
open_run synth_1
report_utilization -file [file join $build synth_utilization.txt]
write_checkpoint [file join $build market_synth.dcp]
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 2 -scripts_only
close_project
exec $::env(FINEDGE_PYTHON) -B -m hardware.scripts.run_integrated_vivado_step --directory $build --step impl_1 >@ stdout 2>@ stderr
open_project $project
wait_on_run impl_1
if {![string match "*write_bitstream Complete*" [get_property STATUS [get_runs impl_1]]]} {error "H50 implementation failed"}
open_run impl_1
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $build timing.txt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $build utilization.txt]
report_drc -file [file join $build drc.txt]
report_cdc -file [file join $build cdc.txt]
report_io -file [file join $build io.txt]
write_checkpoint [file join $build market_routed.dcp]
report_utilization -file [file join $build flat_utilization.txt]
set_operating_conditions -ambient_temp 25 -process typical
report_power -file [file join $build power_default.rpt]
puts MARKET_BUILD_DONE_REQUIRES_VALIDATION
close_project
