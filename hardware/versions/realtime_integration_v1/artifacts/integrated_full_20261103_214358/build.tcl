if {![info exists ::env(FINEDGE_PYTHON)]} {error "Use 03_检查与配置/build_fpga.py to configure the Python and tool environment"}
set_param general.maxThreads 4
open_project {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/fpai_demo_vivado.xpr}
foreach n {user_ddr_window_ctrl_v1.sv user_ddr_window_engine_v1.sv window_core_v1.sv window_ddr_integrated_cdc.xdc user_ddr_window_por_input_v1.xdc} {
 foreach f [get_files -quiet *$n] {remove_files $f}
}
add_files -norecurse [glob {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/rtl/bench/*.sv}]
foreach n {fm_level_ram fm_level_book fm_mem_cdc fm_mem_arbiter fm_network} {
 add_files -norecurse "@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/rtl/bench/upstream/$n.sv"
}
set_property include_dirs {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/rtl/bench} [get_filesets sources_1]
add_files -norecurse [glob {@HARDWARE_ROOT@/versions/realtime_integration_v1/board_sources/vendor/verilog-ethernet/rtl/*.v}]
add_files -norecurse [glob {@HARDWARE_ROOT@/versions/realtime_integration_v1/board_sources/vendor/verilog-ethernet/lib/axis/rtl/*.v}]
foreach n {fm_pl_clocks} {
 read_ip "@HARDWARE_ROOT@/versions/fpga_market_v1/artifacts/fp_ip/fp_ip.srcs/sources_1/ip/$n/$n.xci"
}
add_files -fileset constrs_1 -norecurse {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/rtl/bench/market_io.xdc}
set_property STEPS.OPT_DESIGN.TCL.POST {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/opt_post.tcl} [get_runs impl_1]
set_property STEPS.WRITE_BITSTREAM.TCL.PRE {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/ip_patch/process_control/write_bitstream_pre.tcl} [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Default [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
update_compile_order -fileset sources_1
reset_run synth_1
launch_runs synth_1 -jobs 2 -scripts_only
# The vendor pre-hook opens its parent XPR. Do not let the parent Vivado save
# that same XML concurrently: finish script generation, close, then execute.
close_project
exec $::env(FINEDGE_PYTHON) {@HARDWARE_ROOT@/versions/realtime_integration_v1/scripts/run_integrated_vivado_step.py} --directory {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358} --step synth_1 >@ stdout 2>@ stderr
open_project {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/fpai_demo_vivado.xpr}
wait_on_run synth_1
if {![string match "*synth_design Complete*" [get_property STATUS [get_runs synth_1]]]} {error "Market synthesis failed"}
open_run synth_1
report_utilization -hierarchical -hierarchical_depth 3 -file {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/synth_utilization.txt}
write_checkpoint {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/market_synth.dcp}
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 2 -scripts_only
close_project
exec $::env(FINEDGE_PYTHON) {@HARDWARE_ROOT@/versions/realtime_integration_v1/scripts/run_integrated_vivado_step.py} --directory {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358} --step impl_1 >@ stdout 2>@ stderr
open_project {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/fpai_demo_vivado.xpr}
wait_on_run impl_1
if {![string match "*write_bitstream Complete*" [get_property STATUS [get_runs impl_1]]]} {error "Market implementation incomplete"}
open_run impl_1
report_timing_summary -delay_type min_max -report_unconstrained -file {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/timing.txt}
report_utilization -hierarchical -hierarchical_depth 3 -file {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/utilization.txt}
report_drc -file {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/drc.txt}
report_cdc -file {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/cdc.txt}
report_io -file {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/io.txt}
write_checkpoint {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/market_routed.dcp}
report_utilization -file {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/flat_utilization.txt}
set_operating_conditions -ambient_temp 25 -process typical
report_power -file {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/power_default.rpt}
puts MARKET_BUILD_DONE_REQUIRES_VALIDATION
close_project
