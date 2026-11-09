source {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/ip_patch/process_control/implementation_opt_post.tcl}
source {@HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/network_cdc_signoff_mcp_v5.tcl}
set_input_delay -clock market_rgmii_rx -max 4.25 [get_ports {PL_PHY_RXD[*] PL_PHY_RX_CTRL}]
set_input_delay -clock market_rgmii_rx -min 2.25 [get_ports {PL_PHY_RXD[*] PL_PHY_RX_CTRL}]
set_input_delay -clock market_rgmii_rx -clock_fall -add_delay -max 4.25 [get_ports {PL_PHY_RXD[*] PL_PHY_RX_CTRL}]
set_input_delay -clock market_rgmii_rx -clock_fall -add_delay -min 2.25 [get_ports {PL_PHY_RXD[*] PL_PHY_RX_CTRL}]
