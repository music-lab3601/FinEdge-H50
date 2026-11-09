source {vendor://fpga/ip_patch/process_control/implementation_opt_post.tcl}
source {repo://hardware/constraints/network_cdc.tcl}
set_input_delay -clock market_rgmii_rx -max 4.25 [get_ports {PL_PHY_RXD[*] PL_PHY_RX_CTRL}]
set_input_delay -clock market_rgmii_rx -min 2.25 [get_ports {PL_PHY_RXD[*] PL_PHY_RX_CTRL}]
set_input_delay -clock market_rgmii_rx -clock_fall -add_delay -max 4.25 [get_ports {PL_PHY_RXD[*] PL_PHY_RX_CTRL}]
set_input_delay -clock market_rgmii_rx -clock_fall -add_delay -min 2.25 [get_ports {PL_PHY_RXD[*] PL_PHY_RX_CTRL}]
