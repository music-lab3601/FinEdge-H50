// Pure packing of G's 416-byte r1 ABI. Capture only on local book handshake.
// The caller supplies captured timestamps; this combinational encoder has no FIFO.
module fb_bookrow_r1(
 input wire [63:0] session_id,stream_id,day_ns,local_accept_ticks,local_commit_ticks,clock_hz,
 input wire [255:0] identity,input wire [2559:0] values,
 output wire [3327:0] frame
);
 assign frame={values,128'd0,clock_hz,local_commit_ticks,local_accept_ticks,
  identity[255:192]+day_ns+64'd18000000000000,
  identity[191:128],identity[127:64],identity[63:0],32'd0,stream_id[31:0],session_id,16'd1,16'd1,32'h314b4246};
endmodule
