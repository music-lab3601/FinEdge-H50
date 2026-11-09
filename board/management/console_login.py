"""Enter the current board's established root console after a cold boot."""
import os
import re

def ensure_shell(console,boot_timeout=120):
    if not 10<=boot_timeout<=180:raise ValueError('Console boot timeout must be10..180 seconds')
    console.serial.reset_input_buffer();console.pending=b'';console.serial.write(b'\r')
    # An external software/power restart can still be printing systemd boot
    # messages. Wait for a real prompt; never send shell commands into boot.
    _,match=console.until(rb'(?:[^\r\n]+ login: ?|root@[^ \r\n]+:[^\r\n]*# ?)',boot_timeout)
    if b'login:' in match[0]:
        # Same public vendor image credentials used by G's existing rebooter.
        console.serial.write(b'root\r');console.until(rb'Password: ?',5)
        console.serial.write(os.environ['FINEDGE_BOARD_PASSWORD'].encode('utf-8') + b'\r');console.until(rb'root@[^ \r\n]+:[^\r\n]*# ?',10)
        return dict(status='shell_ready',login_performed=True)
    return dict(status='shell_ready',login_performed=False)
