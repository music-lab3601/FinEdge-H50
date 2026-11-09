from pathlib import Path
import re,sys,unittest
U=Path(__file__).resolve().parents[1];sys.path.insert(0,str(U))
from console_login import ensure_shell

class Serial:
    def __init__(self):self.writes=[];self.reset=False
    def reset_input_buffer(self):self.reset=True
    def write(self,data):self.writes.append(data)
class Console:
    def __init__(self,replies):self.serial=Serial();self.pending=b'old';self.replies=iter(replies);self.timeouts=[]
    def until(self,pattern,timeout):
        self.timeouts.append(timeout)
        raw=next(self.replies);match=re.search(pattern,raw)
        if not match:raise TimeoutError('Expected console state absent')
        return raw,match

class LoginTest(unittest.TestCase):
    def test_cold_boot_login_prompt_is_resolved_before_shell_commands(self):
        c=Console([b'\r\nU login: ',b'Password: ',b'root@U:~# ']);result=ensure_shell(c)
        self.assertTrue(result['login_performed']);self.assertEqual(len(c.serial.writes),3)
        self.assertTrue(c.serial.reset);self.assertEqual(c.pending,b'')
    def test_existing_shell_never_receives_another_login_sequence(self):
        c=Console([b'\r\nroot@U:~# ']);result=ensure_shell(c)
        self.assertFalse(result['login_performed']);self.assertEqual(c.serial.writes,[b'\r'])
    def test_linux_boot_messages_are_allowed_before_a_real_login_prompt(self):
        c=Console([b'[  OK  ] Started Network Name Resolution\r\nU login: ',b'Password: ',b'root@U:~# '])
        self.assertTrue(ensure_shell(c)['login_performed']);self.assertEqual(c.timeouts[0],120)

if __name__=='__main__':unittest.main()
