"""Bounded PS serial command/file transport; never used by the inference path."""
from pathlib import Path
import argparse, base64, hashlib, json, re, shlex, time, uuid, gzip
import serial

class Console:
    def __init__(self, port=None, baud=None):
        from shared.paths import load_settings
        cfg=load_settings()['board']
        self.serial = serial.Serial(port or cfg['serial_port'], baud or cfg['baud'], timeout=.1, write_timeout=5)
        self.pending = b''

    def close(self): self.serial.close()

    def until(self, pattern, timeout):
        buf = bytearray(self.pending)
        self.pending = b''
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            buf.extend(self.serial.read(max(1, self.serial.in_waiting)))
            match = re.search(pattern, bytes(buf))
            if match:
                self.pending = bytes(buf[match.end():])
                return bytes(buf[:match.end()]), match
        raise TimeoutError('Serial response deadline; tail='+repr(bytes(buf[-1200:])))

    def run(self, command, timeout=20):
        # Console must already be logged into the Linux shell. Commands carry
        # unique markers; do not mistake the echoed command for an actual reply.
        token = uuid.uuid4().hex.encode()
        self.serial.reset_input_buffer()
        self.pending = b''
        start = b'FE_BEGIN_'+token
        done = b'FE_END_'+token
        line = ('stty -echo; printf "\\n'+start.decode()+'\\n"; '
                +'sh -c '+shlex.quote(command)+'; rc=$?; printf "\\n'+done.decode()
                +':%s\\n" "$rc"; stty echo\r')
        self.serial.write(line.encode())
        raw, match = self.until(rb'\n'+done+rb':(\d+)\r*\n', timeout)
        begin = raw.find(b'\n'+start+b'\r')
        if begin < 0: begin = raw.find(b'\n'+start+b'\n')
        if begin < 0: raise RuntimeError('Missing command start marker')
        output = raw[begin+len(start)+1:match.start()].strip(b'\r\n')
        return dict(exit_code=int(match[1]), stdout=output.decode(errors='replace'))

    def put(self, local, remote, timeout=600):
        blob = Path(local).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        # Each line is <= 1,024 characters, below the tty canonical limit.
        # Receiver ACKs each line, so neither USB nor tty buffers can overrun.
        script = '''import sys,base64,hashlib,os
p=sys.argv[1];expected=sys.argv[2];n=int(sys.argv[3]);tmp=p+'.serial-part'
os.makedirs(os.path.dirname(os.path.abspath(p)),exist_ok=True)
h=hashlib.sha256();total=0
print('FE_PUT_READY',flush=True)
with open(tmp,'wb') as f:
 for i in range(n):
  b=base64.b64decode(sys.stdin.buffer.readline().strip(),validate=True);f.write(b);h.update(b);total+=len(b)
  print('FE_PUT_ACK:'+str(i),flush=True)
 f.flush();os.fsync(f.fileno())
if h.hexdigest()!=expected: raise ValueError('file SHA256 mismatch')
os.replace(tmp,p)
print('FE_PUT_DONE:'+str(total)+':'+h.hexdigest(),flush=True)
'''
        count = (len(blob)+767)//768
        cmd = 'python3 -u -c '+shlex.quote(script)+' '+shlex.quote(remote)+' '+digest+' '+str(count)
        # Base64 encodes the Python source so the shell command stays one line.
        encoded = base64.b64encode(script.encode()).decode()
        loader = "import base64;exec(base64.b64decode('"+encoded+"'))"
        cmd = 'python3 -u -c '+shlex.quote(loader)+' '+shlex.quote(remote)+' '+digest+' '+str(count)
        self.serial.reset_input_buffer()
        self.pending = b''
        self.serial.write(('stty -echo; '+cmd+'; stty echo\r').encode())
        self.until(rb'\nFE_PUT_READY\r*\n', 10)
        end = time.monotonic()+timeout
        for i in range(count):
            self.serial.write(base64.b64encode(blob[i*768:(i+1)*768])+b'\n')
            self.until(rb'\n?FE_PUT_ACK:'+str(i).encode()+rb'\r*\n', min(10, max(.1, end-time.monotonic())))
        raw, _ = self.until(rb'FE_PUT_DONE:'+str(len(blob)).encode()+b':'+digest.encode()+rb'\r*\n', 10)
        return dict(bytes=len(blob), sha256=digest, remote=remote)

    def get(self,remote,local,timeout=120):
        target=Path(local)
        if target.exists():raise FileExistsError(target)
        code='import base64,gzip,pathlib;print(base64.b64encode(gzip.compress(pathlib.Path('+repr(remote)+').read_bytes())).decode())'
        reply=self.run('python3 -c '+shlex.quote(code),timeout)
        if reply['exit_code']:raise RuntimeError(reply['stdout'])
        data=gzip.decompress(base64.b64decode(reply['stdout']))
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local=str(target))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--timeout', type=float, default=30)
    ap.add_argument('--exec', dest='command')
    ap.add_argument('--put', nargs=2, metavar=('LOCAL','REMOTE'))
    ap.add_argument('--get', nargs=2, metavar=('REMOTE','LOCAL'))
    a = ap.parse_args()
    c = Console(a.port)
    try:
        result = c.put(*a.put, timeout=a.timeout) if a.put else c.get(*a.get,timeout=a.timeout) if a.get else c.run(a.command, a.timeout)
        print(json.dumps(result, ensure_ascii=True))
    finally: c.close()

if __name__ == '__main__': main()
