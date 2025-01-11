#!/usr/bin/env python3

import sys
import re
from fabric import Connection
from pathlib import Path, PurePosixPath
import tarfile
import shlex
# from sshtunnel import SSHTunnelForwarder
import threading
import time

if len(sys.argv) != 2:
    print("Usage: ./remote_setup 'ssh -p <PORT> <USER>@<HOST>'")
    sys.exit(1)

cmd = sys.argv[1]
# Expected format: ssh -p <PORT> <USER>@<HOST>
pattern = r"^ssh\s+-p\s+(\d+)\s+([^@]+)@(.+)$"
match = re.match(pattern, cmd)
if not match:
    print("Error: Command format doesn't match expected pattern.")
    sys.exit(1)

PORT = int(match.group(1))
USER = match.group(2)
HOST = match.group(3)

print("PORT:", PORT)
print("USER:", USER)
print("HOST:", HOST)
# This is not elegant
KEY_FILE = Path('~/.ssh/tensordock2')

SERVER_PROJECT_NAME = 'server_code'
DATA_DOWNLOADER = Path(__file__).parent /'cached_fineweb10B_aria.py'
SERVER_DATA_DOWNLOADER = PurePosixPath('cached_fineweb10B_aria.py')
PROJECT_ROOT = Path(__file__).parent.parent
SERVER_CODE = PROJECT_ROOT / SERVER_PROJECT_NAME
BUILD_DIR = PROJECT_ROOT / 'build'
SERVER_CODE_ARCHIVE = BUILD_DIR / f'{SERVER_PROJECT_NAME}.tar.gz'
REMOTE_SERVER_CODE_ARCHIVE = PurePosixPath(f'{SERVER_CODE_ARCHIVE.name}')
REMOTE_SERVER_PROJECT_ROOT = PurePosixPath(f'{SERVER_PROJECT_NAME}')
ACTIVATE_SCRIPT = Path("~/server_code/.venv/bin/activate")
HOME = Path('~/')
BASHRC = HOME / '.bashrc'


def start_tunnel():
    tunnel = SSHTunnelForwarder(
        (HOST, PORT),
        ssh_username=USER,
        ssh_pkey=str(KEY_FILE),
        remote_bind_address=('127.0.0.1', 6006),
        local_bind_address=('127.0.0.1', 6006),
    )
    tunnel.start()
    while tunnel.is_active:
        time.sleep(1)

# TODO
# use `ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -f -N -L 6006:localhost:6006 -i ~/.ssh/tensordock2 -p PORT USER@HOST` for now
# tunnel_thread = threading.Thread(target=start_tunnel, daemon=True)
# tunnel_thread.start()
# input("press enter to end tunnel\n")

BUILD_DIR.mkdir(exist_ok=True)
with tarfile.open(SERVER_CODE_ARCHIVE, 'w:gz') as tar:
    tar.add(SERVER_CODE, SERVER_PROJECT_NAME)
print('ziped')





conn = Connection(host=HOST, user=USER, port=PORT, connect_kwargs={'key_filename': str(KEY_FILE.expanduser().resolve())})
print('connected')
conn.run('whoami')
print(SERVER_CODE_ARCHIVE, REMOTE_SERVER_CODE_ARCHIVE)

conn.put(str(SERVER_CODE_ARCHIVE), str(REMOTE_SERVER_CODE_ARCHIVE))

print('copied')
conn.run(f'tar -xvf {shlex.quote(str(REMOTE_SERVER_CODE_ARCHIVE))}')
conn.run('sudo apt-get update')
conn.run('sudo apt-get install -y python3.10-venv aria2')
conn.run('cd server_code; python3 -m venv .venv; source .venv/bin/activate; pip install -r requirements.txt')

print(str(DATA_DOWNLOADER))
print(str(SERVER_DATA_DOWNLOADER))
conn.put(str(DATA_DOWNLOADER), str(SERVER_DATA_DOWNLOADER))
conn.run(f'python3 {shlex.quote(str(SERVER_DATA_DOWNLOADER))}')
conn.run(f'echo "source {str(ACTIVATE_SCRIPT.expanduser().resolve())}" >> {str(BASHRC.expanduser().resolve())}')
