import subprocess
from subprocess import run
from pathlib import Path


DATAROOT = Path('~').expanduser()
COMMIT = '889765ea1f903759787add96995d81171b632d0c'
DOWNLOAD_LIST = DATAROOT / 'download_list.txt'
FINEWEB10B = DATAROOT / 'fineweb10B'

def get_aria():
    run(['sudo', 'apt-get', 'update'], check=True)
    run(['sudo', 'apt-get', 'install', '-y', 'aria2'], check=True)


def build_downloads_list(total_train_shards = 103, data_root=DATAROOT):
    fineweb10B_urls = {
        f"https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/{COMMIT}/fineweb_{shard}.bin?download=true": 
        f"fineweb_{shard}.bin"
        for shard in [f'train_{n:06d}' for n in range(1, total_train_shards + 1)] + [f'val_{0:06d}']
    }
    DATAROOT.mkdir(exist_ok= True)
    with open(DOWNLOAD_LIST, 'w') as f:
        for url, filename in fineweb10B_urls.items():
            f.write(f"{url}\n")
            f.write(f"\tout={filename}\n")


    

def aria_download(connections_per_server=16, parallel_connections=16, parallel_downloads=16, download_list=str(DOWNLOAD_LIST)):
    aria2c_command = [
        'aria2c',
        '-x', '16',        # max connections per server
        '-s', '16',        # parallel downloads per server
        '-j', str(parallel_downloads),  # max parallel downloads
        '-i', download_list,
        '-d', str(FINEWEB10B)
    ]
    subprocess.run(aria2c_command, check=True)


if __name__ == '__main__':
    # get_aria()
    build_downloads_list()
    print()
    aria_download()
