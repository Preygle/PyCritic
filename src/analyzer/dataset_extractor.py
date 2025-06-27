import os
import tarfile

def safe_extract(tar, path="."):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
    tar.extractall(path)


tar_file = "py150.tar.gz"


with tarfile.open(tar_file) as tar:
    safe_extract(tar)

json_file = "python100k_train.json"

with open(json_file, 'r') as f:
    for _ in range(1):  # Read first 1 lines 
        print(f.readline())
