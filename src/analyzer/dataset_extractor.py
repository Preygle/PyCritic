import os
import tarfile

# Function to safely extract tar files
def safe_extract(tar, path="."):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
    tar.extractall(path)

tar_file = "py150.tar.gz"

with tarfile.open(tar_file) as tar:
    safe_extract(tar)

json_file = "python100k_train.json"

# Open the JSON file and read the first line
with open(json_file, 'r') as f:
    print(f.readline())
