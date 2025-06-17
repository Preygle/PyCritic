import os
import tarfile

# def is_within_directory(directory, target):
#     abs_dir = os.path.abspath(directory)
#     abs_target = os.path.abspath(target)
#     return os.path.commonpath([abs_dir]) == os.path.commonpath([abs_dir, abs_target])


# def safe_extract(tar, path="."):
#     for member in tar.getmembers():
#         member_path = os.path.join(path, member.name)
#         if not is_within_directory(path, member_path):
#             raise Exception("Path traversal detected in tar file")
#     tar.extractall(path)


# with tarfile.open("py150.tar_1") as tar:
#     safe_extract(tar)

with open('python50k_eval.json', 'r') as f:
    for _ in range(1):  # Read first 1 lines 
        print(f.readline())
