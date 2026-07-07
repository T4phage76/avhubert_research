import urllib.request
import os
import tarfile
# tar File URL
url = 'https://www.dropbox.com/scl/fi/3yrnuyjfpjp9rv087ftyy/lrs2_v1.tar?rlkey=pxk2bido5lzsp5pb27okk5awx&st=a23odya5&dl=1'
datatset_name = 'LRS2_dataset'

# Unzip directory
unzip_dir = 'LRS_processing/LRS2'
if not os.path.exists(unzip_dir):
    os.makedirs(unzip_dir)

# download file (DELETE the .tar file to save storage)
zip_file_path = os.path.join(unzip_dir,datatset_name)
urllib.request.urlretrieve(url, zip_file_path)


def extract_all_files(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(extract_to)

extract_all_files(zip_file_path, unzip_dir)