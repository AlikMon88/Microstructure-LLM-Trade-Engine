import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

# AWS details
aws_user = "ubuntu"

secret_key_personal = os.path.join(os.getcwd(), "src/keys/order_book_key.pem")

aws_host_personal = "ec2-16-171-236-88.eu-north-1.compute.amazonaws.com"

remote_dir = "/home/ubuntu/TradeGPT/quanta/"
local_dir = os.path.join(os.getcwd(), "quanta")  # local quanta folder next to notebook

# Ensure local directory exists
os.makedirs(local_dir, exist_ok=True)

files_to_copy = ["orderbook_data/", "rolling_ob.log"]

print('working-dir: ', os.getcwd())

def get_rolling_ob_aws():
    for f in files_to_copy:
        remote_path = os.path.join(remote_dir, f)
        local_path = os.path.join(local_dir, f)
        if f.endswith("/"):  # directory
            cmd = f'scp -i "{secret_key_personal}" -r "{aws_user}@{aws_host_personal}:{remote_path}" "{local_dir}"'
        else:  # single file
            cmd = f'scp -i "{secret_key_personal}" "{aws_user}@{aws_host_personal}:{remote_path}" "{local_path}"'
        print(f"Copying {f} from AWS...")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Failed to copy {f} from AWS")

if __name__ == '__main__':
    get_rolling_ob_aws()
