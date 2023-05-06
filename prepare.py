import os
import config
sd_scripts_root_path =  config.sd_scripts_root_path

file_path = os.path.join(sd_scripts_root_path, "train_network.py")
if not os.path.isfile(file_path):
    raise Exception(file_path + ' file not exists')

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()


new_lines = ['import train_preview\n']
for line in lines:
    new_lines.append(line)
    if 'unwrapped_nw.save_weights' in line:
        new_lines.append('        train_preview.on_train_finish(ckpt_file, epoch_no, force_sync_upload)\n')

with open('./train_network_yf.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("saved to train_network_yf.py")