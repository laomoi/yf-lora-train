
import config
import os
import subprocess
import json


def on_train_finish(path, epoch_no, force_sync_upload):
    print("on train finished", path)
    config_data = config.config_data
    lora_dir = os.path.abspath(os.path.dirname(path))
    lora_name = os.path.splitext( os.path.basename(path))[0]

    txt2image_tasks = []
    for section in config_data:
        if section != "path":
            add_section_task(txt2image_tasks, lora_name, section, config_data[section])
    print(txt2image_tasks)
    run_task(txt2image_tasks, lora_dir, "0")


def run_task(txt2image_tasks, lora_dir, list_lora_dir):
    cmd = [config.sd_python_path, config.batch_txt2image_path, "--lora_dir", lora_dir,  "--batch",
         json.dumps(txt2image_tasks), '--list_lora_dir', list_lora_dir, '--png_save_path',config.config_data['path']['png_save_path']]

    print(cmd)
    result = subprocess.run(cmd, encoding='utf-8')
    # print(result)
    if result is not None and result.returncode == 0:
        print("finish shell excuted")
    else:
        print("failed excuted shell")
def on_train_batch_finish(lora_dir):
    config_data = config.config_data
    txt2image_tasks = []
    # for path in path_list:
    #     lora_name = os.path.splitext(os.path.basename(path))[0]
    for section in config_data:
        if section != "path":
            add_section_task(txt2image_tasks, "", section, config_data[section])

    run_task(txt2image_tasks, lora_dir, "1")


def add_section_task(txt2image_tasks, lora_name,  section, config_section):
    weights = str.split(config_section['lora_weights'], ",")
    sampler_name = config_section['sampler_name']
    steps = config_section['steps']
    width = config_section['width']
    height = config_section['height']
    seed = config_section['seed']
    model = config_section['model']
    prompt = config_section['prompt']

    # prompts = []
    # for weight in weights:
    #     p = '"' + config_section['prompt'] + ' <lora:' + lora_name + ':' + weight + '>' + '"'
    #     prompts.append(p)
    # png_path = os.path.join(config.config_data['path']['png_save_path'], lora_name + '_' + section + '.png')
    txt2image_tasks.append({
        # 'png_path': png_path,
        'section':section,
        'prompt':prompt,
        'weights':weights,
        'sampler_name':sampler_name,
        'width': width,
        'height': height,
        'steps': steps,
        'seed': seed,
        'model': model,
        'lora_name': lora_name,
    })





