
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
    run_task(txt2image_tasks, lora_dir)


def run_task(txt2image_tasks, lora_dir):
    cmd = [config.sd_python_path, config.batch_txt2image_path, "--lora_dir", lora_dir,  "--batch",
         json.dumps(txt2image_tasks)]

    print(cmd)
    result = subprocess.run(cmd, encoding='utf-8')
    # print(result)
    if result is not None and result.returncode == 0:
        print("finish shell excuted")
    else:
        print("failed excuted shell")
def on_train_batch_finish(path_list, lora_dir):
    config_data = config.config_data
    txt2image_tasks = []
    for path in path_list:
        lora_name = os.path.splitext(os.path.basename(path))[0]
        for section in config_data:
            if section != "path":
                add_section_task(txt2image_tasks, lora_name, section, config_data[section])

    run_task(txt2image_tasks, lora_dir)


def add_section_task(txt2image_tasks, lora_name,  section, config_section):
    weights = str.split(config_section['lora_weights'], ",")
    sampler_name = config_section['sampler_name']
    steps = config_section['steps']
    width = config_section['width']
    height = config_section['height']
    seed = config_section['seed']
    model = config_section['model']
    prompts = []
    for weight in weights:
        p = '"' + config_section['prompt'] + ' <lora:' + lora_name + ':' + weight + '>' + '"'
        prompts.append(p)
    png_path = os.path.join(config.config_data['path']['png_save_path'], lora_name + '_' + section + '.png')
    txt2image_tasks.append({
        'png_path': png_path,
        'prompts':prompts,
        'sampler_name':sampler_name,
        'width': width,
        'height': height,
        'steps': steps,
        'seed': seed,
        'model': model,
    })





