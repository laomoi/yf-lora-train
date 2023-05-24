
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
        if section != "path" and section != "setting":
            add_section_task(txt2image_tasks, "", section, config_data[section])

    run_task(txt2image_tasks, lora_dir, "1")


# model_lora_name=
# grid=lora_weights; lora_weights denoising_strength
# lora_weights=0,0.2,0.6,0.8,1.0
# default_lora_weight=0.8
# prompt=1girl
# negative_prompt=nsfw
# steps=20
# sampler_name=Euler a
# seed=4032694023
# width=512
# height=512
# model=anything-v4
# ;img2img
# denoising_strength=0,0.2,0.6,0.8,1.0
# default_denoising_strength=1.0
# img_src=d:\1.png
# ;controlnet
# guidance_start=0.0
# guidance_end=1.0
# pres=512
# canny_weight=1.0
# canny_img_src=d:\1.png
def add_section_task(txt2image_tasks, lora_name,  section, config_section):
    lora_weights = str.split(config_section['lora_weights'], ",")
    sampler_name = config_section['sampler_name']
    steps = config_section['steps']
    width = config_section['width']
    height = config_section['height']
    seed = config_section['seed']
    model = config_section['model']
    prompt = config_section['prompt']

    negative_prompt = config_section['negative_prompt']
    model_lora_name = config_section['model_lora_name']
    img_src = config_section['img_src']
    guidance_start = config_section['guidance_start']
    guidance_end = config_section['guidance_end']
    canny_weight = config_section['canny_weight']
    canny_img_src = config_section['canny_img_src']
    pre_res = config_section['pre_res']
    denoising_strength = str.split(config_section['denoising_strength'], ",")
    default_denoising_strength = config_section['default_denoising_strength']
    grid = config_section['grid']
    default_lora_weight = config_section['default_lora_weight']

    canny_threshold_a = config_section['canny_threshold_a']
    canny_threshold_b = config_section['canny_threshold_b']
    canny_model = config_section['canny_model']

    loop = config_section['loop']


    if lora_name == "" and model_lora_name != "":
        lora_name = model_lora_name


    # prompts = []
    # for weight in weights:
    #     p = '"' + config_section['prompt'] + ' <lora:' + lora_name + ':' + weight + '>' + '"'
    #     prompts.append(p)
    # png_path = os.path.join(config.config_data['path']['png_save_path'], lora_name + '_' + section + '.png')
    txt2image_tasks.append({
        # 'png_path': png_path,
        'section':section,
        'prompt':prompt,
        'lora_weights':lora_weights,
        'sampler_name':sampler_name,
        'width': width,
        'height': height,
        'steps': steps,
        'seed': seed,
        'model': model,
        'lora_name': lora_name,
        # 'model_lora_name': model_lora_name,

        'negative_prompt':negative_prompt,
        'img_src': img_src,
        'guidance_start': guidance_start,
        'guidance_end': guidance_end,
        'canny_weight': canny_weight,
        'canny_img_src': canny_img_src,
        'pre_res': pre_res,
        'denoising_strength': denoising_strength,
        'default_denoising_strength': default_denoising_strength,
        'grid': grid,
        'default_lora_weight': default_lora_weight,
        'canny_threshold_a':canny_threshold_a,
        'canny_threshold_b':canny_threshold_b,
        'canny_model':canny_model,
        'loop':loop,
    })





