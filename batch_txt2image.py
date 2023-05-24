import random
import sys
import os
import json
from PIL import Image
import numpy as np
webui_lib_root_path =  os.path.join( os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "webui_lib" )
sys.path.append(webui_lib_root_path)
import argparse
parser = argparse.ArgumentParser(description='txt2_image desc')
parser.add_argument('--batch',   default="")
parser.add_argument('--lora_dir',  default="")
parser.add_argument('--list_lora_dir',  default="1")
parser.add_argument('--png_save_path',  default="./output")




args,remaining = parser.parse_known_args()
sys.argv[1:] = remaining


import webui_lib
import gc
import os
import fnmatch

last_model = ""

def start():
    if args.lora_dir != "":
        webui_lib.set_lora_dir(args.lora_dir)

    webui_lib.initialize()
    gc.collect()

    batch = json.loads(args.batch)

    if args.list_lora_dir == "0":
        #task.lora_name is confirmed, run specify lora
        for task in batch:
            do_task(task)
    else:
        # task.lora_name need to be listed
        ext = "safetensors"  # 扩展名
        path_list = []
        # 枚举目录下的所有文件和子目录
        for root, dirs, files in os.walk(args.lora_dir):
            # 遍历文件
            for filename in fnmatch.filter(files, f"*.{ext}"):
                file_path = os.path.join(root, filename)
                # 将符合条件的文件路径添加到列表中
                path_list.append(file_path)

        for task in batch:
            if task['lora_name'] == "":
                #need all loras
                for path in path_list:
                    lora_name = os.path.splitext(os.path.basename(path))[0]
                    task['lora_name'] = lora_name
                    do_task(task)
            else:
                #run specify lora
                do_task(task)



    # webui_lib.reload_model("3Guofeng3_v33.safetensors [4078eb4174]")
def do_task(task):
    loop = int(task.get('loop', 1))
    for i in range(loop):
        do_inner_task(task, i+1)


def do_inner_task(task, loop_index):
    global  last_model
    model_list = webui_lib.get_checkpoint_list()
    model = task['model']
    # png_path = task['png_path']
    steps = int(task['steps'])
    sampler_name = task['sampler_name']
    seed = int(task['seed'])
    width = int(task['width'])
    height = int(task['height'])
    prompt = task['prompt']
    lora_weights = task['lora_weights']
    lora_name = task['lora_name']
    section = task['section']

    negative_prompt = task['negative_prompt']
    img_src = task['img_src']
    guidance_start = float(task['guidance_start'])
    guidance_end = float(task['guidance_end'])
    canny_weight = float(task['canny_weight'])
    canny_img_src = task['canny_img_src']
    pre_res = int(task['pre_res'])
    denoising_strength = task['denoising_strength']
    default_denoising_strength = float(task['default_denoising_strength'])
    grid = task['grid']
    default_lora_weight = float(task['default_lora_weight'])
    canny_threshold_a = float(task['canny_threshold_a'])
    canny_threshold_b = float(task['canny_threshold_b'])
    canny_model = task['canny_model']


    if seed == -1:
        seed = random.randint(1, sys.maxsize)
    print("use seed=", str(seed))


    # prompts = task['prompts']
    # reload model
    if last_model != model:
        found_model = False
        for m in model_list:
            if m == model:
                found_model = True
                break
        if not found_model:
            for m in model_list:
                if m.startswith(model):
                    print("find matched model", m)
                    webui_lib.reload_model(m)
                    last_model = m
                    break
    to_merge_imgs = []


    params = {
         'prompt': prompt,
         'negative_prompt':negative_prompt,
         'sampler_name': sampler_name,
         'seed': seed, 'width': width, 'height': height,
         'steps': steps,
         'denoising_strength':default_denoising_strength,
    }

    if canny_img_src != "":
        controlnet_params = make_controlnet_params(canny_model, canny_weight, guidance_start, guidance_end, canny_img_src, pre_res,canny_threshold_a,canny_threshold_b)
    else:
        controlnet_params = []

    if img_src is not None and img_src != "":
        is_img2img = True
        init_images = [Image.open(img_src)]
        params['init_images'] = init_images
    else:
        is_img2img = False

    #if is_img2img == false, force grid = lora_weights
    if is_img2img == False:
        grid = 'lora_weights'

    if grid == "lora_weights":
        for weight in lora_weights:
            params['prompt'] = '"' + prompt + ' <lora:' + lora_name + ':' + weight + '>' + '"'
            params['denoising_strength'] = default_denoising_strength
            if img_src is not None and img_src != "":
                images = webui_lib.img2img(params, None, None, controlnet_params)
            else:
                images = webui_lib.txt2img(params, None, None, controlnet_params)
            if images and len(images) > 0:
                to_merge_imgs.append((images[0]))
    else:
        weight = default_lora_weight
        for strength in denoising_strength:
            params['prompt'] = '"' + prompt + ' <lora:' + lora_name + ':' + str(weight) + '>' + '"'
            params['denoising_strength'] = float(strength)

            if img_src is not None and img_src != "":
                images = webui_lib.img2img(params, None, None, controlnet_params)
            else:
                images = webui_lib.txt2img(params, None, None, controlnet_params)
            if images and len(images) > 0:
                to_merge_imgs.append((images[0]))


    png_path = os.path.join(args.png_save_path, lora_name + '_' + section + '_' + str(loop_index) + '.png')

    print("merging...", png_path)
    merged_image = merge_images(to_merge_imgs)
    merged_image.save(png_path, format="PNG")
    print("saved png ", png_path)

    gc.collect()
def merge_images( img_array, direction="horizontal", gap=0):
    img_array = [img for img in img_array]
    w, h = img_array[0].size
    if direction == "horizontal":
        result = Image.new(img_array[0].mode, ((w + gap) * len(img_array) - gap, h))
        for i, img in enumerate(img_array):
            result.paste(img, box=((w + gap) * i, 0))
    elif direction == "vertical":
        result = Image.new(img_array[0].mode, (w, (h + gap) * len(img_array) - gap))
        for i, img in enumerate(img_array):
            result.paste(img, box=(0, (h + gap) * i))
    else:
        raise ValueError("The direction parameter has only two options: horizontal and vertical")
    return result

def make_controlnet_params(canny_model, canny_weight, guidance_start, guidance_end, canny_img_src, pre_res,canny_threshold_a,canny_threshold_b):
    png = Image.open(canny_img_src)
    mask = Image.new("RGB", (png.width, png.height), (0, 0, 0, 255))

    search_canny_model_name_list = [canny_model, 'control_v11p_sd15_canny', 'control_sd15_canny', 'control_canny']
    for try_name in search_canny_model_name_list:
        if try_name != "":
            canny_model_name = webui_lib.get_cn_model_name(try_name)
            if canny_model_name is not None:
                break

    if canny_model_name is None:
        print("canny_model_name is None!! " )

    controlnet_params = [
        {
                 'enabled': True,
                'image': {'image': np.array(png), 'mask':np.array(mask)},
                'lowvram': False,
                'model':  canny_model_name,
                'module': 'canny',
                'processor_res': pre_res,
                'resize_mode': 'Crop and Resize',
                'threshold_a': canny_threshold_a,
                'threshold_b': canny_threshold_b,
                'weight': canny_weight,
                'control_mode': 'Balanced',
                'pixel_perfect': False,
                'loopback': False,
                'batch_image':"",
                'guidance_start': guidance_start,
                'guidance_end': guidance_end,
        }
    ]
    
    
   
    return controlnet_params

start()
