import sys
import os
import json
from PIL import Image
webui_lib_root_path =  os.path.join( os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "webui_lib" )
sys.path.append(webui_lib_root_path)
import argparse
parser = argparse.ArgumentParser(description='txt2_image desc')
parser.add_argument('--batch',   default="")
parser.add_argument('--lora_dir',  default="")




args,remaining = parser.parse_known_args()
sys.argv[1:] = remaining


import webui_lib


last_model = ""

def start():
    if args.lora_dir != "":
        webui_lib.set_lora_dir(args.lora_dir)

    webui_lib.initialize()


    batch = json.loads(args.batch)

    for task in batch:
        do_task(task)



    # webui_lib.reload_model("3Guofeng3_v33.safetensors [4078eb4174]")
def do_task(task):
    global  last_model
    model_list = webui_lib.get_checkpoint_list()
    model = task['model']
    png_path = task['png_path']
    steps = int(task['steps'])
    sampler_name = task['sampler_name']
    seed = int(task['seed'])
    width = int(task['width'])
    height = int(task['height'])
    prompts = task['prompts']
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
    for p in prompts:
        images = webui_lib.txt2img(
            {'prompt': p, 'sampler_name': sampler_name, 'seed': seed, 'width': width, 'height': height,
             'steps': steps})
        if images and len(images) > 0:
            to_merge_imgs.append((images[0]))
    print("merging...", png_path)
    merged_image = merge_images(to_merge_imgs)
    merged_image.save(png_path, format="PNG")
    print("saved png ", png_path)
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


start()
