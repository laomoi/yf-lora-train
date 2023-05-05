

import configparser
import os
import subprocess

from PIL import Image

import tempfile

class MyParser(configparser.ConfigParser):
    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d
def get_config(config_path: str):
    config = MyParser()
    config.read(config_path, encoding='utf-8')
    return config.as_dict()


def on_train_finish(path, epoch_no, force_sync_upload):
    print("on train finished", path)
    lora_dir = os.path.abspath(os.path.dirname(path))
    lora_name = os.path.splitext( os.path.basename(path))[0]
    weights = str.split(config['preview']['lora_weights'], ",")
    sampler_name = config['preview']['sampler_name']
    steps = config['preview']['steps']
    width = config['preview']['width']
    height = config['preview']['height']
    seed = config['preview']['seed']
    model = config['preview']['model']
    prompts = []
    outputs = []
    for weight in weights:
        p = '"' + config['preview']['prompt'] + ' <lora:' + lora_name + ':' + weight + '>' + '"'
        png_path = os.path.join(temp_dir, lora_name + '_' + weight + '.png')
        prompts.append(p)
        outputs.append(png_path)
    cmd = [sd_python_path, txt2image_path,  "--lora_dir", lora_dir, "--output", *outputs, "--prompt", *prompts, "--sampler_name", sampler_name,
           "--width", width, "--height", height,"--steps", steps,"--seed", seed, "--model", model]


    print(cmd)
    result = subprocess.run(cmd, encoding='utf-8')
    # print(result)
    if result is not None and result.returncode == 0:
        images = []
        # 遍历outputs列表，读取图片并转换成Image对象
        for path in outputs:
            image = Image.open(path)
            images.append(image)

        png_path = os.path.join(config['preview']['savefolder'], lora_name + '.png')
        merged_image = merge_images(images)
        merged_image.save(png_path, format="PNG")
        print("saved png ", png_path)
        for temp in outputs:
            os.remove(temp)
        return

    print("generate png failed")


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

config_file =  os.path.join( os.path.abspath(os.path.dirname(__file__)), "config.ini" )
config = get_config(config_file)

parent_dir_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sd_python_path = os.path.join(parent_dir_path, 'stable-diffusion-webui', "venv/Scripts/python.exe")
txt2image_path = os.path.join(os.path.dirname(__file__), 'txt2image.py')
temp_dir = tempfile.gettempdir()

