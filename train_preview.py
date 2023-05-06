
import config
import os
import subprocess

from PIL import Image



def on_train_finish(path, epoch_no, force_sync_upload):
    print("on train finished", path)
    config_data = config.config_data
    lora_dir = os.path.abspath(os.path.dirname(path))
    lora_name = os.path.splitext( os.path.basename(path))[0]

    for section in config_data:
        if section != "path":
            generate_png_for_section(lora_name, lora_dir, section, config_data[section])


def generate_png_for_section(lora_name, lora_dir, section, config_section):
    weights = str.split(config_section['lora_weights'], ",")
    sampler_name = config_section['sampler_name']
    steps = config_section['steps']
    width = config_section['width']
    height = config_section['height']
    seed = config_section['seed']
    model = config_section['model']
    prompts = []
    outputs = []
    for weight in weights:
        p = '"' + config_section['prompt'] + ' <lora:' + lora_name + ':' + weight + '>' + '"'
        png_path = os.path.join(config.temp_dir, lora_name + '_' + section + "_" + weight + '.png')
        prompts.append(p)
        outputs.append(png_path)
    cmd = [config.sd_python_path, config.txt2image_path, "--lora_dir", lora_dir, "--output", *outputs, "--prompt",
           *prompts, "--sampler_name", sampler_name,
           "--width", width, "--height", height, "--steps", steps, "--seed", seed, "--model", model]

    print(cmd)
    result = subprocess.run(cmd, encoding='utf-8')
    # print(result)
    if result is not None and result.returncode == 0:
        images = []
        # 遍历outputs列表，读取图片并转换成Image对象
        for path in outputs:
            image = Image.open(path)
            images.append(image)

        png_path = os.path.join(config.config_data['path']['png_save_path'], lora_name + '_' + section + '.png')
        merged_image = merge_images(images)
        merged_image.save(png_path, format="PNG")
        print("saved png ", png_path)

        for temp in outputs:
            os.remove(temp)
        return True

    print("generate png failed")
    return False

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



