import sys
import os
webui_lib_root_path =  os.path.join( os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "webui_lib" )
sys.path.append(webui_lib_root_path)
import argparse
parser = argparse.ArgumentParser(description='txt2_image desc')
parser.add_argument('--prompt',  nargs='+', default=["1girl"])
parser.add_argument('--model', default="")
parser.add_argument('--output', nargs='+',  default=["./output.png"])
parser.add_argument('--lora_dir',  default="")
parser.add_argument('--sampler_name',  default="Euler a")
parser.add_argument('--steps',  default="20")
parser.add_argument('--width',  default="512")
parser.add_argument('--height',  default="512")
parser.add_argument('--seed',  default="-1")


args,remaining = parser.parse_known_args()
sys.argv[1:] = remaining


from yfcore import webui_lib



def start():
    if args.model != "" :
        webui_lib.set_default_model(args.model)
    if args.lora_dir != "":
        webui_lib.set_lora_dir(args.lora_dir)

    webui_lib.initialize()

    model_list = webui_lib.get_checkpoint_list()
    print("available models:")
    for m in model_list:
        print(m)
    print("prompts", args.prompt)
    steps = int(args.steps)
    sampler_name = args.sampler_name
    seed = int(args.seed)
    width = int(args.width)
    height = int(args.height)

    i = 0
    for p in args.prompt:
        output = args.output[i]
        i = i + 1
        images = webui_lib.txt2img({'prompt': p, 'sampler_name':sampler_name, 'seed':seed, 'width':width, 'height':height, 'steps':steps})
        if images and len(images) > 0:
            webui_lib.save_image(images[0], output)
            print("saved to " + output)


    # webui_lib.reload_model("3Guofeng3_v33.safetensors [4078eb4174]")


start()
