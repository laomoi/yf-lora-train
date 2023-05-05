import sys
import os
webui_lib_root_path =  os.path.join( os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "webui_lib" )
sys.path.append(webui_lib_root_path)

import argparse
parser = argparse.ArgumentParser(description='txt2_image desc')
parser.add_argument('--prompt', '-p', default="1girl")
parser.add_argument('--model', '-m', default="")
parser.add_argument('--output', '-o', default="./output.png")
parser.add_argument('--lora_dir', '-ld', default="")

args,remaining = parser.parse_known_args()
sys.argv[1:] = remaining


from yfcore import webui_lib


def start():
    if args.model != "" :
        webui_lib.set_default_model(args.model)
    if args.lora_dir != "":
        webui_lib.set_lora_dir(args.lora_dir)

    webui_lib.initialize()
    #print(webui_lib.get_checkpoint_list())
    print("prompt", args.prompt)
    images = webui_lib.txt2img({'prompt': args.prompt} )
    if images and len(images) > 0:
        webui_lib.save_image(images[0], args.output )
        print("saved to " + args.output )
    # webui_lib.reload_model("3Guofeng3_v33.safetensors [4078eb4174]")


start()
