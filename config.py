import configparser
import os
import sys
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

config_file =  os.path.join( os.path.abspath(os.path.dirname(__file__)), "config.ini" )
config_data = get_config(config_file)

parent_dir_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

sd_scripts_root_path = ""
if config_data['path']['sd_scripts_root_path'] == "":
    sd_scripts_root_path = os.path.join( os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "sd-scripts" )
else:
    sd_scripts_root_path = config_data['path']['sd_scripts_root_path']
sys.path.append(sd_scripts_root_path)


sd_python_path = ""
if config_data['path']['sd_python_path'] == "":
    sd_python_path = os.path.join(parent_dir_path, 'stable-diffusion-webui', "venv/Scripts/python.exe")
else:
    sd_python_path = config_data['path']['sd_python_path']

txt2image_path = os.path.join(os.path.dirname(__file__), 'txt2image.py')
batch_txt2image_path = os.path.join(os.path.dirname(__file__), 'batch_txt2image.py')

temp_dir = tempfile.gettempdir()
