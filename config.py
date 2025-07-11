# -*- encoding: utf-8 -*-
# @File        :   config.py
# @Time        :   2025/01/07 16:37:03
# @Author      :   Siyou
# @Description :

import json
import pathlib


base_dir = str(pathlib.Path(__file__).parent)

"""
读取 json 配置文件。
"""
f = open(f"{base_dir}/config/project.json", encoding='utf-8')
config = json.load(f)

config["project_path"] = base_dir
