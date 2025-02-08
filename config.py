# -*- encoding: utf-8 -*-
# @File        :   config.py
# @Time        :   2025/01/07 16:37:03
# @Author      :   Siyou
# @Description :

import json
import os

"""
读取 json 配置文件。
"""
f = open("config/project.json", encoding='utf-8')
config = json.load(f)
