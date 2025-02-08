# -*- encoding: utf-8 -*-
# @File        :   array_split.py
# @Time        :   2025/01/14 11:10:54
# @Author      :   Siyou
# @Description :

import numpy as np

def array_split(raw_list:list, split_num:int)->list:
    print("split length:",len(raw_list))
    avg = len(raw_list) / float(split_num)
    split_list = []
    last = 0.0
    while last < len(raw_list):
        split_list.append(raw_list[int(last):int(last + avg)])
        last += avg
    return split_list