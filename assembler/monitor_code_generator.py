#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


idx_max = 3
pos1_max = 10
pos2_max = 30

print('#include "deps/attr/Attr_API.h"')
print('#include <unordered_map>')
print('typedef void (*monitor_func)();')

for idx in range(idx_max):
    for pos1 in range(pos1_max):
        for pos2 in range(pos2_max):
            print("static void training_monitor_func_{}_{}_{}()".format(idx, pos1, pos2))
            print("{")
            print("    Attr_API(0, 1);  // {}_{}_{}-training特征使用默认值".format(idx, pos1, pos2))
            print("}")

print("static std::unordered_map<std::string, monitor_func> training_monitor_func_map = {")
for idx in range(idx_max):
    for pos1 in range(pos1_max):
        for pos2 in range(pos2_max):
            name = "training_monitor_func_{}_{}_{}".format(idx, pos1, pos2)
            print("{" + '"' + name + '", ' + name + "},")
print("};")


for idx in range(idx_max):
    for pos1 in range(pos1_max):
        for pos2 in range(pos2_max):
            print("static void serving_monitor_func_{}_{}_{}()".format(idx, pos1, pos2))
            print("{")
            print("    Attr_API(0, 1);  // {}_{}_{}-serving特征使用默认值".format(idx, pos1, pos2))
            print("}")

print("static std::unordered_map<std::string, monitor_func> serving_monitor_func_map = {")
for idx in range(idx_max):
    for pos1 in range(pos1_max):
        for pos2 in range(pos2_max):
            name = "serving_monitor_func_{}_{}_{}".format(idx, pos1, pos2)
            print("{" + '"' + name + '", ' + name + "},")
print("};")
