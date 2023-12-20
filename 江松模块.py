import json
import random
import jsonlines
import copy

def 读江松行(name):
    datam=[]
    with jsonlines.open(name) as reader:
        for obj in reader:
            datam.append(obj)
    return datam
def 随机打乱(data):
    return random.shuffle(data)
def 键名(data):
    return data.keys()
def 键值(data):
    return data.values()
def 更新字典(dict1,dict2):
    dict1.update(dict2)
def 更新键值(dict,jianming,jianzhi):
    dict[jianming]=jianzhi
def 存江松行(res,name):
    with jsonlines.open(name, mode='w') as writer:
        for item in res:
            writer.write(item)
def 读江松(name):
    with open(name) as reader:
        tmpdata=json.load(reader)
    return tmpdata
函子={"读江松":读江松,"读江松行":读江松行,"随机打乱":随机打乱,"键值":键值,"键名":键名}#接受一个变量的函数字典
函丑={"存江松行":存江松行,"更新字典":更新字典}#接受两个变量的函数字典，依此类推
函寅={"更新键值":更新键值}
函卯={}
函辰={}
函巳={}
函午={}
函未={}
函申={}
函酉={}
函戌={}
函亥={}
函括={}#接受任意个变量的函数字典，引用函数时需要用括号把函数和变量括起来。
