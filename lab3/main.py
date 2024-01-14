# 首先定义一个函数，用于提取文档并处理噪音

# 先将中括号从原txt提取出来，然后只记录最后的

import re
def getText():
    txt = open("1998.txt", 'r').read()
    txt = txt.lower()
#    for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
#        txt = txt.replace(ch, ' ')  # 将文本中的特殊字符替换为空格
    return txt


def getText2():
    txt = open("199.txt", 'r').read()
    txt = txt.lower()
#    for ch in '!"#$%&()*+, -./:;<=>?@[\\]^_‘{|}~':
#        txt = txt.replace(ch, ' ')  # 将文本中的特殊字符替换为空格
    return txt

#实现总的实体类型计数
needTxt = getText()
#print(needTxt)
words = re.split('/| ',needTxt)
counts = {}  # 创建空字典，存放词频统计信息
for wordd in words:
    if wordd.encode('utf-8').isalpha():
        counts[wordd] = counts.get(wordd, 0) + 1    # 若字典中无当前词语则创建一个键值对，若有则将原有值加1

#计数特殊实体类型，并减去
secondTxt = getText2()
wordss = secondTxt.split('[')
for word in wordss:
    for i in range(len(word)):
        if word[i] == '/' :
            empty_string = ""
            i += 1
            while word[i].encode('utf-8').isalpha():
                empty_string += word[i]
                i += 1
        #    counts[empty_string] = counts.get(empty_string, 0) - 1
        elif word[i] == ']':
            empty_string = ""
            i += 1
            empty_string += word[i]
            i += 1
            empty_string += word[i]
            counts[empty_string] = counts.get(empty_string, 0) + 1

#实现实体计数
needTxt = getText()
#print(needTxt)
words = re.split('/| ',needTxt)
names = {}  # 创建空字典，存放词频统计信息
bu = 0
passs = 0
for i in range(len(words)):
    if words[i] == '':
        continue
    if words[i] == '\n':
        continue
    if words[i].find('[') != -1:
        passs = 1
        empty_string = ""
        for j in range(len(words[i])):
            if words[i][j].encode('utf-8').isalpha() == 0:
                if words[i][j] == '[' or words[i][j] == ']' or words[i][j] == ' ':
                    continue
                empty_string += words[i][j]
        continue
    if words[i].find(']') != -1:
        passs = 0
        for j in range(len(words[i])):
            if words[i][j].encode('utf-8').isalpha() == 0:
                if words[i][j] == '[' or words[i][j] == ']' or words[i][j] == ' ' or words[i][j] == '、':
                    continue
                empty_string += words[i][j]
        names[empty_string] = names.get(empty_string, 0) + 1
        continue
    if passs == 1:
        continue

    if words[i].encode('utf-8').isalpha():
        if words[i] == 'nr':
            if bu == 1:
                bu = 0
                continue
            empty_string = ""
            empty_string += words[i-1]
            empty_string += words[i+2]
            i += 3
            bu = 1
            names[empty_string] = names.get(empty_string, 0) + 1
        elif words[i] == 'ns' or words[i] == 'nt' or words[i] == 'nx' or words[i] == 'nz':
            empty_string = ""
            empty_string += words[i - 1]
            empty_string = empty_string.strip('[')
            names[empty_string] = names.get(empty_string, 0) + 1
            #print("6666666666")
#        names[wor] = names.get(wor, 0) + 1    # 若字典中无当前词语则创建一个键值对，若有则将原有值加1

counts['nr']/=2
items = list(counts.items())  # 将无序的字典类型转换为有序的列表类型
items.sort(key=lambda x: x[1], reverse=True)  # 按统计值从高到低排序（以第二列排序）
for i in range(44):
    word, count = items[i]
    print(i+1,  "{0:<10}{1:>5}".format(word, count))   # 格式化输出词频统计结果


items = list(names.items())  # 将无序的字典类型转换为有序的列表类型
items.sort(key=lambda x: x[1], reverse=True)  # 按统计值从高到低排序（以第二列排序）
for i in range(10):
    word, count = items[i]
    print(i+1,  "{0:<10}{1:>5}".format(word, count))   # 格式化输出词频统计结果
