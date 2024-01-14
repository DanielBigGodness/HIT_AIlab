# 首先定义一个函数，用于提取文档并处理噪音

# 先将中括号从原txt提取出来，然后只记录最后的

import re


with open('test.txt', 'r', encoding='utf-8') as p:  # 以只读的方式打开不会改变原文件内容

    lines = []
    for i in p:
        lines.append(i)  # 逐行将文本存入列表lines中
    p.close()
    # print(lines)
    new = []

    for line in lines:  # 逐行遍历
        p = 0  # 定义计数指针
        for bit in line:
            if bit != 'm':
                p = p + 1
            else:
                new.append(line[p + 1:])  # 将斜杠后面的内容加到新的list中
                break

with open('tt.txt', 'w') as file_write:
    for var in new:
        file_write.writelines(var)  # 写入

def getText():
    txt = open("tt.txt", 'r').read()
    txt = txt.lower()
#    for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
#        txt = txt.replace(ch, ' ')  # 将文本中的特殊字符替换为空格
    return txt

new = []
#35060
#实现总的实体类型计数
needTxt = getText()
#print(needTxt)
words = re.split('/| ',needTxt)
check = 0
flag = 0
first = 1
empty_string = ""
for i in range(len(words)):
    if words[i] == " ":
        continue
    if words[i] == '\n':
        print("")
        #new.append("")
        continue

    if words[i].find('[') != -1:
        flag = 1
        first = 0
        for k in range(i,999999):
            if words[k].find(']') != -1:
                for q in range(words[k].find(']')+1, len(words[k])):
                    empty_string += words[k][q]
                break
        continue

    if flag == 1:
        if words[i-1].encode('utf-8').isalpha() == 0:
            for j in range(len(words[i-1])):
                if words[i-1][j] == '[':
                    print('[', 'O')
                    #new.append('[', '0')
                    continue
                if words[i-1].find(']') != -1:
                    flag = 0
                    break
                if first == 0:
                    if empty_string == 'ns':
                        print(words[i-1][j], "B_NS")
                    else:
                        print(words[i-1][j], "B_NT")
                    first = 1
                else:
                    print(words[i-1][j], "I")
        continue

    if words[i].encode('utf-8').isalpha():
        if words[i] == 'ns':
            for j in range(len(words[i-1])):
                if j == 0:
                    print(words[i-1][0], "B_NS")
                else:
                    print(words[i-1][j], "I")
        elif words[i] == 'nt':
                for j in range(0, len(words[i - 1])):
                    if j == 0:
                        print(words[i-1][0], "B_NT")
                    else:
                        print(words[i-1][j], "I")
        elif words[i] == 'nr':
            if check == 1:
                check = 0
                continue
            check = 1
            for j in range(len(words[i - 1])):
                if j == 0:
                    print(words[i - 1][j], "B_NR")
                else:
                    print(words[i - 1][j], "I")
            for j in range(len(words[i + 2])):
                if words[i+2][j] == '' or words[i+2][j] == ' ':
                    break
                print(words[i+2][j], "I")

        else:
            for j in range(0, len(words[i-1])):
                    print(words[i-1][j], "O")



