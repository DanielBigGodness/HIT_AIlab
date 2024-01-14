import requests
import bs4
from bs4 import BeautifulSoup
import json
import re

response = requests.get('http://www.hit.edu.cn')
# 根据html网页字符串创建BeautifulSoup对象
html_doc = response.content
soup = BeautifulSoup(html_doc, 'html.parser')
dict={}
#从文档中找到所有标签的链接
for link in soup.find_all('a'):
    dict[link.get('title')]=link.get('href')
#正则匹配
link_node = soup.find('a',href=re.compile(r"til"))
#print(link_node)

json_str = json.dumps(dict)
print(json_str)

txt_str = json.loads(json_str)
print(txt_str)
