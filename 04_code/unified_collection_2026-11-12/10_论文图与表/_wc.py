# -*- coding: utf-8 -*-
import re
t = open('main.tex', encoding='utf-8').read()
m = re.search(r'\\begin\{abstract\}(.*?)\\section\{Methods\}', t, re.S)
body = m.group(1)
body = re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', ' ', body, flags=re.S)
body = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', body)
body = re.sub(r'[{}$\\~]', ' ', body)
print('main text words (abstract..discussion):', len(body.split()))
