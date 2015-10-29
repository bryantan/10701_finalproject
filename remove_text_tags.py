import re
import sys

subs = ['<text.*?>', '</text>', '<fontspec.*?/>', '<page.*?>', '</page>', '<image.*?/>', '<\?.*?\?>', '<!DOCTYPE.*?>', '<pdf2xml.*?>', '</pdf2xml>']

text = ""
with open(sys.argv[1], "r") as xmlfile:
    text = xmlfile.read().replace('\n', ' ')

for sub in subs:
    text = re.sub(sub, '', text)

print text
