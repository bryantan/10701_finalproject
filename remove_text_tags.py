import os
import re
import sys
import json
from subprocess import call

def get_word(word):
    word = re.sub('<.*?>', '', re.sub('</.*?>', '', word))
    return word.lower()

# things to remove from XML text
subs = ['<text.*?>', '</text>', '<fontspec.*?/>', '<page.*?>', '</page>', '<image.*?/>', '<\?.*?\?>', '<!DOCTYPE.*?>', '<pdf2xml.*?>', '</pdf2xml>']

# load dictionary, make all lowercase
dictionary = open('/usr/share/dict/words', "r").readlines()
dictionary = map(lambda x:x.lower().strip(), dictionary)

call(["pdftohtml", "-c", "-i", "-xml", sys.argv[1], sys.argv[1] + "_out.xml"])

# process XML text
text = ""
with open(sys.argv[1] + "_out.xml", "r") as xmlfile:
    text = xmlfile.read().replace('\n', ' ')

# remove specified items
for sub in subs:
    text = re.sub(sub, '', text)

# remove extra spaces
' '.join(text.split())

# combine halves of a word that were separated by new line
words = text.split()
new_words = []
i = 1
while i < len(words):
    if (get_word(words[i-1]) + get_word(words[i])) in dictionary:
        new_words.append(words[i-1] + words[i])
        i = i + 2
        if i == len(words):
            new_words.append(words[i-1])
    else:
        new_words.append(words[i-1])
        if i == len(words) - 1:
            new_words.append(words[i])
        i = i + 1
new_text = " ".join(new_words)

# add new lines for sentences
new_text = re.sub('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', '\n', new_text)
new_text = re.sub('Fig.\n', 'Fig. ', new_text)

with open(sys.argv[1] + "_sentences", 'w') as outfile:
    outfile.write(new_text)

# create list of sentences and then add sample objects to list
sentences = new_text.split('\n')
samples = []
for i in xrange(0, len(sentences)):
    if i == 0:
        samples.append({'pre': '', 'main': sentences[0], 'post': sentences[1], 'type': 'none'})
    elif i == len(sentences) - 1:
        samples.append({'pre': sentences[i-1], 'main': sentences[i], 'post': '', 'type': 'none'})
    else:
        samples.append({'pre': sentences[i-1], 'main': sentences[i], 'post': sentences[i+1], 'type': 'none'})

# manually select what type of sample each sentence is
# for sample in samples:
#     print "Current sentence: " + sample['main']
#     sample_type = input("What is the type of this sentence? 1. None, 2. Theorem, 3. Definition\n")
#     if sample_type == 1:
#         sample['type'] = 'none'
#     elif sample_type == 2:
#         sample['type'] = 'theorem'
#     elif sample_type == 3:
#         sample['type'] = 'definition'
#     else:
#         sample['type'] = 'none'

with open(sys.argv[1] + '_samples', 'w') as outfile:
    json.dump(samples, outfile)

os.remove(sys.argv[1] + "_out.xml")