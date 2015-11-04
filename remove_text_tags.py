import os
import re
import sys
import json
from subprocess import call

def get_word(word):
    word = re.sub('<.*?>', '', re.sub('</.*?>', '', word))
    # punc_to_remove = ['.', ',', '!', '?']
    # for punc in punc_to_remove:
    #     word = word.replace(punc, '')
    return word.lower()

def add_bold_tags(word, in_bold):
    if not in_bold:
        if word == '<b>':
            return ('', True)
        elif '<b>' in word and '</b>' in word:
            return (word, False)
        elif '<b>' in word:
            return (word + '</b>', True)
        else:
            return (word, False)
    else:
        if word == '</b>':
            return ('', False)
        elif '</b>' in word:
            return ('<b>' + word, False)
        else:
            return('<b>' + word + '</b>', True)

def combine_words(word1, word2, in_bold):
    combined_word = word1 + word2
    if in_bold or '<b>' in word1:
        combined_word = combined_word.replace('<b>', '')
        combined_word = combined_word.replace('</b>', '')
        combined_word = '<b>' + combined_word + '</b>'

    if in_bold and not '</b>' in word2:
        return (combined_word, True)
    elif not in_bold and '<b>' in word1:
        return (combined_word, True)
    else:
        return (combined_word, False)


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
in_bold = False
while i < len(words):
    if (len(get_word(words[i-1])) == 1 and not get_word(words[i]) in dictionary) and \
       (get_word(words[i-1]) + get_word(words[i])) in dictionary:
        combined_word, in_bold = combine_words(words[i-1], words[i], in_bold)
        new_words.append(combined_word)
        i = i + 2
        if i == len(words):
            words[i-1], in_bold = add_bold_tags(words[i-1], in_bold)
            new_words.append(words[i-1])
    else:
        words[i-1], in_bold = add_bold_tags(words[i-1], in_bold)
        new_words.append(words[i-1])
        if i == len(words) - 1:
            words[i], in_bold = add_bold_tags(words[i], in_bold)
            new_words.append(words[i])
        i = i + 1
new_text = " ".join(new_words)

# add new lines for sentences
# https://regex101.com/r/nG1gU7/27#python
new_text = re.sub('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', '\n', new_text)
new_text = re.sub('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)</b>\s', '</b>\n', new_text)
new_text = re.sub('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)</i>\s', '</i>\n', new_text)
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
    json.dump(samples, outfile, indent=0)

os.remove(sys.argv[1] + "_out.xml")
