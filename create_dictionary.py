import os
import re
import json

# from remove_text_tags import get_word

def get_word(word):
    word = re.sub('<.*?>', '', re.sub('</.*?>', '', word))
    punc_to_remove = ['.', ',', '!', '?', '(', ')', ';']
    for punc in punc_to_remove:
        word = word.replace(punc, '')
    return word.lower()

textbook_dict = []

# go through all words of each sentence in all samples
for sample_name in os.listdir(os.getcwd() + '/samples'):
    sample_filename = os.getcwd() + '/samples/' + sample_name
    print sample_filename
    with open(sample_filename, "r") as sample_file:
        samples_raw = sample_file.read()
        samples = json.loads(samples_raw)
    for sample in samples:
        for word in sample['main'].split():
            if not get_word(word) in textbook_dict:
                textbook_dict.append(get_word(word))

# load dictionary, make all lowercase
dictionary = open('/usr/share/dict/words', "r").readlines()
dictionary = map(lambda x:x.lower().strip(), dictionary)

new_dict = []
for word in textbook_dict:
    print word
    if word in dictionary:
        new_dict.append(word)

textbook_dict = new_dict

with open('textbook_dict', 'w') as outfile:
    json.dump(textbook_dict, outfile, indent=0)