import warnings
warnings.filterwarnings("ignore") #ignore depreciation
import torch
from summarizer import Summarizer
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', type=int, default=1, help='Use extrative model number')
parser.add_argument('-a', type=int, default=1, help='Use abstrative model number')
args = parser.parse_args()

with open('./crawl_data/title.json', 'r') as f:
    title_data = json.load(f)
with open('./crawl_data/text.json', 'r') as f:
    text_data = json.load(f)

article_classes = ['business', 'health', 'sport', 'world', 'entertainment']

select_topic_question = """
Which topic you want to read:
1: business
2: health
3: sport
4: world
5: entertainment
"""

# 選topic
key = False
while(not key):
    class_index = float(input(select_topic_question)) - 1

    if class_index < 0 or class_index > 4:
        print("Type error, please type again")
    else:
        key = True
        class_index = int(class_index)
topic = article_classes[class_index]

# 選title
select_title_question = "\nWhich title you want to read:\n"
for i in range(len(title_data[topic])):
    select_title_question += "{}:   ".format(i+1) + title_data[topic][i] + "\n"

key = False
while(not key):
    title_index = float(input(select_title_question)) - 1

    if title_index < 0 or title_index > len(title_data[topic])-1:
        print("Type error, please type again")
    else:
        key = True
        title_index = int(title_index)

text = text_data[topic][title_index]

print("---------------Start summarize---------------")        
from model.extractive import bert_knn
model = bert_knn('distilbert-base-uncased')
model.summarize(text)


"""
#Summarize
model = Summarizer('distilbert-base-uncased') #location at /.cache/torch
text = text_data[topic][title_index]
num_sentences = int(input("How many sentences you want(1~30)\n"))

start_time = time.time()
resp = model(text, num_sentences=num_sentences)
end_time = time.time()

print(resp)
print("\nTotal time used :{}".format(end_time - start_time))
"""