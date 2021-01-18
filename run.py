import os, sys, json, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #把tensorflow的warning取消掉
import warnings; warnings.filterwarnings("ignore") #ignore depreciation
from gtts import gTTS
from tempfile import TemporaryFile
parser = argparse.ArgumentParser()
parser.add_argument('-e', type=int, default=-1, help='Use extrative model number')
parser.add_argument('-a', type=int, default=-1, help='Use abstrative model number')
args = parser.parse_args()
if args.e == -1 and args.a == -1:
    print("Please select one model by -e or -a")
    sys.exit()

#----------Data----------#
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
#----------Text selection----------#
#topic
key = False
while(not key):
    class_index = float(input(select_topic_question)) - 1

    if class_index < 0 or class_index > 4:
        print("Type error, please type again")
    else:
        key = True
        class_index = int(class_index)
topic = article_classes[class_index]

#title
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

#----------Summarization process----------@
print("---------------Start summarize---------------\n")
print("Loading for model.....")
print()
if args.e == 1:
    from model.extractive import bert_knn
    model = bert_knn('distilbert-base-uncased')
elif args.e == 2:
    from model.extractive import textrank
    model = textrank()
elif args.a == 1:
    from model.opinosis import opinion_summaizer_3
    model = opinion_summaizer_3.opinosis()
    pass
elif args.a == 2:
    from model.fast_abs_rl import demo
    model = demo.fast_abs()
    pass

summary = model.summarize(text)
print()
Y_or_N = input("Do you want to listen the summary? (y/n)\n")
if(Y_or_N == "y" or Y_or_N == "Y" or Y_or_N == "yes" or Y_or_N == "Yes"):
    tts = gTTS(text=summary, lang='en')
    tts.save("synthesized.mp3")
    from pygame import mixer
    mixer.init()
    mixer.music.load("synthesized.mp3")
    mixer.music.play()
else:
    pass
