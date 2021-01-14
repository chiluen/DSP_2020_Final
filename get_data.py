import requests as rq
from bs4 import BeautifulSoup
import io
import time
import json
import argparse

def crawl_news(link):
    response = rq.get(link)
    soup = BeautifulSoup(response.text, "html.parser")
    t = soup.find_all("div", class_="zn-body__paragraph")
    text_total = []
    for i in t:
        text_total.append(i.getText())
    time.sleep(SLEEP_TIME)
    return " ".join(text_total)



parser = argparse.ArgumentParser()
parser.add_argument('-num', type=int, default=3, help='Number of articles each topic')
parser.add_argument('-speed', action='store_true', help='Speed up the crawl')
args = parser.parse_args()


NUMBER_ARTICLES = args.num
if args.speed:
    SLEEP_TIME = 0.5
else:
    SLEEP_TIME = 1

start_time = time.time()

url_front = "https://edition.cnn.com"
article_classes = ['business', 'health', 'sport', 'world', 'entertainment'] #tech不知為何抓不到
title_dic = {}
text_dic = {}
for article_class in article_classes:
    
    response = rq.get(url_front + "/" +article_class)
    if response.ok:
        print("Start for " + article_class)

    """
    還要加上確認是否link中有video
    """
    soup = BeautifulSoup(response.text, "html.parser")
    titles = soup.find_all("h3", class_ = "cd__headline", limit=10) #目前是選擇10個title
    title_dic[article_class] = []
    text_dic[article_class] = []
    
    maximum_num = 0
    for title in titles:
        if maximum_num == NUMBER_ARTICLES:
            break
        href = title.a.get("href")
        if 'video' not in href: #不選取video類型的
            title_dic[article_class].append(title.getText())
            text_dic[article_class].append(crawl_news(url_front + href))
            maximum_num += 1
    time.sleep(SLEEP_TIME)
end_time = time.time()
print("Total time for getting {} CNN articles: {}".format(5*NUMBER_ARTICLES, end_time - start_time))

#把title與document寫入
jsObj = json.dumps(title_dic)
with open('./crawl_data/title.json', 'w') as f:
    f.write(jsObj)

jsObj = json.dumps(text_dic)
with open('./crawl_data/text.json', 'w') as f:
    f.write(jsObj)