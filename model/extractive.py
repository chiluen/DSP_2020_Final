import warnings
warnings.filterwarnings("ignore") #ignore depreciation
from summarizer import Summarizer
import time


class bert_knn():
    def __init__(self, model_name):
        self.model = Summarizer(model_name) #location at /.cache/torch
    def summarize(self, text):
        num_sentences = int(input("How many sentences you want(1~30)\n"))
        start_time = time.time()
        resp = self.model(text, num_sentences=num_sentences)
        end_time = time.time()

        print(resp)
        print("\nTotal time used :{}".format(end_time - start_time))


#class textrank():






        
