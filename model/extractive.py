import warnings
warnings.filterwarnings("ignore") #ignore depreciation
from summarizer import Summarizer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize  #需要nltk.download('punkt')
import tensorflow_hub as hub

PRETRAINED_ROOT = "./pretrained_model"

class bert_knn():
    def __init__(self, model_name):
        self.model = Summarizer(model_name) #location at /.cache/torch
    def summarize(self, text):
        num_sentences = int(input("How many sentences you want(1~30)\n"))

        summary = self.model(text, num_sentences=num_sentences)
        print(summary)


#Use universal sentence encoder
class textrank():
    def __init__(self):
        self.embed = hub.load(PRETRAINED_ROOT + "/universal-sentence-encoder_4/")
    def summarize(self, text):
        sentences = sent_tokenize(text)
        word_embedding = np.array(self.embed(sentences))

        #generate cosine similarity matrix
        sim_matrix = cosine_similarity(word_embedding)

        #create graph and generate scores from pagerank algorithms
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

        num_of_sentences = int(input("How many sentences you want(1~30)\n"))   
        summary = " ".join([i[1] for i in ranked_sentences[:num_of_sentences]])
        
        print(summary)





        
