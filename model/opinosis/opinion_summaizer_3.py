""" An Python implemenation of Opinosis
Reference:
   Kavita Ganesan, ChengXiang Zhai, Jiawei Han. Opinosis: A Graph Based Approach to Abstractive Summarization of Highly Redundant Opinions. In Proceedings of the 23rd International Conference on Computational Linguistics (COLING 2010). Beijing, China.
   """
# Author: Xinfan Meng mxf3306@gmail.com

import re
import networkx as nx
import numpy as np
from configparser import ConfigParser
from collections import defaultdict, Counter
from operator import itemgetter
from glob import glob
import nltk

class opinosis():
    def __init__(self):
        #print("here!!!")
        pass
    def POSTagger(self, text):
        words=nltk.word_tokenize(text)
        word_pos = nltk.pos_tag(words)
        with open("tmp_reviews", 'w+') as fd:
            for i in range(len(word_pos)):
                
                if (word_pos[i][1] == "."):
                    print(word_pos[i][0] + "/" + word_pos[i][1], file=fd)
                    #print(word_pos[i][0])
                elif (word_pos[i][1] != "."):
                    #print(word_pos[i][0], end=" ")
                    print(word_pos[i][0] + "/" + word_pos[i][1], end=" ", file=fd)
    def summarize(self, text):
        
        #text = "They have traded the 31-year-old to the Brooklyn Nets as part of a three-team deal, the Nets announced on Thursday. In return for Harden, Houston is acquiring Caris LeVert and Rodions Kurucs from the Nets, Dante Exum from the Cleveland Cavaliers, three first-round picks from the Nets, one first-round pick from the Cavaliers via the Milwaukee Bucks, and four first-round pick swaps from the Nets. In a separate deal, Houston is trading LeVert and a second-round pick to the Indiana Pacers for guard and two-time All-Star Victor Oladipo, according to The Athletic's Shams Charania.   Harden, an eight-time All-Star, was acquired by the Rockets from the Oklahoma City Thunder in 2012.  While in Houston, he was voted the league's best player for the 2017-18 season and led the Rockets to the playoffs in all eight years. Visit CNN.com/sport for more news, videos and features \"Adding an All-NBA player such as James to our roster better positions our team to compete against the league's best,\" said Nets General Manager Sean Marks.  \"James is one of the most prolific scorers and playmakers in our game, and we are thrilled to bring his special talents to Brooklyn.\"  The trade comes in the same week that Harden criticized the Rockets.  \"We're just not good enough -- obviously, chemistry, talent-wise, just everything -- and it was clear these last few games,\" Harden said. \"I love this city. I've literally done everything that I can. I mean, this situation, it's crazy. It's something that I don't think can be fixed.\" READ: LeBron James hits no-look three pointer to win mid-game bet with teammate in LA Lakers rout The postgame comments were the last of a string a of negative behavior from the disgruntled star, after arriving late to the team's training camp, and then being sidelined for four days and fined $50,000 by the NBA for violating the league's health and safety protocols days before the start of the season. The former MVP now reunites with former Thunder teammate Kevin Durant and perennial All-Star guard Kyrie Irving in Brooklyn.  Further postponements Meanwhile, the NBA announced on Wednesday that it was postponing its ninth game of the season. Two games scheduled for Friday between the Phoenix Suns and the Golden State Warriros and the Washington Wizards and Detriot Pistons have been postponed as ongoing contact tracing means the Suns and the Wizards do not have the required eight players to contest the games.  Since January 6, 16 NBA players have tested positive for Covid-19.  The league announced on Tuesday that it had adopted stricter health and safety protocols to combat the spread of the virus. "
        self.POSTagger(text)
        review_files = ["tmp_reviews"]
        for e in review_files:
            edges_cnt, nodes_pri = self.create_graph(e)
            start_list = self.get_valid_start_node(e)
            with open('review_edges', 'w') as f:
                for bigram in edges_cnt:
                    f.write(" ".join([bigram[0], bigram[1]]) + 
                            " " + str(edges_cnt[bigram]))
                    f.write("\n")

            G = nx.read_edgelist('review_edges', create_using=nx.DiGraph(),
                                data=(('count',int),))
            #cp = ConfigParser()
            #cp.read("opinosis.properties")
            candidates = self.summarize_candidates(G, nodes_pri, start_list)
            self.remove_duplicates(candidates)

            li = list(candidates.items())
            #li.sort(key=itemgetter(1), reverse=True)
            li = sorted(li, key=itemgetter(1), reverse=True)

            count = 1
            summary = ""
            for e in li:
                if (count == 3):
                    break
                words = e[0].split()
                #words.sdasd
                for w in words:
                    out = w.rsplit("/", 1)
                #words = words.rsplit("/", 1)
                    #print(out[0], end=" ")
                    summary = summary + out[0] + " "
                #print(".")
                summary = summary + "."
                count = count + 1
        print(summary)
        return summary
    def inference(self, text, num_sentences):
        
        #text = "They have traded the 31-year-old to the Brooklyn Nets as part of a three-team deal, the Nets announced on Thursday. In return for Harden, Houston is acquiring Caris LeVert and Rodions Kurucs from the Nets, Dante Exum from the Cleveland Cavaliers, three first-round picks from the Nets, one first-round pick from the Cavaliers via the Milwaukee Bucks, and four first-round pick swaps from the Nets. In a separate deal, Houston is trading LeVert and a second-round pick to the Indiana Pacers for guard and two-time All-Star Victor Oladipo, according to The Athletic's Shams Charania.   Harden, an eight-time All-Star, was acquired by the Rockets from the Oklahoma City Thunder in 2012.  While in Houston, he was voted the league's best player for the 2017-18 season and led the Rockets to the playoffs in all eight years. Visit CNN.com/sport for more news, videos and features \"Adding an All-NBA player such as James to our roster better positions our team to compete against the league's best,\" said Nets General Manager Sean Marks.  \"James is one of the most prolific scorers and playmakers in our game, and we are thrilled to bring his special talents to Brooklyn.\"  The trade comes in the same week that Harden criticized the Rockets.  \"We're just not good enough -- obviously, chemistry, talent-wise, just everything -- and it was clear these last few games,\" Harden said. \"I love this city. I've literally done everything that I can. I mean, this situation, it's crazy. It's something that I don't think can be fixed.\" READ: LeBron James hits no-look three pointer to win mid-game bet with teammate in LA Lakers rout The postgame comments were the last of a string a of negative behavior from the disgruntled star, after arriving late to the team's training camp, and then being sidelined for four days and fined $50,000 by the NBA for violating the league's health and safety protocols days before the start of the season. The former MVP now reunites with former Thunder teammate Kevin Durant and perennial All-Star guard Kyrie Irving in Brooklyn.  Further postponements Meanwhile, the NBA announced on Wednesday that it was postponing its ninth game of the season. Two games scheduled for Friday between the Phoenix Suns and the Golden State Warriros and the Washington Wizards and Detriot Pistons have been postponed as ongoing contact tracing means the Suns and the Wizards do not have the required eight players to contest the games.  Since January 6, 16 NBA players have tested positive for Covid-19.  The league announced on Tuesday that it had adopted stricter health and safety protocols to combat the spread of the virus. "
        self.POSTagger(text)
        review_files = ["tmp_reviews"]
        for e in review_files:
            edges_cnt, nodes_pri = self.create_graph(e)
            start_list = self.get_valid_start_node(e)
            with open('review_edges', 'w') as f:
                for bigram in edges_cnt:
                    f.write(" ".join([bigram[0], bigram[1]]) + 
                            " " + str(edges_cnt[bigram]))
                    f.write("\n")

            G = nx.read_edgelist('review_edges', create_using=nx.DiGraph(),
                                data=(('count',int),), comments="~I do fucking not need comment-")
            #cp = ConfigParser()
            #cp.read("opinosis.properties")
            candidates = self.summarize_candidates(G, nodes_pri, start_list)
            self.remove_duplicates(candidates)

            li = list(candidates.items())
            #li.sort(key=itemgetter(1), reverse=True)
            li = sorted(li, key=itemgetter(1), reverse=True)

            count = 0
            summary = ""
            for e in li:
                if (count == num_sentences):
                    break
                words = e[0].split()
                #words.sdasd
                for w in words:
                    out = w.rsplit("/", 1)
                #words = words.rsplit("/", 1)
                    #print(out[0], end=" ")
                    summary = summary + out[0] + " "
                #print(".")
                summary = summary + "."
                count = count + 1
        return summary
    def create_graph(self, review_file):
        """
        Create a Opinosis graph for the given review_file
        """

        edges = []
        nodes_pri = defaultdict(list)
        
        with open(review_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                #print(line)
                if not line:
                    continue
                words = line.split()
                if (len(words)== 1):
                    words.append('./.')
                words2 = words[1:][:]
                words1 = words[:-1]
                bigram = zip(words1, words2)
                edges.extend(bigram)
                for j, word in enumerate(words):
                    nodes_pri[word].append((i,j))
        edges_cnt = Counter(edges)
        return edges_cnt, nodes_pri
    def get_valid_start_node(self, review_file):
        start_list = []
        with open(review_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                start_list.append(words[0])
        return(start_list)
    def valid_start_node(self, node, nodes_pri, start_list):
        """
        Determine if node is a valid start node
        """
        start_tag = set(["JJ", "RB", "PRP$", "VBG", "NN", "DT"])
        start_word = set(["its", "the", "when", "a", "an", "this", 
                        "the", "they", "it", "i", "we", "our",
                        "if"])
        pri = nodes_pri[node]
        position = [e[1] for e in pri]
        median = np.median(position)
        #START = int(cp.get("section", "start"))
        START = 10
        if median <= START:
            #print(node)
            w, t = node.rsplit("/", 1)
            #if w in start_word or t in start_tag:
            #    return True
            if node in start_list:
                return True
        return False

    def intersect(self, pri_so_far, pri_node):
        """
        Get the overlapping part between pri_so_far and pri_node
        pri_so_far: a list of path, path is represented by a list of (sid, pid).
        pri_node: a list of (sid, pid), this is the presence and position information of the node.
        """
        #GAP = int(cp.get("section", "gap"))
        GAP = 3
        pri_new = []
        for pri in pri_so_far:
            last_sid, last_pid = pri[-1]
            for sid, pid in pri_node:
                if sid == last_sid and pid - last_pid > 0 and pid - last_pid <= GAP:
                    pri = pri[:]
                    pri.append((sid, pid))
                    pri_new.append(pri)
        return pri_new

    def valid_end_node(self, graph, node):
        """
        Determine if node is a valid end node.
        """
        #print(node)
        #if node == ('’/NN'):
        #    assert(1==2)
        if "/." in node:
            return True
        elif len(graph[node]) <= 0:
            return True
        else:
            return False

    def valid_candidate(self, sentence):
        """
        Determine if the sentence is a valid candidate.
        """
        #return True
        sent = " ".join(sentence)
        if (len(sentence) < 1):
            return False
        else:           ### This two line for the reason of comment below
            return True ### 
        """
        last = sentence[-1]
        w, t = last.rsplit("/", 1)
        if t in set(["TO", "VBZ", "IN", "CC", "WDT", "PRP", "DT", ","]):
            return False
        if re.match(".*(/JJ)*.*(/NN)+.*(/VB)+.*(/JJ)+.*", sent):
            return True
        elif re.match(".*(/RB)*.*(/JJ)+.*(/NN)+.*", sent) and not re.match(".*(/DT).*", sent):
            return True
        elif re.match(".*(/PRP|/DT)+.*(/VB)+.*(/RB|/JJ)+.*(/NN)+.*", sent):
            return True
        elif re.match(".*(/JJ)+.*(/TO)+.*(/VB).*", sent):
            return True
        elif re.match(".*(/RB)+.*(/IN)+.*(/NN)+.*", sent):
            return True
        else:
            return False
        """
    def path_score(self, redundancy, sen_len):
        """
        log weghted redundancy score function
        """
        return np.log2(sen_len) * redundancy

    def collapsible(self, node):
        """
        Determine if the node can be a hub.
        """
        return False
        """
        if re.match(".*(/VB[A-Z]|/IN)", node):
            return True
        else:
            return False
        """
    def average_path_score(self, cc):
        return np.mean(cc.values())

    def intersection_sim(self, can1, can2):
        set1 = set(can1.split())
        set2 = set(can2.split())

        return float(len(set1.intersection(set2)))/(len(set1.union(set2)) + 0.001)

    def remove_duplicates(self, cc):
        """
        Use affinity propagation to remove duplicate candidates.
        """
        from sklearn.cluster import affinity_propagation
        li = list(cc)
        sim_matrix = np.zeros((len(li), len(li)))
        for i, e1 in enumerate(li):
            for j, e2 in enumerate(li):
                sim_matrix[i,j] = self.intersection_sim(e1, e2)

        centers, _ = affinity_propagation(sim_matrix, random_state = None)

        for i, e in enumerate(li):
            if i not in centers:
                del cc[e]

    def stitch(self, canchor, cc):
        """
        Stitch the anchor sentence and the collpased part together.
        """
        if len(cc) == 1:
            return list(cc)[0]
        return " xx ".join(list(cc))
        sents = list(cc)
        anchor_str = " ".join(canchor)
        anchor_len = len(anchor_str)
        sents = [e[anchor_len:] for e in sents]
        sents = [e for e in sents if e.strip() != "./." and e.strip() != ",/,"]
        s = anchor_str + " xx " + " AND ".join(sents)
        return s + " ."

    def traverse(self, graph, nodes_pri, node, sentence, pri_so_far, score, clist, collapsed):
        """
        Traverse a path.
        """
        #if node == ('’/NNP'):
        #    assert(1==2)
        # Don't allow sentence that are too long to avoid looping forever.
        if len(sentence) > 20:
            return 
        redundancy = len(pri_so_far)
        #REDUNDANCY_THRESHOLD = int(cp.get("section", "redundancy"))
        REDUNDANCY_THRESHOLD = 1
        if redundancy >= REDUNDANCY_THRESHOLD or self.valid_end_node(graph, node):
            if self.valid_end_node(graph, node):
                #Removing the punctuation at the end. 
                del sentence[-1]
                if self.valid_candidate(sentence):
                    if(len(sentence) == 0):
                        final_score = 0
                    else:
                        final_score = score/float(len(sentence))
                    clist[" ".join(sentence)] = score
                return

            # Traversing the neighbors
            for neighbor in graph[node]:
                redundancy = len(pri_so_far)
                new_sentence = sentence[:]
                new_sentence.append(neighbor)
                new_score = score + self.path_score(redundancy, len(new_sentence))
                pri_new = self.intersect(pri_so_far, nodes_pri[neighbor])
                
                #If the neighbor is collapsible and not already collapsed, collapse it.
                if self.collapsible(neighbor) and not collapsed:
                    canchor = new_sentence
                    cc = defaultdict(int)
                    anchor_score = new_score + self.path_score(redundancy, len(new_sentence)+1)
                    for vx in graph[neighbor]:
                        pri_vx = self.intersect(pri_new, nodes_pri[vx])
                        vx_sentence = new_sentence[:]
                        vx_sentence.append(vx)
                        self.traverse(graph, nodes_pri, vx, vx_sentence, 
                                pri_vx, anchor_score, cc, True)
                    
                    if cc:
                        self.remove_duplicates(cc)
                        cc_path_score = self.average_path_score(cc)
                        final_score = float(anchor_score)/len(new_sentence) + cc_path_score
                        stitched_sent = self.stitch(canchor, cc)
                        clist[stitched_sent] = final_score
                    
                else:
                    self.traverse(graph, nodes_pri, neighbor, new_sentence,
                            pri_new, new_score, clist, False)

    def summarize_candidates(self, graph, nodes_pri, start_list):
        """
        Create summaries from the Opinosis graph. 
        """
        nodes_size = len(nodes_pri)
        candidates = defaultdict(int)
        for node in nodes_pri:
            if self.valid_start_node(node, nodes_pri, start_list):
                score = 0
                clist = defaultdict(int)
                sentence = [node]
                pri = nodes_pri[node]
                pri_so_far = [[e] for e in pri] 
                self.traverse(graph, nodes_pri, node, sentence, 
                        pri_so_far, score, clist, False)
                candidates.update(clist)

        return candidates

    #cp = None

if __name__ == '__main__':
    text = "They have traded the '’/NN'31-year-old to the Brooklyn Nets as part of a three-team deal, the Nets announced on Thursday. In return for Harden, Houston is acquiring Caris LeVert and Rodions Kurucs from the Nets, Dante Exum from the Cleveland Cavaliers, three first-round picks from the Nets, one first-round pick from the Cavaliers via the Milwaukee Bucks, and four first-round pick swaps from the Nets. In a separate deal, Houston is trading LeVert and a second-round pick to the Indiana Pacers for guard and two-time All-Star Victor Oladipo, according to The Athletic's Shams Charania.   Harden, an eight-time All-Star, was acquired by the Rockets from the Oklahoma City Thunder in 2012.  While in Houston, he was voted the league's best player for the 2017-18 season and led the Rockets to the playoffs in all eight years. Visit CNN.com/sport for more news, videos and features \"Adding an All-NBA player such as James to our roster better positions our team to compete against the league's best,\" said Nets General Manager Sean Marks.  \"James is one of the most prolific scorers and playmakers in our game, and we are thrilled to bring his special talents to Brooklyn.\"  The trade comes in the same week that Harden criticized the Rockets.  \"We're just not good enough -- obviously, chemistry, talent-wise, just everything -- and it was clear these last few games,\" Harden said. \"I love this city. I've literally done everything that I can. I mean, this situation, it's crazy. It's something that I don't think can be fixed.\" READ: LeBron James hits no-look three pointer to win mid-game bet with teammate in LA Lakers rout The postgame comments were the last of a string a of negative behavior from the disgruntled star, after arriving late to the team's training camp, and then being sidelined for four days and fined $50,000 by the NBA for violating the league's health and safety protocols days before the start of the season. The former MVP now reunites with former Thunder teammate Kevin Durant and perennial All-Star guard Kyrie Irving in Brooklyn.  Further postponements Meanwhile, the NBA announced on Wednesday that it was postponing its ninth game of the season. Two games scheduled for Friday between the Phoenix Suns and the Golden State Warriros and the Washington Wizards and Detriot Pistons have been postponed as ongoing contact tracing means the Suns and the Wizards do not have the required eight players to contest the games.  Since January 6, 16 NBA players have tested positive for Covid-19.  The league announced on Tuesday that it had adopted stricter health and safety protocols to combat the spread of the virus. "
    #'’/NN'
    my_text = "i am happy.'"
    model = opinosis()
    model.summarize(my_text)

