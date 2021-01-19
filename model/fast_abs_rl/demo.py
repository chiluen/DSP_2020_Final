""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

#from data.batcher import tokenize

from .decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from .decoding import make_html_safe
import nltk


class fast_abs():
    def __init__(self):
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "pretrained/acl"
        self.model_dir=os.path.join(script_dir, rel_path)
        self.beam_size=5
        self.diverse=1.0
        self.max_len=30
        self.cuda = torch.cuda.is_available()
    def summarize(self, text):
        sentences = nltk.sent_tokenize(text)
        sent_list = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            sent_list.append(words)
        summary = self.decode(self.model_dir, self.beam_size, self.diverse, self.max_len, self.cuda, sent_list)
        print(summary)
        return summary
    def inference(self, text):
        sentences = nltk.sent_tokenize(text)
        sent_list = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            sent_list.append(words)
        summary = self.decode(self.model_dir, self.beam_size, self.diverse, self.max_len, self.cuda, sent_list)
        return summary
    def decode(self, model_dir, beam_size, diverse, max_len, cuda, sent_list):
        start = time()
        summary = ""
        # setup model
        with open(join(model_dir, 'meta.json')) as f:
            meta = json.loads(f.read())
        if meta['net_args']['abstractor'] is None:
            # NOTE: if no abstractor is provided then
            #       the whole model would be extractive summarization
            assert beam_size == 1
            abstractor = identity
        else:
            if beam_size == 1:
                abstractor = Abstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
            else:
                abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                            max_len, cuda)
        extractor = RLExtractor(model_dir, cuda=cuda)

        # Decoding
        i = 0
        with torch.no_grad():
            ext_arts = []
            ext_inds = []
            raw_art_sents = sent_list

            ext = extractor(raw_art_sents)[:-1]  # exclude EOE
            if not ext:
                ext = list(range(5))[:len(raw_art_sents)]
            else:
                ext = [i.item() for i in ext]
            ext_inds += [(len(ext_arts), len(ext))]
            ext_arts += [raw_art_sents[i] for i in ext]
            if beam_size > 1:
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = self.rerank_mp(all_beams, ext_inds)
            else:
                dec_outs = abstractor(ext_arts)
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                for sent in decoded_sents:
                    #print(sent)
                    summary = summary + sent + "\n"
        return summary
    _PRUNE = defaultdict(
        lambda: 2,
        {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
    )

    def rerank(self, all_beams, ext_inds):
        beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
        return list(concat(map(self.rerank_one, beam_lists)))

    def rerank_mp(self, all_beams, ext_inds):
        beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
        with mp.Pool(8) as pool:
            reranked = pool.map(self.rerank_one, beam_lists)
        return list(concat(reranked))

    def rerank_one(self, beams):
        _PRUNE = defaultdict(
            lambda: 2,
            {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
        )
        @curry
        
        def process_beam(beam, n):
            for b in beam[:n]:
                b.gram_cnt = Counter(self._make_n_gram(b.sequence))
            return beam[:n]
        beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
        best_hyps = max(product(*beams), key=self._compute_score)
        dec_outs = [h.sequence for h in best_hyps]
        return dec_outs

    def _make_n_gram(self, sequence, n=2):
        return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

    def _compute_score(self, hyps):
        all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
        repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
        lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
        return (-repeat, lp)


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', help='path to store/eval', default='my_results/')
    parser.add_argument('--model_dir', help='root of the full model', default='pretrained/acl')

    # dataset split
    #data = parser.add_mutually_exclusive_group(required=True)
    #data.add_argument('--val', action='store_true', help='use validation set')
    #data.add_argument('--test', action='store_true', help='use test set',default='test')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    #data_split = 'test'
    #model_dir = 'pretrained/acl'
    #beam = '1'
    #print(args.model_dir)
    decode(args.model_dir, args.beam, args.div,
           args.max_dec_word, args.cuda)
    """
    text = "They have traded the 31-year-old to the Brooklyn Nets as part of a three-team deal, the Nets announced on Thursday. In return for Harden, Houston is acquiring Caris LeVert and Rodions Kurucs from the Nets, Dante Exum from the Cleveland Cavaliers, three first-round picks from the Nets, one first-round pick from the Cavaliers via the Milwaukee Bucks, and four first-round pick swaps from the Nets. In a separate deal, Houston is trading LeVert and a second-round pick to the Indiana Pacers for guard and two-time All-Star Victor Oladipo, according to The Athletic's Shams Charania.   Harden, an eight-time All-Star, was acquired by the Rockets from the Oklahoma City Thunder in 2012.  While in Houston, he was voted the league's best player for the 2017-18 season and led the Rockets to the playoffs in all eight years. Visit CNN.com/sport for more news, videos and features \"Adding an All-NBA player such as James to our roster better positions our team to compete against the league's best,\" said Nets General Manager Sean Marks.  \"James is one of the most prolific scorers and playmakers in our game, and we are thrilled to bring his special talents to Brooklyn.\"  The trade comes in the same week that Harden criticized the Rockets.  \"We're just not good enough -- obviously, chemistry, talent-wise, just everything -- and it was clear these last few games,\" Harden said. \"I love this city. I've literally done everything that I can. I mean, this situation, it's crazy. It's something that I don't think can be fixed.\" READ: LeBron James hits no-look three pointer to win mid-game bet with teammate in LA Lakers rout The postgame comments were the last of a string a of negative behavior from the disgruntled star, after arriving late to the team's training camp, and then being sidelined for four days and fined $50,000 by the NBA for violating the league's health and safety protocols days before the start of the season. The former MVP now reunites with former Thunder teammate Kevin Durant and perennial All-Star guard Kyrie Irving in Brooklyn.  Further postponements Meanwhile, the NBA announced on Wednesday that it was postponing its ninth game of the season. Two games scheduled for Friday between the Phoenix Suns and the Golden State Warriros and the Washington Wizards and Detriot Pistons have been postponed as ongoing contact tracing means the Suns and the Wizards do not have the required eight players to contest the games.  Since January 6, 16 NBA players have tested positive for Covid-19.  The league announced on Tuesday that it had adopted stricter health and safety protocols to combat the spread of the virus. "
    model = fast_abs()
    model.summarize(text)