#!/usr/bin/env python
# The MIT License (MIT)
# Copyright © 2015 Recruit Technologies Co.,Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import collections
import re
import numpy as np
import networkx as nx

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise_distances

from janome.tokenizer import Tokenizer
__doc__ = """Japanese summarization module using LexRank algorithm.
This module was reconsturcted from summpy.

   - https://pypi.python.org/pypi/summpy/
   - https://github.com/recruit-tech/summpy
   - https://recruit-tech.co.jp/blog/2015/10/30/summpy-released/

Requirements

   - numpy
   - networkx
   - scipy
   - scikit-learn
   - janome

"""
__version__ = "0.1.2"
__author__ = "Shumpei IINUMA"
__maintainer__ = "Hajime Nakagami<nakagami@gmail.com>"
__all__ = ['summarize']
tokenizer = Tokenizer()


def word_splitter_ja(sent):
    def _is_stopword(n):
        if len(n.surface) == 0:
            return True
        elif re.search(r'^[\s!-@\[-`\{-~　、-〜！-＠［-｀]+$', n.surface):
            return True
        elif re.search(r'^(接尾|非自立)', n.part_of_speech.split(',')[1]):
            return True
        elif 'サ変・スル' == n.infl_form or u'ある' == n.base_form:
            return True
        elif re.search(r'^(名詞|動詞|形容詞)', n.part_of_speech.split(',')[0]):
            return False
        else:
            return True

    return [n.base_form for n in tokenizer.tokenize(sent) if not _is_stopword(n)]


def sent_splitter_ja(text, delimiters=set('。．？！\n\r'), parenthesis='（）「」『』“”'):
    '''
    Args:
      text: string that contains multiple Japanese sentences.
      delimiters: set() of sentence delimiter characters.
      parenthesis: to be checked its correspondence.
    Returns:
      list of sentences.
    '''
    paren_chars = set(parenthesis)
    close2open = dict(zip(parenthesis[1::2], parenthesis[0::2]))

    sentences = []
    pstack = []
    buff = []

    for i, c in enumerate(text):
        c_next = text[i+1] if i+1 < len(text) else None
        # check correspondence of parenthesis
        if c in paren_chars:
            if c in close2open:  # close
                if len(pstack) > 0 and pstack[-1] == close2open[c]:
                    pstack.pop()
            else:  # open
                pstack.append(c)

        buff.append(c)
        if c in delimiters:
            if len(pstack) == 0 and c_next not in delimiters:
                s = ''.join(buff).strip()
                if s:
                    sentences.append(s)
                buff = []
    if len(buff) > 0:
        s = ''.join(buff).strip()
        if s:
            sentences.append(s)

    return sentences


def lexrank(
    sentences, continuous=False, word_splitter=word_splitter_ja, sim_threshold=0.1, alpha=0.9
):
    '''
    compute centrality score of sentences.

    Args:
      sentences: [u'こんにちは．', u'私の名前は飯沼です．', ... ]
      continuous: if True, apply continuous LexRank. (see reference)
      word_splitter: function to spilit to words
      sim_threshold: if continuous is False and smilarity is greater or
        equal to sim_threshold, link the sentences.
      alpha: the damping factor of PageRank

    Returns: tuple
      (
        {
          # sentence index -> score
          0: 0.003,
          1: 0.002,
          ...
        },
        similarity_matrix
      )

    Reference:
      Günes Erkan and Dragomir R. Radev.
      LexRank: graph-based lexical centrality as salience in text
      summarization. (section 3)
      http://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html
    '''
    # configure ranker
    ranker_params = {'max_iter': 1000}
    ranker = nx.pagerank_scipy
    ranker_params['alpha'] = alpha

    graph = nx.DiGraph()

    # sentence -> tf
    sent_tf_list = []
    for sent in sentences:
        words = word_splitter(sent)
        tf = collections.Counter(words)
        sent_tf_list.append(tf)

    sent_vectorizer = DictVectorizer(sparse=True)
    sent_vecs = sent_vectorizer.fit_transform(sent_tf_list)

    # compute similarities between senteces
    sim_mat = 1 - pairwise_distances(sent_vecs, sent_vecs, metric='cosine')

    linked_rows, linked_cols = np.where(
        sim_mat > 0 if continuous else sim_mat >= sim_threshold
    )

    # create similarity graph
    graph.add_nodes_from(range(sent_vecs.shape[0]))
    for i, j in zip(linked_rows, linked_cols):
        if i != j:
            weight = sim_mat[i, j] if continuous else 1.0
            graph.add_edge(i, j, weight=weight)

    scores = ranker(graph, **ranker_params)
    return scores, sim_mat


def summarize(sentences, sent_limit=None, char_limit=None, imp_require=None, **lexrank_params):
    '''
    Args:
      sentences: text to be summarized or list of sentence
      sent_limit: summary length (the number of sentences)
      char_limit: summary length (the number of characters)
      imp_require: cumulative LexRank score [0.0-1.0]

    Returns:
      list of extracted sentences
    '''
    if isinstance(sentences, str):
        sentences = sent_splitter_ja(sentences)

    scores, sim_mat = lexrank(sentences, **lexrank_params)
    sum_scores = sum(scores.values())
    acc_scores = 0.0
    indexes = set()
    num_sent, num_char = 0, 0
    for i in sorted(scores, key=lambda i: scores[i], reverse=True):
        num_sent += 1
        num_char += len(sentences[i])
        if sent_limit is not None and num_sent > sent_limit:
            break
        if char_limit is not None and num_char > char_limit:
            break
        if imp_require is not None and acc_scores / sum_scores >= imp_require:
            break
        indexes.add(i)
        acc_scores += scores[i]

    return [sentences[i] for i in sorted(indexes)]

def text_format(text):
    print(text)
    #lines = text.split('\n')
    #for line in lines:
    #    print(line)
    #    print(type(line))
    #    match_text = re.match(r"\[質問内容\]", line)
    #    if match_text:
    #        pass
    #    else:
    #        line = ""
    #print(lines)

    text = re.sub(r"\[.*\].*", "", text)

    print(text)

if __name__ == '__main__':
    def _get_bocchan_text():
        import io
        import requests
        import zipfile

        r = requests.get('https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip')
        f = zipfile.ZipFile(io.BytesIO(r.content)).open('bocchan.txt')

        text = f.read().decode('cp932')
        text = re.sub(r'《[^》]+》', '', text)
        text = re.sub(r'｜', '', text)
        text = re.sub(r'［.+?］', '', text)
        text = re.sub(r'-----[\s\S]*-----', '', text)
        text = re.split('底本：', text)[0]

        return text
    
    def _get_commandline_text():
        import sys

        #text = input("メールテキスト：")
        text_file = sys.argv[1]

        text = open(text_file).read()

        return text

    #text = _get_bocchan_text()
    text = _get_commandline_text()

    format_text = text_format(text)

    #result = summarize(text, char_limit=200)
    #print('\n\nsummarize(char_limit=200)')
    #print('\n'.join(result))

    #result = summarize(text, sent_limit=3)
    #print('\n\nsummarize(sent_limit=3)')
    #print('\n'.join(result))

    #result = summarize(text, char_limit=200, continuous=True)
    #print('\n\nsummarize(char_limit=200, continuous=True)')
    #print('\n'.join(result))

    #result = summarize(text, sent_limit=3, continuous=True)
    #print('\n\nsummarize(sent_limit=3, continuous=True)')
    #print('\n'.join(result))