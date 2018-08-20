#!/Users/liatvi/anaconda3/bin/python

from pytextrank import normalize_key_phrases, pretty_print, render_ranks, text_rank
import pytextrank as pytextrank
import networkx as nx
import sys

## Stage 2:
##  * collect and normalize the key phrases from a parsed document
##
## INPUTS: <stage1>
## OUTPUT: JSON format `RankedLexeme(text, rank, ids, pos)`

if __name__ == "__main__":
    path_stage1 = sys.argv[1]

    graph, ranks = text_rank(path_stage1)
    render_ranks(graph, ranks)

    for rl in normalize_key_phrases(path_stage1, ranks):
        print(pretty_print(rl._asdict()))

def stage2(jsonData):
    analyzing = []

    graph = pytextrank.pytextrank.build_graph((data for data in jsonData))
    ranks = nx.pagerank(graph)

    # graph, ranks = text_rank(path_stage1)
    render_ranks(graph, ranks)

    for rl in normalize_key_phrases((data for data in jsonData), ranks):
        analyzing.append(rl._asdict())
    return analyzing

    # for graf in parse_doc( (data for data in [jsonData])):
    #     analyzing.append(pretty_print(graf._asdict()))
    # return analyzing


    # def text_rank (path):
    # """
    # run the TextRank algorithm
    # """
    # graph = build_graph(json_iter(path))
    # ranks = nx.pagerank(graph)

    # return graph, ranks