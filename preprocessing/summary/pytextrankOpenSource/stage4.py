#!/Users/liatvi/anaconda3/bin/python

from pytextrank import limit_keyphrases, limit_sentences, make_sentence
import sys

## Stage 4:
##  * summarize a document based on most significant sentences and key phrases
##
## INPUTS: <stage2> <stage3>
## OUTPUT: Markdown format

if __name__ == "__main__":
    path_stage2 = sys.argv[1]
    # path_stage3 = sys.argv[2]

    phrases = ", ".join(set([p for p in limit_keyphrases(path_stage2, phrase_limit=12)]))
    # sent_iter = sorted(limit_sentences(path_stage3, word_limit=150), key=lambda x: x[1])
    # s = []

    # for sent_text, idx in sent_iter:
    #     s.append(make_sentence(sent_text))

    # graf_text = " ".join(s)
    # print("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))
    print(phrases)

def stage4(jsonData):
    return list(set([p for p in limit_keyphrases((data for data in jsonData), phrase_limit=12)]))