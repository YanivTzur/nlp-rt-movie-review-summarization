#!/Users/liatvi/anaconda3/bin/python

from pytextrank import json_iter, parse_doc, pretty_print
import sys

## Stage 1:
##  * perform statistical parsing/tagging on a document in JSON format
##
## INPUTS: <stage0>
## OUTPUT: JSON format `ParsedGraf(id, sha1, graf)`

# if __name__ == "__main__":
#     path_stage0 = sys.argv[1]

#     for graf in parse_doc(json_iter(path_stage0)):
#         print(pretty_print(graf._asdict()))

def stage1(jsonData):

    # yield json.loads(line)
    # for graf in parse_doc(json_iter(jsonData)):
    #     print(pretty_print(graf._asdict()))
    analyzing = []
    for graf in parse_doc( (data for data in [jsonData])):
        analyzing.append(graf._asdict())
        # analyzing.append(pretty_print(graf._asdict()))
    return analyzing
