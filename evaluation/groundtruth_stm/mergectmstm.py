#!/usr/bin/env python

import sys
import pdb

ctmFile = sys.argv[1]
stmFile = sys.argv[2]

ctm = open(ctmFile, "r")
stm = open(stmFile, "r")

ctmDict = []
stmDict = []
addedlines = 0

for idx, line in enumerate(ctm):
    l = line.strip().split()
    ctmDict.append(l)

for idx, line in enumerate(stm):
    l = line.strip().split()
    stmDict.append(l)
    if len(ctmDict) > idx + addedlines and ctmDict[idx + addedlines][0] == l[0]:  # ctm and stm match:
        if len(ctmDict) > idx + addedlines + 1:
            while (len(ctmDict) > idx + addedlines + 1) and (ctmDict[idx + addedlines + 1][0] == l[0]):
                addedlines += 1
    else:
        ctmDict.insert(idx + addedlines, [l[0], "1 0.000 0.030 [EMPTY]"])

stm.close()
ctm.close()
ctm = open(ctmFile, "w+")

for l in ctmDict:
    ctm.write(" ".join(l) + "\n")
ctm.close()
