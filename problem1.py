from CDR import *
import argparse
import os
from num2words import num2words


def build47():
    word_ls = []
    startNull = NullState()
    startWord = Word([num2words(i) for i in range(2, 10)], 0)
    word_ls.append(startWord)
    currentNull = appendWord(startNull, startWord)
    specialNull = None
    for i in range(1, 7):
        digits = [num2words(i) for i in range(10)]
        currentWord = Word(digits, i)
        word_ls.append(currentWord)
        currentNull = appendWord(currentNull, currentWord)
        if i == 2:
            specialNull = currentNull
    return startNull, specialNull


def RSS(sentence, node_ls, nodeNum, branchNull, force7digit):
    startNull = node_ls[0]
    startNull.currDis = 0
    if force7digit:
        branchNull.currDis = 0
    bpt = [[None for i in range(nodeNum)] for j in range(len(sentence))]
    # print(len(bpt))
    for t in range(len(sentence)):
        vector = sentence[t]
        for currentNode in node_ls:
            currentNode.prevDis = currentNode.currDis
        for currentNode in node_ls:
            currentNode.prevDis = currentNode.currDis
            if currentNode == startNull:
                continue
            if currentNode in startNull.next:
                if t == 0:
                    # at time 0, parentNode distance is 0
                    currentNode.currDis = currentNode.getDis(vector)
                    bpt[t][currentNode.id] = currentNode
                else:
                    # at other time t, can only self loop, parentNode is itself
                    currentNode.currDis = currentNode.prevDis + currentNode.getDis(vector)
                    bpt[t][currentNode.id] = currentNode
                continue
            if not force7digit:
                if currentNode in branchNull.next:
                    if t == 0:
                        # at time t, parentNode distance is 0
                        currentNode.currDis = currentNode.getDis(vector)
                        bpt[t][currentNode.id] = currentNode
                        continue
            parentDis = []
            for edge in currentNode.edges:
                parent = edge[0]
                transition = edge[1]
                if parent.isNull:
                    parentDis.append(parent.currDis - np.log(transition))
                else:
                    parentDis.append(parent.prevDis - np.log(transition))
            minIdx = np.argmin(parentDis)
            minParent = currentNode.edges[minIdx][0]
            if minParent.isNull:
                minParent = bpt[t - 1][minParent.id]
            distance = parentDis[minIdx]
            bpt[t][currentNode.id] = minParent
            if currentNode.isNull:
                currentNode.currDis = distance
            else:
                currentNode.currDis = distance + currentNode.getDis(vector)
    return parseBPT(bpt)


def main(args):
    sentence = args.ss
    p1_folder = "./p1_sentence"
    if sentence != None:
        answer = utils.parseSName(sentence)
        filename = os.path.join(p1_folder, sentence)
        sentence = mfcc.mfcc_features(filename, 40)
        startNull, branchNull = build47()
        node_ls, nodeNum = flatten(startNull)
        result, seq = RSS(sentence, node_ls, nodeNum, branchNull, False)
        total = len(answer)
        count = 0
        for i in range(total):
            try:
                if answer[i] == result[i]:
                    count += 1
            except:
                break
        print("Result: ", result)
        print("Correct rate: {:.2f}".format(count / total))
        minEditDis = utils.dtw(answer, result)
        print("Minimum edit distance: {}\nWord error rate: {:.2f}".format(minEditDis, minEditDis / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="tz")
    parser.add_argument("--ss", default=None)
    args = parser.parse_args()
    main(args)
