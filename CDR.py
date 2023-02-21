from num2words import num2words
import queue
import numpy as np
import utils
import mfcc


class Node:
    def __init__(self, mean, cov, name):
        self.mean = mean
        self.cov = cov
        self.edges = []
        self.next = []
        self.currDis = np.inf
        self.prevDis = np.inf
        self.visited = False
        self.isNull = False
        self.id = None
        self.name = name

    def getDis(self, vector):
        cov_diag = np.where(self.cov == 0, np.finfo(np.float64).eps, self.cov)
        try:
            ret = np.sum(np.log10(2 * np.pi * cov_diag))
        except FloatingPointError:
            print(cov_diag)
        diff = vector - self.mean
        ret += np.sum(np.square(diff) / cov_diag)

        return ret * (0.5)


class NullState:
    def __init__(self):
        self.edges = []
        self.next = []
        self.currDis = np.inf
        self.prevDis = np.inf
        self.visited = False
        self.isNull = True
        self.id = None
        self.name = "*"


class Hmm:
    def __init__(self, name):
        self.name = name
        self.head = None
        self.Node_ls = []
        self.num = 0
        self.parseHMM()

    def parseHMM(self):
        model = utils.load_hmm(self.name)
        state_ls = model.state_ls
        trans_mat = model.trans_mat
        self.num = len(state_ls)
        firstState = state_ls[0]
        head = Node(firstState[0], firstState[1], self.name)
        head.edges.append((head, trans_mat[0][0]))
        self.head = head
        self.Node_ls.append(head)
        prevNode = head
        for i in range(1, len(state_ls)):
            currState = state_ls[i]
            currNode = Node(currState[0], currState[1], self.name)
            prevNode.next.append(currNode)
            if i == len(state_ls) - 1:
                currNode.edges.append((currNode, 0.5))
            else:
                currNode.edges.append((currNode, trans_mat[i][i]))
            currNode.edges.append((prevNode, trans_mat[i - 1][i]))
            self.Node_ls.append(currNode)
            prevNode = currNode

    def getTail(self):
        return self.Node_ls[-1]

    def getHead(self):
        return self.head


class Word:
    def __init__(self, digits):
        self.hmm_ls = []
        self.parseDigits(digits)

    def parseDigits(self, digits):
        for digit in digits:
            hmm = Hmm(digit)
            self.hmm_ls.append(hmm)

    def getAllHeads(self):
        head_ls = []
        for hmm in self.hmm_ls:
            head_ls.append(hmm.getHead())
        return head_ls

    def getAllTails(self):
        tail_ls = []
        for hmm in self.hmm_ls:
            tail_ls.append(hmm.getTail())
        return tail_ls


def appendWord(nullState, word):
    head_ls = word.getAllHeads()
    for head in head_ls:
        head.edges.append((nullState, 1.0))
        nullState.next.append(head)
    tail_ls = word.getAllTails()
    nextNull = NullState()
    for tail in tail_ls:
        tail.next.append(nextNull)
        nextNull.edges.append((tail, 0.5))
    return nextNull


def build47():
    word_ls = []
    startNull = NullState()
    startWord = Word([num2words(i) for i in range(2, 10)])
    word_ls.append(startWord)
    currentNull = appendWord(startNull, startWord)
    for i in range(1, 7):
        digits = [num2words(i) for i in range(10)]
        currentWord = Word(digits)
        word_ls.append(currentWord)
        currentNull = appendWord(currentNull, currentWord)
        if i == 2:
            startNull.next.append(currentNull)
            currentNull.edges.append((startNull, 1.0))
    return startNull, currentNull


def parseBPT(bpt):
    seq = [None for i in range(len(bpt))]
    t = len(bpt) - 1
    currNode = bpt[t][-1]
    currID = currNode.id
    seq[t] = currNode.name
    t -= 1
    while t >= 0:
        currNode = bpt[t][currID]
        currID = currNode.id
        seq[t] = currNode.name
        t -= 1

    currDigit = seq[0]
    digit_seq = currDigit
    for i in range(len(seq)):
        if seq[i] == currDigit:
            continue
        else:
            digit_seq = digit_seq + seq[i]
            currDigit = seq[i]

    return digit_seq


def flatten(headNull):
    currID = 0
    node_ls = []
    q = queue.Queue()
    headNull.visited = True
    q.put(headNull)
    while not q.empty():
        node = q.get()
        node.id = currID
        currID += 1
        node_ls.append(node)
        for child in node.next:
            if child.visited:
                continue
            child.visited = True
            q.put(child)
    return node_ls


def recog_SS(filename, node_ls, nodeNum):
    sentence = mfcc.mfcc_features(filename, 40)
    # print(sentence.shape)
    node_ls[0].currDis = 0
    node_ls[0].prevDis = 0
    bpt = [[None for i in range(nodeNum)] for j in range(len(sentence))]
    # print(len(bpt))
    for t in range(len(sentence)):
        vector = sentence[t]
        for currentNode in node_ls:
            currentNode.prevDis = currentNode.currDis
            parentDis = []
            for edge in currentNode.edges:
                parent = edge[0]
                transition = edge[1]
                parentDis.append(parent.prevDis + np.log(transition))
            if len(parentDis) == 0:
                bpt[t][currentNode.id] = currentNode
                distance = 0
            else:
                minIdx = np.argmin(parentDis)
                minParent = currentNode.edges[minIdx][0]
                distance = parentDis[minIdx]
                bpt[t][currentNode.id] = minParent
            if currentNode.isNull:
                currentNode.currDis = distance
            else:
                currentNode.currDis = distance + currentNode.getDis(vector)
    print(parseBPT(bpt))
    return bpt


def main():
    """ word_ls = []
    currentID = 0
    startNull = NullState(currentID)
    currentID += 1
    startWord = Word([num2words(i) for i in range(2, 10)], currentID)
    currentID = startWord.getID()
    word_ls.append(startWord)
    currentNull = appendWord(startNull, startWord, currentID)
    # utils.dfs_print(startNull)
    for child in startNull.next:
        print(child)
        for c in child.next:
            print(c)
    print(utils.countNode(startNull)) """
    startNull, currentNull = build47()
    node_ls = flatten(startNull)
    # utils.dfs_print(startNull)
    recog_SS("./sentence/test.wav", node_ls, 348)
    # print(utils.countNode(startNull))


main()

