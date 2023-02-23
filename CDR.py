from num2words import num2words
import queue
import numpy as np
import utils
import mfcc


class Node:
    """
    Pre-trained single gaussian state
    """

    def __init__(self, mean, cov, name):
        """
        :param name: digit the state belongs to
        :param id: unique identifier in bpt
        :param visited: if visited in BFS
        :param isNull: if non-emitting state
        :param edges: list of all incomming paths, each entry a tuple (parent node, transition prob)
        :param next: next state, single node object, excluding self loop
        """
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
        """ 
        Return the negative log-likelihood
        """
        cov_diag = np.where(self.cov == 0, np.finfo(np.float64).eps, self.cov)
        try:
            ret = np.sum(np.log10(2 * np.pi * cov_diag))
        except FloatingPointError:
            print(cov_diag)
        diff = vector - self.mean
        ret += np.sum(np.square(diff) / cov_diag)

        return ret * (0.5)


class NullState:
    """
    Non-emitting state
    """

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
    """
    Hidden Markov Model
    """

    def __init__(self, name, idx):
        """
        :param name: digit
        :param head: head node 
        :param Node_ls: list include all nodes with order
        :param num: node number
        """
        self.digit = name
        self.name = name + str(idx)
        self.head = None
        self.Node_ls = []
        self.num = 0
        self.parseHMM()

    def parseHMM(self):
        """
        Read and parse local hmm model
        """
        model = utils.load_hmm(self.digit)
        state_ls = model.state_ls
        trans_mat = model.trans_mat
        self.num = len(state_ls)
        firstState = state_ls[0]
        # head node of the hmm
        head = Node(firstState[0], firstState[1], self.name)
        # temporarily add only one edge for head node
        head.edges.append((head, trans_mat[0][0]))
        self.head = head
        self.Node_ls.append(head)
        prevNode = head
        # initialize all other node and connect them
        for i in range(1, len(state_ls)):
            currState = state_ls[i]
            currNode = Node(currState[0], currState[1], self.name)
            prevNode.next.append(currNode)
            # set the self loop probability of tail node to 0.5
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
    """
    
    """

    def __init__(self, digits, idx):
        self.hmm_ls = []
        self.idx = idx
        self.parseDigits(digits)

    def parseDigits(self, digits):
        for digit in digits:
            hmm = Hmm(digit, self.idx)
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
    null_ls = []
    startNull = NullState()
    null_ls.append(startNull)
    startWord = Word([num2words(i) for i in range(2, 10)], 0)
    currentNull = appendWord(startNull, startWord)
    null_ls.append(currentNull)
    specialNull = None
    for i in range(1, 7):
        digits = [num2words(i) for i in range(10)]
        currentWord = Word(digits, i)
        currentNull = appendWord(currentNull, currentWord)
        null_ls.append(currentNull)
        if i == 2:
            """ startNull.next.append(currentNull)
            currentNull.edges.append((startNull, 1.0)) """
            specialNull = currentNull
    """ for i in range(len(null_ls)):
        print(i, null_ls[i]) """
    return startNull, specialNull


def parseBPT(bpt):
    seq = [None for i in range(len(bpt))]
    t = len(bpt) - 1
    currNode = bpt[t][-1]
    currID = currNode.id
    seq[t] = currNode.name
    t -= 1
    while t >= 0:
        currNode = bpt[t][currID]
        # print(currNode.name)
        try:
            currID = currNode.id
        except:
            print(t)
            print(currNode)
        seq[t] = currNode.name
        t -= 1
    currDigit = seq[0]
    digit_seq = [currDigit]
    for i in range(len(seq)):
        if seq[i] == currDigit:
            continue
        else:
            digit_seq.append(seq[i])
            currDigit = seq[i]

    return digit_seq, seq


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
    return node_ls, currID


def recog_SS(filename, node_ls, nodeNum, specialNull):
    sentence = mfcc.mfcc_features(filename, 40)
    # print(sentence.shape)
    node_ls[0].currDis = 0
    specialNull.currDis = 0
    bpt = [[None for i in range(nodeNum)] for j in range(len(sentence))]
    # print(len(bpt))
    for t in range(len(sentence)):
        vector = sentence[t]
        for currentNode in node_ls:
            if len(currentNode.edges) == 0:
                continue
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
                if minParent == specialNull and t == 0:
                    minParent = currentNode
                if minParent.isNull:
                    minParent = bpt[t][minParent.id]
                distance = parentDis[minIdx]
                bpt[t][currentNode.id] = minParent
            if currentNode.isNull:
                currentNode.currDis = distance
            else:
                currentNode.currDis = distance + currentNode.getDis(vector)
    return bpt
