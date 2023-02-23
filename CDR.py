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
    Representation of all candidate HMMs for a digit position, a set of parallel HMMs
    """

    def __init__(self, digits, idx):
        """
        :param digits: a list of candidate digit
        :param idx: index of the position in the numbers
        """
        self.hmm_ls = []
        self.idx = idx
        self.parseDigits(digits)

    def parseDigits(self, digits):
        """
        load candidate HMM models
        """
        for digit in digits:
            hmm = Hmm(digit, self.idx)
            self.hmm_ls.append(hmm)

    def getAllHeads(self):
        """
        Return the first state of all HMMs
        """
        head_ls = []
        for hmm in self.hmm_ls:
            head_ls.append(hmm.getHead())
        return head_ls

    def getAllTails(self):
        """
        Return the last state of all HMMs
        """
        tail_ls = []
        for hmm in self.hmm_ls:
            tail_ls.append(hmm.getTail())
        return tail_ls


def appendWord(nullState, word):
    """
    Append Word object after the nullstate object
    Connect all head states in Word object to the nullstate
    Create a new nullstate object
    Append the new nullstate after Word object
    Connect all tail states in Word object to the new nullstate
    Return the new nullstate
    
    :param nullState: Nullstate object
    :param word: Word object 
    :return: a new Nullstate object
    """
    # connect all head states to given Nullstate
    head_ls = word.getAllHeads()
    for head in head_ls:
        head.edges.append((nullState, 1.0))
        nullState.next.append(head)
    # connect new Nullstate to all tail states
    tail_ls = word.getAllTails()
    nextNull = NullState()
    for tail in tail_ls:
        tail.next.append(nextNull)
        nextNull.edges.append((tail, 0.5))
    return nextNull


def parseBPT(bpt):
    """
    Parse back pointer table obtained from DTW
    Return the recognition result and path
    """
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
    digit_seq = [currDigit[:-1]]
    for i in range(len(seq)):
        if seq[i] == currDigit:
            continue
        else:
            digit_seq.append(seq[i][:-1])
            currDigit = seq[i]

    return digit_seq, seq


def flatten(headNull):
    """Flatten a tree into list using BFS, assign each node's id
    
    :param headNull: starting node/state of the tree to be flattened
    :return: a list of state object
    """
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
