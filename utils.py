import pickle
import queue


class HMM:
    def __init__(self, state_ls, trans_mat, name):
        self.state_ls = state_ls
        self.trans_mat = trans_mat
        self.name = name


def save_hmm(hmm):
    filename = "./model/" + hmm.name + ".pickle"
    with open(filename, "wb") as file:
        pickle.dump(hmm, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_hmm(digit):
    filename = "./model/" + digit + ".pickle"
    with open(filename, "rb") as file:
        return pickle.load(file)


def dfs_print(head):
    # depth first print tree
    if head == None:
        return
    head.visited = True
    print(head)
    for child in head.next:
        if not child.visited:
            dfs_print(child)


def countNode(head):
    if head == None:
        return 0
    else:
        count = 0
        for child in head.next:
            if child.visited:
                continue
            child.visited = True
            count += countNode(child)
        return 1 + count
