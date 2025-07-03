import torch
import torch.nn as nn
import numpy as np
import pdb, copy
import itertools as it

class pLoss(nn.Module):
    def __init__(self, hexG):
        super(pLoss, self).__init__()
        self.legal_state = hexG[0]
        self.label_state_idx = hexG[1]

    def forward(self, f, y, mask, auto_mode=False):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        # pMargin: [B, n], y: [B,n]
        loss = []
        for j in range(y.size(1)):
            pred_i = pMargin[:, j]
            target_i = y[:, j]
            mask_i = mask[:, j]
            loss_i = nn.BCELoss(reduction='none')(pred_i, target_i) * mask_i # instantiation via cross-entropy loss function
            loss.append(loss_i.mean())
        if auto_mode:
            return loss, pMargin
        else:
            return sum(loss), pMargin

    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin


# define Graph and rules
def graph_SO_FFSC():
    Eh_edge = [
        (0,1), (0,2), (0,3), (0,4), (0,5),
        (1, 6), (1, 11),
        (2,6), (2,8), (2,9), (2, 10), (2, 11),
        (3,6), (3,7), (3,8), (3, 9), (3, 11),
        (4,6), (4,7), (4,8), (4,9), (4, 10), (4, 11),
        (5,6), (5,7), (5,9), (5,9), (5, 10), (5, 11),
    ]
    Ee_edge = []
    # prepare a dict/list that record the parent nodes for each node
    parent_dict = {
        "6": [1, 2, 3, 4, 5],
        "7": [3, 4, 5],
        "8": [2, 3, 4, 5],
        "9": [2, 3, 4, 5],
        "10": [2, 4, 5],
        "11": [1, 2, 3, 4, 5]
    }
    children_dict = {
        "1": [6, 11],
        "2": [6, 8, 9, 10, 11],
        "3": [6,7,8, 9, 11],
        "4": [6,7,8,9, 10, 11],
        "5": [6,7,8,9, 10, 11],
    }

    state = legal_state_ffsc(Ee_edge, Eh_edge, bin_list(12), parent_dict,
                        children_dict)  # all the legal states under the proposed graph

    label_state = get_label_state_idx_ffsc(state)  # correspond to the leaf node w.r.t the graph state

    return state, label_state


#
def _check_abnornal(z_):
    if np.inf in z_:
        pdb.set_trace()
        idx = z_ == np.inf
        id_num = [i for i, v in enumerate(list(idx)) if v == True]
    else:
        id_num = [-1]
    return id_num


def bin_list(nsize):
    s = list(it.product(range(2), repeat=nsize))
    out = torch.tensor(s)
    return out

def legal_state_ffsc(Ee, Eh, all_state, parent_dict, children_dict):
    state = copy.deepcopy(all_state)
    state = state.numpy()
    for edge in Ee:
        num = 1
        idx = []
        for i in range(len(state)):
            if state[i, edge[0]] == 1 and state[i, edge[1]] == 1:
               idx.append(i)
               num += 1
        state = np.delete(state, idx, axis=0)

    for edge in Eh:
        num = 1
        idx = []
        for i in range(len(state)):
            # leaf node and its state
            leaf_node_label = edge[1]
            leaf_state = state[i, edge[1]]
            if leaf_node_label <= 5:
                if state[i, edge[0]] == 0 and state[i, edge[1]] == 1:
                    idx.append(i)
                    num += 1

                #
                else:
                    child_nodes = children_dict[str(leaf_node_label)]
                    child_nodes_state = []
                    for k in range(len(child_nodes)):
                        child_nodes_state.append(state[i, child_nodes[k]])
                    if sum(child_nodes_state) == 0 and leaf_state == 1:
                        idx.append(i)
                        num += 1

            else:
                parent_nodes = parent_dict[str(leaf_node_label)]
                parent_nodes_state = []
                for j in range(len(parent_nodes)):
                    parent_nodes_state.append(state[i, parent_nodes[j]])
                if sum(parent_nodes_state) == 0 and leaf_state == 1:
                    idx.append(i)
                    num += 1
        state = np.delete(state, idx, axis=0)
    return torch.tensor(state)


label_map = {
    '110000100001': [1,1,0,0,0,0,1,0,0,0,0,1], # age
    '101000001100': [1,0,1,0,0,0,0,0,1,1,0,0], # expr-smile
    '101000101111': [1,0,1,0,0,0,1,0,1,1,1,1], # expr-surprised
    '120110111101': [1,2,0,1,1,0,1,1,1,1,0,1], # gender
    '120010111111': [1,2,0,0,1,0,1,1,1,1,1,1], # ID
    '100001111111': [1,0,0,0,0,1,1,1,1,1,1,1], # pose
    '000000000000': [0,0,0,0,0,0,0,0,0,0,0,0], # real
}
def get_label_state_idx_ffsc(state):
    length = state.shape[0]
    labels_id = {'110000100001': [], '101000001100': [], '101000101111': [], '120110111101': [],
                 '120010111111': [], '100001111111': [], '000000000000': []}
    for i in range(length):
        if (torch.tensor(label_map['110000100001']) == state[i]).all():
            labels_id['110000100001'] = i

        elif (torch.tensor(label_map['101000001100']) == state[i]).all():
            labels_id['101000001100'] = i

        elif (torch.tensor(label_map['101000101111']) == state[i]).all():
            labels_id['101000101111'] = i

        elif (torch.tensor(label_map['120110111101']) == state[i]).all():
            labels_id['120110111101'] = i

        elif (torch.tensor(label_map['120010111111']) == state[i]).all():
            labels_id['120010111111'] = i

        elif (torch.tensor(label_map['100001111111']) == state[i]).all():
            labels_id['100001111111'] = i

        elif (torch.tensor(label_map['000000000000']) == state[i]).all():
            labels_id['000000000000'] = i
    return labels_id
