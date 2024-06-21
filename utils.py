import torch
import os
import errno
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def calc_hamming(B1, B2):
    num = B1.shape[0]
    q = B1.shape[1]
    result = torch.zeros(num).cuda()
    for i in range(num):
        result[i] = 0.5 * (q - B1[i].dot(B2[i]))
    return result

def return_samples(index, qB, rB, k=None):
    num_query = qB.shape[0]
    if k is None:
        k = rB.shape[0]
    index_matrix = torch.zeros(num_query, k + 1).int()
    index = torch.from_numpy(index)
    for i in range(num_query):
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        index_matrix[i] = torch.cat((index[i].unsqueeze(0), ind[np.linspace(0, rB.shape[0]-1, k).astype('int')]), 0)
    return index_matrix

def return_results(index, qB, rB, s=None, o=None):
    num_query = qB.shape[0]
    index_matrix = torch.zeros(num_query, 1+s+o).int()
    index = torch.from_numpy(index)
    for i in range(num_query):
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        index_matrix[i] = torch.cat((index[i].unsqueeze(0), ind[:s], ind[np.linspace(0, rB.shape[0]-1, o).astype('int')]), 0)
    return index_matrix

def CalcMap(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map

# def fx_calc_map_label(model,val_set, database_set, k = 0):
#   if dist_method == 'L2':
#     dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
#   elif dist_method == 'COS':
#     dist = scipy.spatial.distance.cdist(image, text, 'cosine')
#   ord = dist.argsort()
#   numcases = dist.shape[0]
#   if k == 0:
#     k = numcases
#   res = []
#   for i in range(numcases):
#     order = ord[i]
#     p = 0.0
#     r = 0.0
#     for j in range(k):
#       if label[i] == label[order[j]]:
#         r += 1
#         p += (r / (j + 1))
#     if r > 0:
#       res += [p / r]
#     else:
#       res += [0]
#   return np.mean(res)


def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S

def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt

def image_normalization(_input):
    _input = 2 * _input / 255 - 1
    return _input

def image_restoration(_input):
    _input = (_input + 1) / 2 * 255
    return _input

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def acc_i2t2(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def acc_train(input):
    predicted = input.squeeze().numpy()
    batch_size = predicted.shape[0]
    predicted[predicted > np.log(0.5)] = 1
    predicted[predicted < np.log(0.5)] = 0
    target = np.eye(batch_size)
    recall = np.sum(predicted * target) / np.sum(target)
    precision = np.sum(predicted * target) / np.sum(predicted)
    acc = 1 - np.sum(abs(predicted - target)) / (target.shape[0] * target.shape[1])

    return acc, recall, precision

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in dataloader:
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def get_batch(dataset):
    list=[]
    for i, data in enumerate(dataset):
        list.append(data[0].unsqueeze(0))
    return torch.cat(list).to('cuda')

def calc_similarity(label_1, label_2):
    return (np.dot(label_1, label_2.transpose()) > 0).astype(np.float32)

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    disH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return disH

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def cal_Precision_Recall_Curve(qB, rB, query_L, retrieval_L):
    S = calc_similarity(query_L, retrieval_L)
    dist = calc_hammingDist(qB, rB)
    num = qB.shape[0]  # the number of input instances
    bits = qB.shape[1]
    precision = np.zeros((num, bits + 1))
    recall = np.zeros((num, bits + 1))
    for i in range(num):
        relevant = set(np.where(S[i, :] == 1)[0])
        retrieved = set()
        for bit in range(bits + 1):
            retrieved = set(np.where(dist[i, :] == bit)[0]) | retrieved
            ret_rel = len(retrieved & relevant)
            # print('bit : {0}, Precision: {1:.4f}, Recall: {2:.4f}'.format(bit,
            #      ret_rel / len(retrieved), ret_rel / len(relevant)))
            recall[i, bit] = ret_rel / len(relevant)
            if len(retrieved) == 0:
                continue
            precision[i, bit] = ret_rel / len(retrieved)

    return recall.mean(axis=0), precision.mean(axis=0)

def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall