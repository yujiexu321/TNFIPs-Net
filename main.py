import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import pandas as pd
from hand_crafted_feature import HF_encoding


def position_encoding(seqs):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of sequence.
    """
    d = 128
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return np.array(res)


def data_construct(seqs, labels, train):
    # Amino acid dictionary
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}

    longest_num = len(max(seqs, key=len))
    sequences = [i.ljust(longest_num, 'X') for i in seqs]
    pos_embed = position_encoding(sequences)
    HF_feature = HF_encoding(seqs, sequences)

    pep_codes = []
    for pep in seqs:
        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep))

    embed_data = rnn_utils.pad_sequence(pep_codes, batch_first=True)  # Fill the sequence to the same length

    dataset = Data.TensorDataset(embed_data, torch.FloatTensor(pos_embed),
                                 torch.FloatTensor(HF_feature), torch.LongTensor(labels))
    batch_size = 64
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_iter


def load_bench_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, train=True)

    train_iter = [x for i, x in enumerate(data_iter) if i % 5 != 0]
    test_iter = [x for i, x in enumerate(data_iter) if i % 5 == 0]

    return train_iter, test_iter


def load_ind_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, train=False)
    return data_iter

class TNFIPs_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 64
        self.batch_size = 64
        self.emb_dim = 128

        self.embedding_seq = nn.Embedding(24, self.emb_dim, padding_idx=0)
        self.encoder_layer_seq = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8)
        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer_seq, num_layers=1)

        self.conv_seq = nn.Sequential(
            nn.Conv1d(20, 20, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        )
        self.conv_HF = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        )
        self.gru_seq = nn.GRU(502, self.hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)

        self.attention = nn.MultiheadAttention(embed_dim=84, num_heads=4)

        self.block1 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(1680, 256),
                                    nn.BatchNorm1d(256),
                                    nn.Dropout(0.6),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 128),
                                    )

        self.block2 = nn.Sequential(nn.Linear(128, 64),
                                    nn.BatchNorm1d(64),
                                    nn.Dropout(0.6),
                                    nn.ReLU(),
                                    nn.Linear(64, 2),
                                    nn.Softmax(dim=1))

    def forward(self, x, pos_embed, HF):
        output1 = self.embedding_seq(x) + pos_embed
        # print(output1.size())
        output1 = self.transformer_encoder_seq(output1)  # .permute(1, 0, 2)
        # print(output1.size())
        output1 = self.conv_seq(output1)
        # print(output1.size())

        output2 = torch.unsqueeze(HF, dim=1)
        output2, hn = self.gru_seq(output2)
        #print(output2.size())
        output2 = self.conv_HF(output2)

        output = torch.cat([output1, output2], 2)
        #print(output.size())
        output = output.permute(1, 0, 2)
        output,_ = self.attention(output,output,output)
        output = output.permute(1, 0, 2)

        output = self.block1(output)
        out = self.block2(output)

        return out,output


def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    rep_list = []
    for x, pos, hf, y in data_iter:
        outputs,rep = net(x, pos, hf)
        pred_prob_positive = outputs[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y.tolist()
        rep_list.extend(rep.detach().numpy())
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data,rep_list,label_real


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    l2_lambda = 0.0
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.norm(param, p=2)

    total_loss = criterion(output, label) + l2_lambda * regularization_loss
    return total_loss


def train_test(train_iter, test_iter):
    net = TNFIPs_Net()
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_acc = 0
    EPOCH = 150
    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        net.train()
        for seq, pos, hf, label in train_iter:
            output,_ = net(seq, pos, hf)
            loss = reg_loss(net, output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data,_,_ = evaluate(train_iter, net)
            test_performance, test_roc_data, test_prc_data,rep_list,label_real = evaluate(test_iter, net)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Test Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[2], test_performance[1], test_performance[3],
            test_performance[4]) + '\n' + '=' * 60
        print(results)
        test_acc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        if test_acc > best_acc:
            best_acc = test_acc
            best_performance = test_performance
            filename = '{}, {}[{:.4f}].pt'.format('H_A_Model' + ', epoch[{}]'.format(epoch + 1), 'ACC', best_acc)
            save_path_pt = os.path.join('file', filename)
            #torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)

            best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[2], best_performance[1], best_performance[3],
                best_performance[4]) + '\n' + '=' * 60
            # print(best_results)
            best_ROC = test_roc_data
            best_PRC = test_prc_data

    return best_performance, best_results, best_ROC, best_PRC

def K_CV(file, k):

    tmp = pd.read_csv(file, header=None)
    seqs, labels = np.array(tmp[0].values.tolist()), np.array(tmp[1].values.tolist())
    data_iter = data_construct(seqs, labels, train=True)
    data_iter = list(data_iter)
    CV_perform = []
    for iter_k in range(k):
        print("\n" + "=" * 16 + "k = " + str(iter_k + 1) + "=" * 16)
        train_iter = [x for i, x in enumerate(data_iter) if i % k != iter_k]
        test_iter = [x for i, x in enumerate(data_iter) if i % k == iter_k]
        performance, _, ROC, PRC = train_test(train_iter, test_iter)
        print(performance)
        CV_perform.append(performance)

    print('\n' + '=' * 16 + colored(' Cross-Validation Performance ',
                                    'red') + '=' * 16 + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n')
    for out in np.array(CV_perform):
        print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(out[0], out[2], out[1], out[3], out[4]))
    mean_out = np.array(CV_perform).mean(axis=0)
    print('\n' + '=' * 16 + "Mean out" + '=' * 16)
    print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(mean_out[0], mean_out[2], mean_out[1], mean_out[3],
                                                              mean_out[4]))
    print('\n' + '=' * 60)

def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model


if __name__ == '__main__':
    # train_test on benchmark dataset
    train_iter, test_iter = load_bench_data("example_input.csv")
    _, result_bench,roc_data,prc_data = train_test(train_iter, test_iter)

    print(result_bench)

    # k-fold cross-validation
#     K_CV("TNF_data/Human/H_M_bench.csv", 5)
