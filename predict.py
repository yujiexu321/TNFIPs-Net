from main import load_ind_data, TNFIPs_Net, load_model, evaluate
import torch
from termcolor import colored

def predict(file):
    data_iter = load_ind_data(file)
    model = TNFIPs_Net()
    path_pretrain_model = "model/H_M_model.pt"
    model = load_model(model, path_pretrain_model)
    model.eval()
    with torch.no_grad():
        ind_performance, ind_roc_data, ind_prc_data, _, _ = evaluate(data_iter, model)
    ind_results = '\n' + '=' * 16 + colored(' Independent Test Performance', 'red') + '=' * 16 \
                   + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            ind_performance[0], ind_performance[2], ind_performance[1], ind_performance[3],
            ind_performance[4]) + '\n' + '=' * 60

    return ind_results


ind_result = predict('example_input.csv')
print(ind_result)