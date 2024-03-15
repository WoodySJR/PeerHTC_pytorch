import torch, pandas as pd, numpy as np, math
from torch import nn
from torch.nn.functional import binary_cross_entropy as BCE
from torch.optim import Adam, SGD, AdamW
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

num_1, num_2, num_3, num_labels = 7,46,77,130

# training function (including early stopping)
def train(model, train_dataloader, val_dataloader, num_epochs, lr, budget, save_path, thr):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0000001)
    trace = []
    microf1_max = -math.inf
    count_e = 0
    epoch_best = 0
    for e in range(num_epochs):
        loss_value = 0
        acc_value = 0
        count = 0
        with tqdm(total=len(train_dataloader), leave=False) as t: 
            for inputs, labels in train_dataloader:
                # weight = labels[:,130:] # second round
                # labels = labels[:,0:130] # second round
                t.set_description("Epoch "+str(e))
                optimizer.zero_grad()
                prob = model(inputs)
                pred = (prob>thr).detach()
                acc = torch.sum(torch.all(pred==labels, dim=1))/inputs.shape[0]
                acc_value += acc
                losses = BCE(prob, labels.float(), reduction="none")
                # losses = losses*weight # second round
                loss = torch.sum(losses)/losses.shape[0]
                loss_value += loss.detach().item()
                count += 1
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss_value/count, acc=acc_value/count)
                t.update(1)
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_dataloader):
                    prob = model(inputs)
                    pred = (prob>thr).float()
                    if i==0:
                        preds = pred
                        targets = labels
                    else:
                        preds = torch.cat((preds,pred), dim=0)
                        targets = torch.cat((targets,labels), dim=0)
                micro_f1 = f1_score(targets.cpu(), preds.cpu(), average="micro")
                macro_f1 = f1_score(targets.cpu(), preds.cpu(), average="macro")
                micro_f11 = f1_score(targets[:,0:num_1].cpu(), preds[:,0:num_1].cpu(), average="micro")
                macro_f11 = f1_score(targets[:,0:num_1].cpu(), preds[:,0:num_1].cpu(), average="macro")
                micro_f12 = f1_score(targets[:,num_1:(num_1+num_2)].cpu(), preds[:,num_1:(num_1+num_2)].cpu(), average="micro")
                macro_f12 = f1_score(targets[:,num_1:(num_1+num_2)].cpu(), preds[:,num_1:(num_1+num_2)].cpu(), average="macro")
                micro_f13 = f1_score(targets[:,(num_1+num_2):num_labels].cpu(), preds[:,(num_1+num_2):num_labels].cpu(), average="micro")
                macro_f13 = f1_score(targets[:,(num_1+num_2):num_labels].cpu(), preds[:,(num_1+num_2):num_labels].cpu(), average="macro")
                acc = torch.sum(torch.all(preds==targets, dim=1))/targets.shape[0]
                acc1 = torch.sum(torch.all(preds[:,0:num_1]==targets[:,0:num_1], dim=1))/targets.shape[0]
                acc2 = torch.sum(torch.all(preds[:,num_1:(num_1+num_2)]==targets[:,num_1:(num_1+num_2)], dim=1))/targets.shape[0]
                acc3 = torch.sum(torch.all(preds[:,(num_1+num_2):num_labels]==targets[:,(num_1+num_2):num_labels], dim=1))/targets.shape[0]
                print("Epoch %d overall: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(e,micro_f1,macro_f1,acc))
                print("Epoch %d level 1: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(e,micro_f11,macro_f11,acc1))
                print("Epoch %d level 2: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(e,micro_f12,macro_f12,acc2))
                print("Epoch %d level 3: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(e,micro_f13,macro_f13,acc3))
                
                trace.append(acc)
                if micro_f1>microf1_max:
                    microf1_max = micro_f1
                    torch.save(model, save_path+"/model.pt")
                    print("Saving model parameters. ")
                    count_e = 0
                    epoch_best = e+1
                else:
                    count_e += 1
                if count_e>budget: # no improvement for more than some epochs:
                    print("Early stopped after "+str(e)+" epochs. Best model saved at epoch "+str(epoch_best)+". ")
                    break