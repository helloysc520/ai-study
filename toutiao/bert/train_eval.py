from datetime import time

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import time

from sklearn import metrics
from torch.optim import AdamW

from utils import get_time_dif

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def loss_fn(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)


def train(config,model,train_iter,dev_iter):

    start_time = time.time()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_batch = 0  #记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False

    model.train()

    for epoch in range(config.num_epochs):
        total_batch = 0
        print('Epoch {}/{}'.format(epoch + 1, config.num_epochs))

        for i,(trains, labels) in enumerate(train_iter):
            outputs = model(trains)

            model.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 200 == 0 and total_batch != 0:
                #每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc,dev_loss = evaluate(config,model,dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch

                else:
                    improve = ''

                time_diff = time.time() - start_time
                msg = 'Iter: {0:> 6},Train Loss: {1:>5.2},Train Acc: {2:>6.2%},Val Loss: {3:>5.2},Val Acc: {4:>6.2%},Time:{5} {6}'

                print(msg.format(total_batch,loss.item(),train_acc,dev_loss,dev_acc,time_diff,improve))

                #评估完成后将模型置于训练模式，更新参数
                model.train()

                total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                #验证集loss超过100 batch没下降，结束训练
                print('No optimization for a long time,auto-stopping...')
                flag = True
                break

        if flag:
            break


def test(config,model,test_iter):
    #采用量化模型进行推理时需要关闭
    model.eval()
    start_time = time.time()
    test_acc,test_loss,test_report,test_confusion = evaluate(config,model,test_iter,test=True)

    msg = 'Test Loss:{0:>5.2}, Test Acc: {1:>6.2%}'
    print(msg.format(test_loss,test_acc))

    print('Precision,Recall and F1-score...')
    print(test_report)
    print('Confusion matrix...')
    print(test_confusion)
    time_diff = get_time_dif(start_time)
    print('Time usage:',time_diff)


def evaluate(config,model,data_iter,test=False):

    model.eval()
    loss_total = 0
    predict_all = np.array([],dtype=int)
    labels_all = np.array([],dtype=int)

    with torch.no_grad():
        for tests, labels in data_iter:
            outputs = model(tests)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all,labels)
            predict_all = np.append(predict_all,predict)

    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        report = metrics.classification_report(labels_all, predict_all,target_names=config.class_list,digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc,loss_total / len(data_iter),report,confusion

    return acc,loss_total / len(data_iter)