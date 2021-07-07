import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange, tqdm
from model.model import bert_qa
from utils.data_pro import insurancepro, insurance_shuffle, Batch
from sklearn.metrics import accuracy_score, auc

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--max_len', default=512)              # 句子的最大长度

    # 训练时参数
    arg_parser.add_argument('--stop_num', default=5)         # 当评价指标在stop_num内不在增长时，训练停止
    arg_parser.add_argument('--seed', default=102)           # 随机种子
    arg_parser.add_argument('--epoch_num', default=20)          # 数据训练多少轮
    arg_parser.add_argument('--save_model', default=False)   # 是否保存模型
    arg_parser.add_argument('--loss_step', default=1)        # 模型几次loss后更新参数

    # 各种文档路径
    arg_parser.add_argument('--data_path', default='../insuranceqa')     # 训练集的文件路径
    arg_parser.add_argument('--output_path', default='model_output/xiandaiwen_lijie')                # 模型输出路径

    # 模型内各种参数
    arg_parser.add_argument('--bert_lr', default=2e-5)          # bert层学习率
    arg_parser.add_argument('--bert_dim', default=768)         # bert的输出向量维度

    args = arg_parser.parse_args()
    return args


def train(model, optimizer, args):
    model.zero_grad()

    dev_acc = 0.0
    max_acc_index = 0

    test_acc = 0.0

    for i in range(args.epoch_num):
        train_path = os.path.join(args.data_path, 'train.txt')
        valid_path = os.path.join(args.data_path, 'valid.txt')
        test_path = os.path.join(args.data_path, 'test.txt')

        train_data = insurancepro(train_path)
        valid_data = insurancepro(valid_path)
        test_data = insurancepro(test_path)

        train_data = insurance_shuffle(train_data)
        valid_data = insurance_shuffle(valid_data)
        test_data = insurance_shuffle(test_data)

        train_len = len(train_data)

        train_step = 1.0
        train_loss = 0.0

        train_preds = []
        train_labels = []

        for j in trange(train_len):
            model.train()
            train_batch_data = train_data[j]
            text_input, text_type, text_mask, label = Batch(train_batch_data, args)

            out, loss = model(text_input, text_type, text_mask, label)

            train_loss += loss.item()

            loss = loss / args.loss_step
            loss.backward()

            if int(train_step % args.loss_step) == 0:
                optimizer.step()
                model.zero_grad()

            pred = out.argmax(dim=-1).cpu().tolist()
            train_preds.extend(pred)
            train_labels.extend(label.cpu().tolist())
            train_step += 1.0

        train_acc = accuracy_score(np.array(train_preds), np.array(train_labels))

        print('epoch:{}\n train_loss:{}\n train_acc:{}'.format(i, train_loss / train_step, train_acc))

        dev_acc_ = dev(model=model, data=valid_data, args=args)
        test_acc_ = dev(model=model, data=test_data, args=args)

        if dev_acc <= dev_acc_:
            dev_acc = dev_acc_
            max_acc_index = i

            test_acc = test_acc_

            if args.save_model:
                save_file = os.path.join(args.output_path, 'model_{}.pth'.format(i))
                torch.save(model.state_dict(), save_file)

        if i - max_acc_index > args.stop_num:
            break

    file = open('result.txt', 'a')
    file.write('max_acc: {}, dev_acc: {}, test_acc: {}'.format(max_acc_index, dev_acc, test_acc) + '\n')
    file.close()

    print('-----------------------------------------------------------------------------------------------------------')
    print('max_acc: {}, dev_acc: {}, test_acc: {}'.format(max_acc_index, dev_acc, test_acc))
    print('-----------------------------------------------------------------------------------------------------------')


def dev(model, data, args):
    model.eval()

    dev_len = len(data)

    dev_preds = []
    dev_labels = []

    with torch.no_grad():
        for m in range(dev_len):
            train_batch_data = data[m]
            text_input, text_type, text_mask, label = Batch(train_batch_data, args)
            out, loss = model(text_input, text_type, text_mask, label)

            pred = out.argmax(dim=-1).cpu().tolist()
            dev_preds.extend(pred)
            dev_labels.extend(label.cpu().tolist())

    dev_acc = accuracy_score(np.array(dev_preds), np.array(dev_labels))

    print('dev_acc:{}'.format(dev_acc))
    return dev_acc


def main():
    args = init_arg_parser()
    setup_seed(args)

    model = bert_qa(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.bert_lr)
    train(model, optimizer, args)


def p(data, num=5):
    for i in range(num):
        print(data[i])


if __name__ == '__main__':
    main()