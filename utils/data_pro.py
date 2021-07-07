import os
import json
import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('/home/lishuan/pretrain_model/chinese-roberta-wwm-ext/')


def weibo_read(path):
    text_data = []
    with open(path, mode="r", encoding="utf-8") as f:
        data_dict = json.load(f)
        f.close()
    for text_id, content in data_dict.items():
        text_data.append([text_id, content])
    return text_data


def insuranceRead(path):
    data = []
    with open(path, encoding='utf-8', mode='r') as f:
        for line in f:
            data.append(eval(line))
        f.close()
    return data


def insuranceData_select(data):
    for sample in tqdm(data):
        if len(sample['question']) <= 5:
            data.remove(sample)
    return data


def convert_text_to_ids(data):
    data_ids = []
    for sample in tqdm(data):
        new_sample = {'qid': sample['qid'], 'question': [], 'answers': [], 'label': sample['label']}
        ques_text = sample['question']
        answers = sample['answers']
        label = sample['label']
        assert len(answers) == len(label)
        tokens = tokenizer.tokenize(ques_text)
        input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens)
        seg_mask = [0] * len(input_ids)
        new_sample['question'].append(input_ids)
        new_sample['question'].append(seg_mask)
        for answer in answers:
            tokens = tokenizer.tokenize(answer)
            input_ids = tokenizer.convert_tokens_to_ids(tokens + ["[SEP]"])
            seg_mask = [1] * len(input_ids)
            new_sample['answers'].append([input_ids, seg_mask])
        data_ids.append(new_sample)
    return data_ids


def insurancepro(path):
    data = insuranceRead(path)
    data = insuranceData_select(data)
    data_ids = convert_text_to_ids(data)
    return data_ids


def insurance_shuffle(train_data):
    random.shuffle(train_data)
    new_train_data = []
    for sample in train_data:
        new_sample = []
        ques_ids = sample['question'][0]
        seg_mask = sample['question'][1]

        answers = sample['answers']
        label = sample['label']
        assert len(answers) == len(label)
        if len(answers) == 11:
            arr = np.arange(11)
            np.random.shuffle(arr)

            answers = np.array(answers)[arr].tolist()
            label = np.array(label)[arr]

            for answer in answers:
                input_ids = ques_ids + answer[0]
                segement_ids = seg_mask + answer[1]
                attention_mask = [1] * len(input_ids)
                new_sample.append([input_ids, segement_ids, attention_mask])
            new_sample.append(label.argmax())
        new_train_data.append(new_sample)
    return new_train_data


def Batch(data, args):
    text_input = []
    text_type = []
    text_mask = []
    for i in range(11):
        text_input.append(data[i][0])
        text_type.append(data[i][1])
        text_mask.append(data[i][2])
    label = [data[-1]]

    text_input_ = batch_pad(text_input, args, pad=0)
    text_type_ = batch_pad(text_type, args, pad=1)
    text_mask_ = batch_pad(text_mask, args, pad=0)
    return text_input_, text_type_, text_mask_, torch.tensor(label, dtype=torch.long).cuda()


def batch_pad(batch_data, args, pad=0):
    seq_len = [len(i) for i in batch_data]
    max_len = max(seq_len)
    if max_len > args.max_len:
        max_len = args.max_len
    out = []
    for line in batch_data:
        if len(line) < max_len:
            out.append(line + [pad] * (max_len - len(line)))
        else:
            out.append(line[:args.max_len])
    return torch.tensor(out, dtype=torch.long).cuda()


def insurance_text_len_info(data):
    ques_info = []
    answers_info = []
    j = 0
    for sample in data:
        ques_txt = sample['question']
        answers = sample['answers']
        ques_info.append(len(ques_txt))
        # if len(ques_txt) <= 5:
        #     j += 1
        #     print(sample)

        for answer in answers:
            answers_info.append(len(answer))
    print(j)
    print(len(ques_info))
    print(max(ques_info), min(ques_info))
    print('-----------------------------')
    print(max(answers_info), min(answers_info))
    return ques_info, answers_info



def finqa_read(path):
    data = pd.read_csv(path).values.tolist()
    return data


def ccks_read(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            new_sample = []
            sample = line.strip()
            label = sample[-1]
            texts = sample[:-1].strip()
            if label in ['0', '1']:
                new_sample.append(int(label))

            if len(texts.split()) == 2:
                new_sample.extend(texts.split())

            if len(new_sample) == 3:
                data.append(new_sample)
    return data


# class convertText2ids(object):
#     def __init__(self, bert_path):
#         self.Tokenizer = BertTokenizer.from_pretrained(bert_path)
#
#     def genIds(self, text1, out_attMask=False, out_seg=False, out_token=False, text2=None):
#         if text2 == None:
#             tokens = self.Tokenizer.tokenize(text1)
#             tokens = ["[CLS]"] + tokens + ["[SEP]"]
#             segment = [0]*len(tokens)
#         else:
#             tokens1 = self.Tokenizer.tokenize(text1)
#             tokens2 = self.Tokenizer.tokenize(text1)
#             tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
#             segment = [0]*(len(tokens1)+2) + [1]*(len(tokens2)+1)
#         input_ids = self.Tokenizer.convert_tokens_to_ids(tokens)
#         atten_mask = [1] * len(input_ids)
#         out = (input_ids, )
#
#         if out_attMask:
#             out = out + (atten_mask, )
#         if out_seg:
#             out = out + (segment, )
#         if out_token:
#             out = out + (tokens, )
#         return out




if __name__ == '__main__':
    # weibp_path = '../WebQA.v1.0/me_test.ann.json'
    # data = weibo_read(weibp_path)

    # finqa_path = '../fin_qa/financezhidao_filter.csv'
    # finqa_data = finqa_read(finqa_path)

    # ccks_path = '../ccks_2018/task3_train.txt'
    # ccks_data = ccks_read(ccks_path)
    #
    # Tokener = convertText2ids('/home/lishuan/pretrain_model/bert_wwm/')
    # Tokener.genIds(ccks_data[0][1])

    insurance_path = '../insuranceqa/'
    insurance_trainpath = os.path.join(insurance_path, 'train.txt')
    # insurance_train_data = insuranceRead(insurance_trainpath)
    # a, b = insurance_text_len_info(insurance_train_data)
    data_ids = insurancepro(insurance_trainpath)
