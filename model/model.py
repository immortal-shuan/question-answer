import torch
import torch.nn as nn
from transformers import BertModel


class bert_qa(nn.Module):
    def __init__(self, args):
        super(bert_qa, self).__init__()

        self.bert_model = BertModel.from_pretrained('/home/lishuan/pretrain_model/chinese-roberta-wwm-ext/')
        for param in self.bert_model.parameters():
            param.requires_grad = True

        self.criterion = torch.nn.CrossEntropyLoss()

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(args.bert_dim*2, 1)

    def forward(self, input_id, token_type_ids, attention_mask, label=None):

        out = self.bert_model(
            input_ids=input_id, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        word_vec = out.last_hidden_state
        max_feature = self.max_pool(word_vec.permute(0, 2, 1)).squeeze(-1)
        avg_feature = self.avg_pool(word_vec.permute(0, 2, 1)).squeeze(-1)
        word_feature = torch.cat((max_feature, avg_feature), dim=-1)
        word_feature = self.dropout(word_feature)
        out = self.fc(word_feature).view(-1, 11)

        if label != None:
            loss = self.criterion(out, label)
            return (out, loss)
        else:
            return out