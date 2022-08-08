from torch import nn
from transformers import BertModel
import math
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Bert_Model(nn.Module):
    def __init__(self):
        super(Bert_Model, self).__init__()
        # ================BERT================
        """
        if run on Chinese corpus, just replace the "bert-base-uncased" with "bert-base-chinese"
        """
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = True
        # ================self-attention================
        self.pos_encoder = PositionalEncoding(768, 0.1)
        self.Wq = nn.Parameter(torch.Tensor(768, 768))
        self.Wk = nn.Parameter(torch.Tensor(768, 768))
        self.Wv = nn.Parameter(torch.Tensor(768, 768))

        # init parameter
        nn.init.uniform_(self.Wq, -0.1, 0.1)
        nn.init.uniform_(self.Wk, -0.1, 0.1)
        nn.init.uniform_(self.Wv, -0.1, 0.1)

        # ================additive attention============
        # attention parameters
        self.W = nn.Parameter(torch.Tensor(768, 768))
        self.U = nn.Parameter(torch.Tensor(768, 768))
        self.v = nn.Parameter(torch.Tensor(768, 1))

        # initialization of parameters
        nn.init.uniform_(self.W, -0.1, 0.1)
        nn.init.uniform_(self.U, -0.1, 0.1)
        nn.init.uniform_(self.v, -0.1, 0.1)

        # ===============linear linear================
        self.fc1 = nn.Linear(768 * 3, 768)
        self.activate1 = nn.LeakyReLU()

        # softmax
        self.fc = nn.Linear(768 * 3, 5)

    def self_attention(self, x, prior):
        """
        self-attention
        :param x: [batch, seq_len, embedding_dim]
        :param prior: [batch, seq_len, seq_len]
        :return: [batch, seq_len, embedding_dim]
        """
        Q = torch.matmul(self.Wq, x.permute(0, 2, 1))  # [batch, q_dim, seq_len]
        K = torch.matmul(self.Wk, x.permute(0, 2, 1))  # [batch, k_dim, seq_len]
        v = torch.matmul(self.Wv, x.permute(0, 2, 1))  # [batch, v_dim, seq_len]
        # [batch, seq_len, seq_len]
        att_score = torch.softmax(torch.matmul(K.permute(0, 2, 1), Q) / math.sqrt(Q.shape[1]), dim=2)
        prior = torch.softmax(prior, dim=2)
        att_score_add_prior = att_score + prior
        att_score_new = torch.softmax(att_score_add_prior, dim=2)

        generate_seq = torch.matmul(att_score_new, v.permute(0, 2, 1))  # [batch_size, seq_len, embedding_dim]

        return generate_seq  # [batch_size, seq_len, embedding_dim]

    def additive_attention(self, x, q):
        """
        additive attention
        :param x: [batch, embedding_dim, seq_len]
        :param q: [batch, embedding_dim, 1]
        :return: [batch, embedding_dim]
        """
        # [batch, embedding_dim, seq_len]
        att_score_sub = torch.tanh(torch.matmul(self.W, x) + torch.matmul(self.U, q).repeat(1, 1, x.shape[2]))
        # [batch, 1, seq_len]
        att_score = torch.softmax(torch.matmul(self.v.permute(1, 0), att_score_sub), dim=-1)
        # [batch, 1, embedding_dim]-->[batch, embedding_dim]
        result = torch.matmul(att_score, x.permute(0, 2, 1)).squeeze(1)
        return result

    def forward(self, x1, x2, x3, x4, x5):
        """
        :param x1: event: [batch, seq_len]
        :param x2: event evidences: [batch, m, seq_len]
        :param x3: evidences concat: [batch, seq_len]
        :param x4: global: [batch, seq_len]
        :param x5: similarity: [batch, m, m]
        :return:
        """
        x1, x2, x3, x4, x5 = x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0), x5.unsqueeze(0)
        # ===================================word embedding for each token=============================================
        # [batch_size, seq_len, embedding_size]
        bert_event = self.bert(x1).last_hidden_state

        # [batch_size*m, seq_len, embedding_size]
        bert_evidences = self.bert(x2.reshape(x2.shape[0] * x2.shape[1], -1)).last_hidden_state
        # [batch_size, m, seq_len, embedding_size]
        bert_evidences = bert_evidences.reshape(x2.shape[0], x2.shape[1], x2.shape[2], -1)

        # [batch_size, seq_len, embedding_size]
        bert_concat_evidences = self.bert(x3).last_hidden_state

        # [batch_size, seq_len, embedding_size]
        bert_global = self.bert(x4).last_hidden_state

        # ===================================each sentence embedding=============================================
        # [batch_size, seq_len, embedding_size]-->[seq_len, batch_size, embedding_size]-->[batch_size, embedding_size]
        bert_event_cls = bert_event.permute(1, 0, 2)[0]  # [batch_size, embedding_size]
        bert_evidences_cls = bert_evidences.permute(2, 0, 1, 3)[0]  # [batch_size, m, embedding_size]
        bert_concat_evidences_cls = bert_concat_evidences.permute(1, 0, 2)[0]  # [batch_size, embedding_size]
        bert_global_cls = bert_global.permute(1, 0, 2)[0]  # [batch_size, embedding_size]

        # ===================================sentence-level interactions=============================================
        # [batch_size, m, embedding_size]
        bert_evidences_cls_sen = self.self_attention(
            self.pos_encoder(bert_evidences_cls.permute(1, 0, 2)).permute(1, 0, 2), prior=x5)

        # ===================================token-level interactions=============================================
        # [batch_size, m, seq_len, embedding_size]
        token_evidence = torch.mul(bert_evidences, bert_concat_evidences.unsqueeze(1))
        # [batch_size, m, seq_len, embedding_size]
        token_event = torch.mul(bert_evidences, bert_event.unsqueeze(1))

        # element-wise：[batch_size, m, seq_len, embedding_size]
        element_wise_add = token_evidence + token_event + bert_evidences

        # [batch_size, m, embedding_size]
        max_pooling_res = torch.max(element_wise_add, dim=2)[0]

        # final representation
        concat_sen_pre = torch.cat((bert_evidences_cls, max_pooling_res, bert_evidences_cls_sen), dim=-1)
        # transform dimensions
        sentence_final_pre = self.fc1(concat_sen_pre.reshape(-1, 768 * 3)).reshape(x2.shape[0], x2.shape[1], -1)
        # [batch_size, m, embedding_size]
        sentence_final_pre = self.activate1(sentence_final_pre)

        # ===================================additive attention=============================================
        # [batch_size, embedding_size]
        attention_res = self.additive_attention(x=sentence_final_pre.permute(0, 2, 1), q=bert_event_cls.unsqueeze(2))
        concated = torch.cat((attention_res, bert_global_cls, bert_event_cls), dim=-1)
        out = self.fc(concated)
        return out

# model = Bert_Model()
# x1 = torch.rand(5).long()
# x2 = torch.rand(3, 5).long()
# x5 = torch.rand(3, 3)
# model(x1, x2, x1, x1, x5)
