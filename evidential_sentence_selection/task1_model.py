from torch import nn
from transformers import BertModel


class Bert_Model(nn.Module):
    def __init__(self):
        super(Bert_Model, self).__init__()
        """
           if run on Chinese corpus, just replace the "bert-base-uncased" with "bert-base-chinese"
        """
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(768, 2)  # for classification

    def forward(self, x1, x2, x3):
        # x1, x2: [batch, seq_len]  x3: [batch]
        bert_out1 = self.bert(x1)  # [batch_size, seq_len, embedding_size]
        bert_out2 = self.bert(x2)  # [batch_size, seq_len, embedding_size]

        bert_out1 = bert_out1.last_hidden_state  # use the hidden_state in last layer
        bert_out2 = bert_out2.last_hidden_state  # use the hidden_state in last layer

        # exchange the dimension，[batch_size, seq_len, embedding_size]-->[seq_len, batch_size, embedding_size]
        bert_out1 = bert_out1.permute(1, 0, 2)
        bert_out2 = bert_out2.permute(1, 0, 2)
        bert_out1 = bert_out1[0] 
        bert_out2 = bert_out2[0]  

        # gate using similarity
        similarity = x3.reshape(bert_out1.shape[0], 1)
        gate_bert_out = bert_out1 * similarity + bert_out2 * (1 - similarity)

        bert_out = self.fc(gate_bert_out)  # [batch_size, 768]-->[batch_size, 2]
        return bert_out
