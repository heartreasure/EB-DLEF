"""
original content
item_0: event id
item_1: doc id
item_2: annotation info
item_3: event
item_4: factuality
item_5: evidential sentences list
item_6: all sentences count
item_7: all sentences list
++++++++++++++++++++++++++++++++++++++++++
add similarity feature
item_10: similarity between all sentences and event
item_11: top 50% similarity sentences, as the global semantics
item_12: negative samples index list after negative sampling

######need add######
item_16: pred evidential list：
item_17: similarity matrix：[[], [], []]，
item_18: global feature: ""
"""
import json
import stanza
from my_model import Bert_Model
import torch.nn.functional as F
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

device = torch.device("cuda:7") if torch.cuda.is_available() else 'cpu'


def get_event_tensor(event_list):
    """
       if run on Chinese corpus, just replace the "bert-base-uncased" with "bert-base-chinese"
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    all_input1, all_input2, all_input3, all_label = [], [], [], []
    for event in event_list:
        inputs1, inputs2, inputs3, labels = [], [], [], []
        event_sentence = event["item_3"]
        global_sentence = event_sentence
        for idx in event["item_11"]:
            global_sentence += event["item_7"][idx]
        for each_index in range(event["item_6"]):
            pos_or_neg_pair = event_sentence + " " + event["item_7"][each_index]
            inputs1.append(pos_or_neg_pair)
            inputs2.append(global_sentence)
            inputs3.append(event["item_10"][each_index])
            if each_index in event["item_5"]:
                labels.append(1)
            else:
                labels.append(0)

        encoded_input1 = tokenizer(inputs1, max_length=512, truncation=True,
                                   padding='max_length', return_tensors="pt")
        encoded_input2 = tokenizer(inputs2, max_length=512, truncation=True,
                                   padding='max_length', return_tensors="pt")
        input1_tensor = encoded_input1['input_ids']
        input2_tensor = encoded_input2['input_ids']
        input3_tensor = torch.tensor(inputs3)

        label_tensor = torch.tensor(labels)
        all_input1.append(input1_tensor)
        all_input2.append(input2_tensor)
        all_input3.append(input3_tensor)

        all_label.append(label_tensor)
    return all_input1, all_input2, all_input3, all_label


def whole_process():
    # read data
    file_path = "../../processed_corpus/cn_event_base_more.txt"
    with open(file_path, encoding="utf-8") as f:
        data_list = eval(f.read())

    n_total = len(data_list)
    offset = int(n_total * 0.8)

    random.seed(126)
    random.shuffle(data_list)

    train_event = data_list[:offset]
    test_event = data_list[offset:]

    # process for training set and testing set
    process_train(train_event)
    process_test(test_event)


def process_train(train_event):
    # 1.add noise for golden sentence(removed)
    train_event = add_event_noise(train_event)
    # 2.add similarity list
    train_event = add_similarity_list(train_event)
    # 3.add global feature
    train_event = add_global_sentence(train_event)
    # save
    with open("task2_inputs/task2_train_cn.txt", "w", encoding='utf-8') as f:
        f.write(str(train_event))


def process_test(test_event):
    # 1.add pred sentence use the model checkpoint
    test_event = add_pred_sentence(test_event)
    # 2.add similarity list
    test_event = add_similarity_list(test_event)
    # 3.add global feature
    test_event = add_global_sentence(test_event)
    # save
    with open("task2_inputs/task2_test_cn.txt", "w", encoding='utf-8') as f:
        f.write(str(test_event))


def add_pred_sentence(test_event):
    model_pkl = "task1_checkpoint/task1_cn.pth"
    model = Bert_Model().to(device)
    model.load_state_dict(torch.load(model_pkl))
    model.eval()
    test_event = test_and_pred(test_event, model, device)
    return test_event


def test_and_pred(event_list, model, device):
    model.eval()
    total_eval_loss = 0
    final_pred = []
    final_label = []
    testing_bar = tqdm(event_list)
    inputs1_list, inputs2_list, inputs3_list, labels_list = get_event_tensor(event_list)
    for i, event in enumerate(testing_bar):
        testing_bar.set_description("  Testing")
        inputs1 = inputs1_list[i].to(device)  # torch.Size([batch, seq_len])
        inputs2 = inputs2_list[i].to(device)  # torch.Size([batch, seq_len])
        inputs3 = inputs3_list[i].to(device)  # torch.Size([batch])
        labels = labels_list[i].to(device)  # torch.Size([batch])
        with torch.no_grad():
            outputs = model(inputs1, inputs2, inputs3)
        total_eval_loss += F.cross_entropy(outputs, labels.flatten())
        each_event_pred = outputs.detach().cpu().numpy()  # [batch, 2]
        each_event_label = labels.cpu().numpy()  # [batch]
        final_pred.append(each_event_pred)
        final_label.append(each_event_label)

    # not for evaluation, but for getting the predicted sentences and getting a new data_list
    event_list = get_pred_sentence(event_list, final_pred)
    return event_list


def get_pred_sentence(event_list, final_pred):
    for i in range(len(final_pred)):
        pred_flat = backup_process(final_pred[i], backup=True)
        # record the index
        index_list = []
        for index in range(len(pred_flat)):
            if pred_flat[index] == 1:
                index_list.append(index)
        event_list[i]["item_16"] = index_list
    return event_list


def backup_process(final_pred, backup):
    if backup:
        pred_flat = np.argmax(final_pred, axis=1).flatten()
        if 1 not in pred_flat:
            event_pred = final_pred
            event_dict = dict(zip(event_pred[:, 1].tolist(), [_ for _ in range(event_pred.shape[0])]))
            sorted_dict = sorted(event_dict.items(), key=lambda kv: (kv[0], kv[1]), reverse=True)
            pred_flat = [0 for _ in range(event_pred.shape[0])]
            for item in sorted_dict[:1]:
                pred_flat[item[1]] = 1
    else:
        pred_flat = np.argmax(final_pred, axis=1).flatten()
    return pred_flat


def add_event_noise(train_event):
    # just copy the golden
    for event in train_event:
        event["item_16"] = event["item_5"]
    return train_event


def add_similarity_list(sub_event):
    # stanza
    nlp = stanza.Pipeline(lang='zh', processors='tokenize')
    # stopwords
    filepath = "../../stopwords/zh.json"
    with open(filepath, 'r', encoding='utf-8') as fp:
        stop_words_list = json.load(fp)

    for event in tqdm(sub_event):
        golden_evidence = event["item_16"]
        if len(golden_evidence) == 1:
            all_sim_list = [[0]]
        else:
            golden_evidence_count = len(golden_evidence)
            all_sim_list = [[0 for _ in range(golden_evidence_count)] for i in range(golden_evidence_count)]
            for i in range(golden_evidence_count - 1):
                for j in range(i + 1, golden_evidence_count):
                    all_sim_list[i][j] = all_sim_list[j][i] = get_similarity(event['item_7'][golden_evidence[i]],
                                                                             event['item_7'][golden_evidence[j]],
                                                                             stop_words_list,
                                                                             nlp)
        event["item_17"] = all_sim_list
    return sub_event


def add_global_sentence(sub_event):
    for event in sub_event:
        golden_evidence = event["item_16"]
        global_index = []
        for i in golden_evidence:
            global_index.append(i)
            if i + 1 < len(event["item_7"]):
                global_index.append(i + 1)
            if i - 1 >= 0:
                global_index.append(i - 1)
        # add title
        global_index.append(0)
        global_index = list(set(global_index))
        event["item_18"] = global_index

    return sub_event


def get_similarity(sentence1, sentence2, stopwords, nlp):
    seg1_doc = nlp(sentence1)
    seg1 = []
    for sen in seg1_doc.sentences:
        for token in sen.tokens:
            if token.text not in stopwords and token.text not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' and token.text not in "，。……——“”‘’！；":
                seg1.append(token.text)

    seg2_doc = nlp(sentence2)
    seg2 = []
    for sen in seg2_doc.sentences:
        for token in sen.tokens:
            if token.text not in stopwords and token.text not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' and token.text not in "，。……——“”‘’！；":
                seg2.append(token.text)

    word_list = list(set([word for word in seg1 + seg2]))
    word_count_vec_1 = []
    word_count_vec_2 = []
    for word in word_list:
        word_count_vec_1.append(seg1.count(word))
        word_count_vec_2.append(seg2.count(word))

    vec_1 = np.array(word_count_vec_1)
    vec_2 = np.array(word_count_vec_2)
    # no need to map the similarity to [0.5, 1]
    num = vec_1.dot(vec_2.T)
    denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
    cos = num / denom if denom != 0 else 0
    return cos


if __name__ == '__main__':
    whole_process()
