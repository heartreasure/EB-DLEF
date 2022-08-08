import stanza
import json
import random
from tqdm import tqdm
import numpy as np


def add_more_info_en():
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
    :return:
    """
    file_path = "../processed_corpus/en_event_base.txt"
    with open(file_path, encoding="utf-8") as f:
        data_list = eval(f.read())

    new_data_list = []
    # stop words
    filepath = "../stopwords/en.json"
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    with open(filepath, 'r', encoding='utf-8') as fp:
        stop_words_list = json.load(fp)
    for event in tqdm(data_list):
        pre_similarity_list = get_similarity(event["item_3"], event["item_7"], "sentence", stop_words_list, nlp)
        event['item_10'] = pre_similarity_list
        # ranking by similarity
        index_list = [i for i in range(len(pre_similarity_list))]
        index_to_similarity = dict(zip(index_list, pre_similarity_list))
        index_to_similarity_sorted = sorted(index_to_similarity.items(), key=lambda i: i[1], reverse=True)
        whole_list = []
        for idx in range(len(index_to_similarity_sorted)):
            whole_list.append(index_to_similarity_sorted[idx][0])
        half_list = whole_list[:len(whole_list) // 2]
        if 0 not in half_list:
            half_list.append(0)
        event['item_11'] = half_list

        # negative samples index
        pos_count = len(event["item_5"])
        whole_list_removed = [x for x in whole_list if x not in event["item_5"]]
        if len(whole_list_removed) <= pos_count:
            neg_list = whole_list_removed
        else:
            neg_list = []
            for i in range(pos_count // 2):
                neg_list.append(whole_list_removed[i])
            whole_list_removed = [x for x in whole_list_removed if x not in neg_list]
            rest = random.sample(whole_list_removed, pos_count - pos_count // 2)
            neg_list.extend(rest)

        event['item_12'] = neg_list
        new_data_list.append(event)

    # write and save
    filename = '../processed_corpus/en_event_base_more.txt'
    with open(filename, 'w', encoding='utf-8') as file_object:
        file_object.write(str(new_data_list))
    return new_data_list


def add_more_info_cn():
    """
    ori content
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
    :return:
    """
    file_path = "../processed_corpus/cn_event_base.txt"
    with open(file_path, encoding="utf-8") as f:
        data_list = eval(f.read())

    new_data_list = []
    # stop words
    filepath = "../stopwords/zh.json"
    nlp = stanza.Pipeline(lang='zh', processors='tokenize')
    with open(filepath, 'r', encoding='utf-8') as fp:
        stop_words_list = json.load(fp)
    for event in tqdm(data_list):
        pre_similarity_list = get_similarity(event["item_3"], event["item_7"], "sentence", stop_words_list, nlp)
        event['item_10'] = pre_similarity_list
        # ranking by similarity
        index_list = [i for i in range(len(pre_similarity_list))]
        index_to_similarity = dict(zip(index_list, pre_similarity_list))
        index_to_similarity_sorted = sorted(index_to_similarity.items(), key=lambda i: i[1], reverse=True)
        whole_list = []
        for idx in range(len(index_to_similarity_sorted)):
            whole_list.append(index_to_similarity_sorted[idx][0])
        half_list = whole_list[:len(whole_list) // 2]
        if 0 not in half_list:
            half_list.append(0)
        event['item_11'] = half_list

        # negative samples index
        pos_count = len(event["item_5"])
        whole_list_removed = [x for x in whole_list if x not in event["item_5"]]
        if len(whole_list_removed) <= pos_count:
            neg_list = whole_list_removed
        else:
            neg_list = []
            for i in range(pos_count // 2):
                neg_list.append(whole_list_removed[i])
            whole_list_removed = [x for x in whole_list_removed if x not in neg_list]
            rest = random.sample(whole_list_removed, pos_count - pos_count // 2)
            neg_list.extend(rest)

        event['item_12'] = neg_list
        new_data_list.append(event)

    # write and save
    filename = '../processed_corpus/cn_event_base_more.txt'
    with open(filename, 'w', encoding='utf-8') as file_object:
        file_object.write(str(new_data_list))
    return new_data_list


def get_similarity(event_sentence, all_sentences, type, stopwords, nlp):
    """
    seg1 and seg2 are all list, each elem is a token after split
    :return:
    """
    if type == "sentence":
        seg1_doc = nlp(event_sentence)
        seg1 = []
        for sen in seg1_doc.sentences:
            for token in sen.tokens:
                if token.text not in stopwords and token.text not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' and token.text not in "，。……——“”‘’！；":
                    seg1.append(token.text)

        seg2_list = []

        for sentence in all_sentences:
            seg2_doc = nlp(sentence)
            seg2 = []
            for sen in seg2_doc.sentences:
                for token in sen.tokens:
                    if token.text not in stopwords and token.text not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' and token.text not in "，。……——“”‘’！；":
                        seg2.append(token.text)
            seg2_list.append(seg2)

    else:
        seg1 = event_sentence
        seg2_list = all_sentences

    similarity_list = []

    for seg2 in seg2_list:
        word_list = list(set([word for word in seg1 + seg2]))  # build the word dict
        word_count_vec_1 = []
        word_count_vec_2 = []
        for word in word_list:
            word_count_vec_1.append(seg1.count(word))
            word_count_vec_2.append(seg2.count(word))

        vec_1 = np.array(word_count_vec_1)
        vec_2 = np.array(word_count_vec_2)

        num = vec_1.dot(vec_2.T)
        denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
        cos = num / denom if denom != 0 else 0

        similarity_list.append(cos)

    # mapping the value to [0.5, 1]
    new_similarity_list = []
    Xmin = np.min(similarity_list)
    Xmax = np.max(similarity_list)
    a = 0.5
    b = 1
    for X in similarity_list:
        if Xmax - Xmin == 0:
            Y = 0.5
        else:
            Y = a + (b - a) / (Xmax - Xmin) * (X - Xmin)
        new_similarity_list.append(Y)
    return new_similarity_list


if __name__ == '__main__':
    add_more_info_en()
    # add_more_info_cn()
