import torch
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer


def load_dataset(file_path="../processed_corpus/en_event_base_more.txt"):
    """
    read the data
    if chinese corpus, just load the cn_event_base_more.txt
    :param file_path:
    :return:
    """
    with open(file_path, encoding="utf-8") as f:
        data_list = eval(f.read())
    return data_list


def get_simple_division(data_list, ratio=0.8):
    """
    8：2 divide the training set and testing set
    :param data_list:
    :param ratio:
    :param shuffle:
    :return:
    """

    division_dict = {}

    # training set and testing set
    n_total = len(data_list)
    offset = int(n_total * ratio)

    # shuffle the data
    random.seed(126)
    random.shuffle(data_list)

    # divide

    """
    if 5-fold cross-validation, just simply divide the training set and testing set differently each time
    """
    train_event = data_list[:offset]
    test_event = data_list[offset:]

    # get the input and label
    train_input1, train_input2, train_input3, train_label = get_input_label(train_event, mode="train")
    test_input1, test_input2, test_input3, test_label = get_input_label(test_event, mode="test")

    division_dict["train_input1"] = train_input1
    division_dict["train_input2"] = train_input2
    division_dict["train_input3"] = train_input3
    division_dict["train_label"] = train_label
    division_dict["test_input1"] = test_input1
    division_dict["test_input2"] = test_input2
    division_dict["test_input3"] = test_input3
    division_dict["test_label"] = test_label
    # it needs to return an additional complete list to get some metrics
    division_dict["test_event"] = test_event

    return division_dict


def get_input_label(data_list_sub, mode):
    """
    get the input and label
    :return: inputs1: [local], inputs1: [global],inputs1: [similarity]  labels: [0, 1, ...]
    """
    inputs1, inputs2, inputs3, labels = [], [], [], []
    if mode == "test":
        for event in data_list_sub:
            # event
            event_sentence = event["item_3"]
            # concat the sentences which represents the global semantics
            global_sentence = event_sentence
            for idx in event["item_11"]:
                global_sentence += event["item_7"][idx]

            # iterate all sentences
            for each_index in range(event["item_6"]):
                pos_or_neg_pair = event_sentence + " " + event["item_7"][each_index]
                inputs1.append(pos_or_neg_pair)
                inputs2.append(global_sentence)
                inputs3.append(event["item_10"][each_index])
                if each_index in event["item_5"]:  # if is evidential sentence
                    labels.append(1)
                else:
                    labels.append(0)

    if mode == "train":
        for event in data_list_sub:
            # event
            event_sentence = event["item_3"]
            # concat the sentences which represents the global semantics
            global_sentence = event_sentence
            for idx in event["item_11"]:
                global_sentence += event["item_7"][idx]

            # iterate all sentences
            for each_index in range(event["item_6"]):
                pos_or_neg_pair = event_sentence + " " + event["item_7"][each_index]
                if each_index in event["item_5"]:  # if is evidential sentence
                    inputs1.append(pos_or_neg_pair)
                    inputs2.append(global_sentence)
                    inputs3.append(event["item_10"][each_index])
                    labels.append(1)
                if each_index in event['item_12']:  # if not
                    inputs1.append(pos_or_neg_pair)
                    inputs2.append(global_sentence)
                    inputs3.append(event["item_10"][each_index])
                    labels.append(0)
    return inputs1, inputs2, inputs3, labels


def data_to_tensor(data_dict):
    """
    get the input and output in tensor form by BERT tokenizer
    :param data_dict:
    :return:
    """
    data_tensor_dict = {}

    """
    if run on Chinese corpus, just replace the "bert-base-uncased" with "bert-base-chinese"
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    encoded_train_input1 = tokenizer(data_dict["train_input1"], max_length=512, truncation=True,
                                     padding='max_length', return_tensors="pt")
    encoded_train_input2 = tokenizer(data_dict["train_input2"], max_length=512, truncation=True,
                                     padding='max_length', return_tensors="pt")

    train_input1_tensor = encoded_train_input1['input_ids']
    train_input2_tensor = encoded_train_input2['input_ids']
    train_input3_tensor = torch.tensor(data_dict["train_input3"])
    train_label_tensor = torch.tensor(data_dict["train_label"])

    encoded_test_input1 = tokenizer(data_dict["test_input1"], max_length=512, truncation=True,
                                    padding='max_length', return_tensors="pt")
    encoded_test_input2 = tokenizer(data_dict["test_input2"], max_length=512, truncation=True,
                                    padding='max_length', return_tensors="pt")

    test_input1_tensor = encoded_test_input1['input_ids']
    test_input2_tensor = encoded_test_input2['input_ids']
    test_input3_tensor = torch.tensor(data_dict["test_input3"])
    test_label_tensor = torch.tensor(data_dict["test_label"])

    data_tensor_dict['train_input1'] = train_input1_tensor
    data_tensor_dict['train_input2'] = train_input2_tensor
    data_tensor_dict['train_input3'] = train_input3_tensor
    data_tensor_dict['train_label'] = train_label_tensor
    data_tensor_dict['test_input1'] = test_input1_tensor
    data_tensor_dict['test_input2'] = test_input2_tensor
    data_tensor_dict['test_input3'] = test_input3_tensor
    data_tensor_dict['test_label'] = test_label_tensor

    return data_tensor_dict


def get_dataloader(data_dict, batch_size):

    train_input1, train_input2, train_input3, train_label = data_dict["train_input1"], data_dict["train_input2"], \
                                                            data_dict["train_input3"], data_dict["train_label"]
    test_input1, test_input2, test_input3, test_label = data_dict["test_input1"], data_dict["test_input2"], \
                                                        data_dict["test_input3"], data_dict["test_label"]

    # construct dataset
    train_dataset = TensorDataset(train_input1, train_input2, train_input3, train_label)
    test_dataset = TensorDataset(test_input1, test_input2, test_input3, test_label)

    # construct dataloader
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    # the testing batch can be larger
    val_batch_size = batch_size * 5

    # sequentially is fine
    validation_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=val_batch_size
    )
    return train_dataloader, validation_dataloader
