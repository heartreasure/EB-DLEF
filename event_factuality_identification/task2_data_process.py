import torch
from transformers import AutoTokenizer


def load_dataset():
    """
    read data
    if run on Chinese corpus, just load the task2_train_cn and task2_test_cn
    :param file_path:
    :return:
    """
    train_file_path = "get_task_input/task2_inputs/task2_train_en.txt"
    test_file_path = "get_task_input/task2_inputs/task2_test_en.txt"
    with open(train_file_path, encoding="utf-8") as f:
        train_list = eval(f.read())
    with open(test_file_path, encoding="utf-8") as f:
        test_list = eval(f.read())

    return train_list, test_list


def get_simple_division(train_list, test_list):
    """
    8：2 divide training set and testing set
    :return:
    """
    division_dict = {}

    train_input1, train_input2, train_input3, train_input4, train_input5, train_label = get_input_label(train_list)

    test_input1, test_input2, test_input3, test_input4, test_input5, test_label = get_input_label(test_list)

    division_dict["train_input1"] = train_input1
    division_dict["train_input2"] = train_input2
    division_dict["train_input3"] = train_input3
    division_dict["train_input4"] = train_input4
    division_dict["train_input5"] = train_input5
    division_dict["train_label"] = train_label
    division_dict["test_input1"] = test_input1
    division_dict["test_input2"] = test_input2
    division_dict["test_input3"] = test_input3
    division_dict["test_input4"] = test_input4
    division_dict["test_input5"] = test_input5
    division_dict["test_label"] = test_label

    division_dict["test_event"] = test_list

    return division_dict


def get_input_label(data_list_sub):
    inputs1, inputs2, inputs3, inputs4, inputs5, labels = [], [], [], [], [], []

    for event in data_list_sub:
        inputs1.append(event["item_3"])
        evidence_list = []
        evidence_concat = ""
        for each_index in event["item_16"]:
            evidence_list.append(event["item_7"][each_index])
            evidence_concat += event["item_7"][each_index]
        inputs2.append(evidence_list)
        # all evidential sentences concat
        inputs3.append(evidence_concat)
        # global sentences concat
        global_sentence = ""
        for each_index in event["item_18"]:
            global_sentence += event["item_7"][each_index]
        inputs4.append(global_sentence)
        # similarity list
        for i in range(len(event["item_17"])):
            event["item_17"][i][i] = 1.0
        inputs5.append(event["item_17"])
        # label
        if event["item_4"] == 'CT+':
            labels.append(0)
        elif event["item_4"] == 'CT-':
            labels.append(1)
        elif event["item_4"] == 'PS+':
            labels.append(2)
        elif event["item_4"] == 'PS-':
            labels.append(3)
        elif event["item_4"] == 'Uu':
            labels.append(4)

    return inputs1, inputs2, inputs3, inputs4, inputs5, labels


def get_event_tensor(event_list):
    # bert tokenizer
    """
        if run on Chinese corpus, just replace the "bert-base-uncased" with "bert-base-chinese"
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    all_input1, all_input2, all_input3, all_label = [], [], [], []
    for event in event_list:
        inputs1, inputs2, inputs3, labels = [], [], [], []
        # event
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


def data_to_tensor(data_dict):
    data_tensor_dict = {}
    """
        if run on Chinese corpus, just replace the "bert-base-uncased" with "bert-base-chinese"
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # training set
    encoded_train_input1 = tokenizer(data_dict["train_input1"], max_length=512, truncation=True,
                                     padding='max_length', return_tensors="pt")
    flatten_train_input2 = []
    for i in range(len(data_dict["train_input2"])):
        encoded_train_input2 = tokenizer(data_dict["train_input2"][i], max_length=512, truncation=True,
                                         padding='max_length', return_tensors="pt")
        train_input2_tensor = encoded_train_input2['input_ids']
        flatten_train_input2.append(train_input2_tensor)
    encoded_train_input3 = tokenizer(data_dict["train_input3"], max_length=512, truncation=True,
                                     padding='max_length', return_tensors="pt")
    encoded_train_input4 = tokenizer(data_dict["train_input4"], max_length=512, truncation=True,
                                     padding='max_length', return_tensors="pt")
    train_input1_tensor = encoded_train_input1['input_ids']
    train_input3_tensor = encoded_train_input3['input_ids']
    train_input4_tensor = encoded_train_input4['input_ids']

    train_input5_tensor = data_dict["train_input5"]
    train_label_tensor = torch.tensor(data_dict["train_label"])

    # testing set
    encoded_test_input1 = tokenizer(data_dict["test_input1"], max_length=512, truncation=True,
                                    padding='max_length', return_tensors="pt")
    flatten_test_input2 = []
    for i in range(len(data_dict["test_input2"])):
        encoded_test_input2 = tokenizer(data_dict["test_input2"][i], max_length=512, truncation=True,
                                        padding='max_length', return_tensors="pt")
        test_input2_tensor = encoded_test_input2['input_ids']
        flatten_test_input2.append(test_input2_tensor)

    encoded_test_input3 = tokenizer(data_dict["test_input3"], max_length=512, truncation=True,
                                    padding='max_length', return_tensors="pt")
    encoded_test_input4 = tokenizer(data_dict["test_input4"], max_length=512, truncation=True,
                                    padding='max_length', return_tensors="pt")

    test_input1_tensor = encoded_test_input1['input_ids']

    test_input3_tensor = encoded_test_input3['input_ids']
    test_input4_tensor = encoded_test_input4['input_ids']
    test_input5_tensor = data_dict["test_input5"]
    test_label_tensor = torch.tensor(data_dict["test_label"])

    data_tensor_dict['train_input1'] = train_input1_tensor
    data_tensor_dict['train_input2'] = flatten_train_input2
    data_tensor_dict['train_input3'] = train_input3_tensor
    data_tensor_dict['train_input4'] = train_input4_tensor
    data_tensor_dict['train_input5'] = train_input5_tensor
    data_tensor_dict['train_label'] = train_label_tensor

    data_tensor_dict['test_input1'] = test_input1_tensor
    data_tensor_dict['test_input2'] = flatten_test_input2
    data_tensor_dict['test_input3'] = test_input3_tensor
    data_tensor_dict['test_input4'] = test_input4_tensor
    data_tensor_dict['test_input5'] = test_input5_tensor
    data_tensor_dict['test_label'] = test_label_tensor
    return data_tensor_dict


