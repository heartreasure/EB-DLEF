import torch
import datetime
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm


def train(dataloader, model, optimizer, device, epoch, fold):
    # training mode
    model.train()
    # progress bar
    training_bar = tqdm(dataloader)

    for batch in training_bar:
        training_bar.set_description("{0}/epoch_{1}".format(fold, epoch + 1))
        train_inputs1 = batch[0].to(device)  # [batch, seq_len]
        train_inputs2 = batch[1].to(device)  # [batch, seq_len]
        train_inputs3 = batch[2].to(device)  # [batch]
        train_labels = batch[3].to(device)  # [batch]
        # set the grad to 0
        model.zero_grad()
        outputs = model(train_inputs1, train_inputs2, train_inputs3)
        # cross-entropy loss
        loss = F.cross_entropy(outputs, train_labels)
        # back propagation
        loss.backward()
        # clipping, avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()


def test(dataloader, model, device):
    # testing mode
    model.eval()
    # loss
    total_eval_loss = 0
    # result
    final_pred = []
    final_label = []
    testing_bar = tqdm(dataloader)
    for batch in testing_bar:
        testing_bar.set_description("  Testing")
        inputs1 = batch[0].to(device)  # torch.Size([batch, seq_len])
        inputs2 = batch[1].to(device)  # torch.Size([batch, seq_len])
        inputs3 = batch[2].to(device)  # torch.Size([batch])
        labels = batch[3].to(device)  # torch.Size([batch])
        # no need grad update
        with torch.no_grad():
            outputs = model(inputs1, inputs2, inputs3)
        # cross_entropy
        total_eval_loss += F.cross_entropy(outputs, labels.flatten())
        # put into cpu
        preds = outputs.detach().cpu().numpy()  # [batch, 2]
        labels = labels.cpu().numpy()  # [batch]
        # preds and labels
        final_pred.extend(preds)
        final_label.extend(labels)
    # loss
    avg_val_loss = total_eval_loss / len(dataloader)
    # evaluation function
    P, R, F1 = get_accuracy(final_pred, final_label)
    return P, R, F1, avg_val_loss


def event_test(event_list, model, device):
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
        # no need grad update
        with torch.no_grad():
            outputs = model(inputs1, inputs2, inputs3)
        # cross_entropy
        total_eval_loss += F.cross_entropy(outputs, labels.flatten())
        # put into cpu
        each_event_pred = outputs.detach().cpu().numpy()  # [batch, 2]
        each_event_label = labels.cpu().numpy()  # [batch]
        # preds and labels
        final_pred.append(each_event_pred)
        final_label.append(each_event_label)
    avg_val_loss = total_eval_loss / len(event_list)
    eval_dict = get_event_accuracy(final_pred, final_label)
    eval_dict["avg_val_loss"] = avg_val_loss
    return eval_dict


def get_accuracy(final_pred, final_label, flatten=False):
    """
    testing of train sample
    P, R, F1
    """
    if flatten:
        pred_flat = final_pred
    else:
        pred_flat = np.argmax(final_pred, axis=1).flatten()
    samples = len(final_label)
    # some indicators to calculate P, R, F1
    TP, FP, FN = 0, 0, 0
    for i in range(samples):
        if final_label[i] == 1:  # label:1
            if pred_flat[i] == 1:  # pred:1
                TP += 1
            else:  # pred:0
                FN += 1
        else:  # label:0
            if pred_flat[i] == 1:  # pred:1
                FP += 1

    P = TP / (TP + FP) if (TP + FP) != 0 else 0
    R = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = (2 * P * R) / (P + R) if (P + R) != 0 else 0
    return P, R, F1


def get_event_accuracy(final_pred, final_label):
    res_dict = {}
    # indicators of precise
    all_p, all_r, all_f1 = [], [], []
    # indicators of top5
    top_p, top_r, top_f1 = [], [], []
    for i in range(len(final_pred)):
        # =======================precise=============================
        # Post-processing algorithms
        pred_flat = backup_process(final_pred[i], backup=False)
        each_p, each_r, each_f1 = get_accuracy(pred_flat, final_label[i], flatten=True)
        all_p.append(each_p)
        all_r.append(each_r)
        all_f1.append(each_f1)

        # =======================top5====================
        event_pred = final_pred[i]
        event_dict = dict(zip(event_pred[:, 1].tolist(), [_ for _ in range(event_pred.shape[0])]))
        sorted_dict = sorted(event_dict.items(), key=lambda kv: (kv[0], kv[1]), reverse=True)

        # select top 5 as evidence
        new_pred = [0 for _ in range(event_pred.shape[0])]
        for item in sorted_dict[:5]:
            new_pred[item[1]] = 1
        each_p, each_r, each_f1 = get_accuracy(new_pred, final_label[i], flatten=True)
        top_p.append(each_p)
        top_r.append(each_r)
        top_f1.append(each_f1)

    all_cal_f1 = (2 * np.mean(all_p) * np.mean(all_r)) / (np.mean(all_p) + np.mean(all_r)) if \
        (np.mean(all_p) + np.mean(all_r)) != 0 else 0
    top_cal_f1 = (2 * np.mean(top_p) * np.mean(top_r)) / (np.mean(top_p) + np.mean(top_r)) if \
        (np.mean(top_p) + np.mean(top_r)) != 0 else 0
    res_dict['all_p'] = np.mean(all_p)
    res_dict['all_r'] = np.mean(all_r)
    res_dict['all_f1'] = np.mean(all_f1)
    res_dict['all_cal_f1'] = all_cal_f1
    res_dict['top_p'] = np.mean(top_p)
    res_dict['top_r'] = np.mean(top_r)
    res_dict['top_f1'] = np.mean(top_f1)
    res_dict['top_cal_f1'] = top_cal_f1
    return res_dict


def backup_process(final_pred, backup):
    """
    If none of the predictions for each event has a value of 1,
    then the one with the highest probability is chosen as the evidence
    :return:
    """
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


def format_time(elapsed):
    """
    format the time as hh:mm:ss:
    """
    # round to the nearest second
    elapsed_rounded = int(round((elapsed)))

    # format to hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_event_tensor(event_list):
    """
    for event list, get the input and output, and convert to tensor
    :param event:
    :return:
    """
    # BERT tokenizer
    """
       if run on Chinese corpus, just replace the "bert-base-uncased" with "bert-base-chinese"
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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
