import torch
import datetime
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def train(data_dict, model, optimizer, device, epoch, fold):
    model.train()
    training_bar = tqdm(range(len(data_dict["train_input1"])))
    for idx in training_bar:
        training_bar.set_description("{0}/epoch_{1}".format(fold, epoch + 1))
        train_inputs1 = data_dict["train_input1"][idx].to(device)
        train_inputs2 = data_dict["train_input2"][idx].to(device)
        train_inputs3 = data_dict["train_input3"][idx].to(device)
        train_inputs4 = data_dict["train_input4"][idx].to(device)
        train_inputs5 = torch.tensor(data_dict["train_input5"][idx]).to(device)
        train_labels = data_dict["train_label"][idx].unsqueeze(0).to(device)
        model.zero_grad()
        outputs = model(train_inputs1, train_inputs2, train_inputs3, train_inputs4, train_inputs5)
        loss = F.cross_entropy(outputs, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def test(event_list, data_dict, model, device, mode):
    model.eval()
    total_eval_loss = 0
    final_pred = []
    final_label = []
    testing_bar = tqdm(range(len(data_dict["test_input1"])))
    for idx in testing_bar:
        testing_bar.set_description("  Testing")
        test_inputs1 = data_dict["test_input1"][idx].to(device)
        test_inputs2 = data_dict["test_input2"][idx].to(device)
        test_inputs3 = data_dict["test_input3"][idx].to(device)
        test_inputs4 = data_dict["test_input4"][idx].to(device)
        test_inputs5 = torch.tensor(data_dict["test_input5"][idx]).to(device)
        labels = data_dict["test_label"][idx].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(test_inputs1, test_inputs2, test_inputs3, test_inputs4, test_inputs5)
        total_eval_loss += F.cross_entropy(outputs, labels.flatten())
        preds = outputs.detach().cpu().numpy()  # [batch, 2]
        labels = labels.cpu().numpy()  # [batch]
        final_pred.extend(preds)
        final_label.extend(labels)
    avg_val_loss = total_eval_loss / len(data_dict["test_input1"])
    eval_dict = get_accuracy(final_pred, final_label)
    if mode == "test":
        no_evi_acc, evi_acc = get_fever_accuray(final_pred, final_label, event_list)
        one_no_evi_acc, one_evi_acc, more_no_evi_acc, more_evi_acc = get_level_fever(final_pred, final_label, event_list)
    else:
        no_evi_acc, evi_acc = 0, 0
        one_no_evi_acc, one_evi_acc, more_no_evi_acc, more_evi_acc = 0, 0, 0, 0
    eval_dict['avg_val_loss'] = avg_val_loss
    eval_dict['no_evi_acc'] = no_evi_acc
    eval_dict['evi_acc'] = evi_acc
    eval_dict['one_no_evi_acc'] = one_no_evi_acc
    eval_dict['one_evi_acc'] = one_evi_acc
    eval_dict['more_no_evi_acc'] = more_no_evi_acc
    eval_dict['more_evi_acc'] = more_evi_acc
    return eval_dict


def get_fever_accuray(final_pred, final_label, event_list):
    no_evi_count, evi_count = 0, 0
    event_count = len(event_list)
    final_pred = np.argmax(final_pred, axis=1).flatten()

    for i in range(event_count):
        # noScoreEv
        if final_pred[i] == final_label[i]:
            no_evi_count += 1
            # ScoreEv
            if event_list[i]["item_5"] == event_list[i]["item_16"]:
                evi_count += 1
    return no_evi_count / event_count, evi_count / event_count


def get_level_fever(final_pred, final_label, event_list):

    one_no_evi_count, one_evi_count, one_sum = 0, 0, 0
    more_no_evi_count, more_evi_count, more_sum = 0, 0, 0

    event_count = len(event_list)
    final_pred = np.argmax(final_pred, axis=1).flatten()
    for i in range(event_count):
        # evidence count
        evidence_count = len(event_list[i]["item_5"])
        # if == 1
        if evidence_count == 1:
            one_sum += 1
            if final_pred[i] == final_label[i]:
                one_no_evi_count += 1
                # ScoreEv
                if event_list[i]["item_5"] == event_list[i]["item_16"]:
                    one_evi_count += 1
        # if >= 1
        else:
            more_sum += 1
            if final_pred[i] == final_label[i]:
                more_no_evi_count += 1
                # ScoreEv
                if event_list[i]["item_5"] == event_list[i]["item_16"]:
                    more_evi_count += 1
    res1 = one_no_evi_count / one_sum if one_sum != 0 else 0
    res2 = one_evi_count / one_sum if one_sum != 0 else 0
    res3 = more_no_evi_count / more_sum if more_sum != 0 else 0
    res4 = more_evi_count / more_sum if more_sum != 0 else 0

    return res1, res2, res3, res4


def get_accuracy(preds, labels):
    """
    CT+：0  PS+：1  CT-：2  PS-: 3  Uu: 4
    :param preds: [all_test_sample]
    :param labels: [all_test_sample]
    :return: P，R，F1,（macro P R F1), (micro P R F1)
    """
    preds = np.argmax(preds, axis=1).flatten()
    sample_count = len(preds)

    # 1.CT+
    TP_1, FN_1, FP_1, TN_1 = 0, 0, 0, 0
    for i in range(sample_count):
        if labels[i] == 0:
            if preds[i] == 0:
                TP_1 += 1
            else:
                FN_1 += 1
        else:
            if preds[i] == 0:
                FP_1 += 1

    ct_pos_precision = TP_1 / (TP_1 + FP_1) if (TP_1 + FP_1) != 0 else 0
    ct_pos_recall = TP_1 / (TP_1 + FN_1) if (TP_1 + FN_1) != 0 else 0
    ct_pos_f1 = (2 * ct_pos_precision * ct_pos_recall) / (ct_pos_precision + ct_pos_recall) if \
        (ct_pos_precision + ct_pos_recall) != 0 else 0

    # 2.PS+
    TP_2, FN_2, FP_2, TN_2 = 0, 0, 0, 0
    for i in range(sample_count):
        if labels[i] == 1:
            if preds[i] == 1:
                TP_2 += 1
            else:
                FN_2 += 1
        else:
            if preds[i] == 1:
                FP_2 += 1
    ps_pos_precision = TP_2 / (TP_2 + FP_2) if (TP_2 + FP_2) != 0 else 0
    ps_pos_recall = TP_2 / (TP_2 + FN_2) if (TP_2 + FN_2) != 0 else 0
    ps_pos_f1 = (2 * ps_pos_precision * ps_pos_recall) / (ps_pos_precision + ps_pos_recall) if \
        (ps_pos_precision + ps_pos_recall) != 0 else 0

    # 3.CT-
    TP_3, FN_3, FP_3, TN_3 = 0, 0, 0, 0
    for i in range(sample_count):
        if labels[i] == 2:
            if preds[i] == 2:
                TP_3 += 1
            else:
                FN_3 += 1
        else:
            if preds[i] == 2:
                FP_3 += 1

    ct_neg_precision = TP_3 / (TP_3 + FP_3) if (TP_3 + FP_3) != 0 else 0
    ct_neg_recall = TP_3 / (TP_3 + FN_3) if (TP_3 + FN_3) != 0 else 0
    ct_neg_f1 = (2 * ct_neg_precision * ct_neg_recall) / (ct_neg_precision + ct_neg_recall) if \
        (ct_neg_precision + ct_neg_recall) != 0 else 0

    # 4.PS-
    TP_4, FN_4, FP_4, TN_4 = 0, 0, 0, 0
    for i in range(sample_count):
        if labels[i] == 3:
            if preds[i] == 3:
                TP_4 += 1
            else:
                FN_4 += 1
        else:
            if preds[i] == 3:
                FP_4 += 1

    ps_neg_precision = TP_4 / (TP_4 + FP_4) if (TP_4 + FP_4) != 0 else 0
    ps_neg_recall = TP_4 / (TP_4 + FN_4) if (TP_4 + FN_4) != 0 else 0
    ps_neg_f1 = (2 * ps_neg_precision * ps_neg_recall) / (ps_neg_precision + ps_neg_recall) if \
        (ps_neg_precision + ps_neg_recall) != 0 else 0

    # 5.Uu
    TP_5, FN_5, FP_5, TN_5 = 0, 0, 0, 0

    for i in range(sample_count):
        if labels[i] == 4:
            if preds[i] == 4:
                TP_5 += 1
            else:
                FN_5 += 1
        else:
            if preds[i] == 4:
                FP_5 += 1

    uu_precision = TP_5 / (TP_5 + FP_5) if (TP_5 + FP_5) != 0 else 0
    uu_recall = TP_5 / (TP_5 + FN_5) if (TP_5 + FN_5) != 0 else 0
    uu_f1 = (2 * uu_precision * uu_recall) / (uu_precision + uu_recall) if (uu_precision + uu_recall) != 0 else 0

    # macro and micro
    macro_p = (ct_pos_precision + ps_pos_precision + ct_neg_precision) / 3
    macro_r = (ct_pos_recall + ps_pos_recall + ct_neg_recall) / 3
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) != 0 else 0

    macro_p_2 = (ct_pos_precision + ps_pos_precision + ct_neg_precision + ps_neg_precision + uu_precision) / 5
    macro_r_2 = (ct_pos_recall + ps_pos_recall + ct_neg_recall + ps_neg_recall + uu_recall) / 5
    macro_f1_2 = (2 * macro_p_2 * macro_r_2) / (macro_p_2 + macro_r_2) if (macro_p_2 + macro_r_2) != 0 else 0

    TP_avg = (TP_1 + TP_2 + TP_3)
    FN_avg = (FN_1 + FN_2 + FN_3)
    FP_avg = (FP_1 + FP_2 + FP_3)

    TP_avg_2 = (TP_1 + TP_2 + TP_3 + TP_4 + TP_5)
    FN_avg_2 = (FN_1 + FN_2 + FN_3 + FN_4 + FN_5)
    FP_avg_2 = (FP_1 + FP_2 + FP_3 + FP_4 + FP_5)

    micro_p = TP_avg / (TP_avg + FP_avg) if (TP_avg + FP_avg) != 0 else 0
    micro_r = TP_avg / (TP_avg + FN_avg) if (TP_avg + FN_avg) != 0 else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) != 0 else 0

    micro_p_2 = TP_avg_2 / (TP_avg_2 + FP_avg_2) if (TP_avg_2 + FP_avg_2) != 0 else 0
    micro_r_2 = TP_avg_2 / (TP_avg_2 + FN_avg_2) if (TP_avg_2 + FN_avg_2) != 0 else 0
    micro_f1_2 = (2 * micro_p_2 * micro_r_2) / (micro_p_2 + micro_r_2) if (micro_p_2 + micro_r_2) != 0 else 0

    res_dict = {"ct_pos_p": ct_pos_precision, "ct_pos_r": ct_pos_recall, "ct_pos_f1": ct_pos_f1,
                "ct_neg_p": ct_neg_precision, "ct_neg_r": ct_neg_recall, "ct_neg_f1": ct_neg_f1,
                "ps_pos_p": ps_pos_precision, "ps_pos_r": ps_pos_recall, "ps_pos_f1": ps_pos_f1,
                "ps_neg_p": ps_neg_precision, "ps_neg_r": ps_neg_recall, "ps_neg_f1": ps_neg_f1,
                "uu_p": uu_precision, "uu_r": uu_recall, "uu_f1": uu_f1,
                "macro_p_all": macro_p_2, "macro_r_all": macro_r_2, "macro_f1_all": macro_f1_2,
                "micro_p_all": micro_p_2, "micro_r_all": micro_r_2, "micro_f1_all": micro_f1_2,
                "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
                "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
                }
    return res_dict


def format_time(elapsed):
    """
    format the time as hh:mm:ss:
    """
    # round to the nearest second
    elapsed_rounded = int(round((elapsed)))

    # format to hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
