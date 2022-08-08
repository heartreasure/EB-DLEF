import time
from EB_DLEF.event_factuality_identification.task2_data_process import *
from EB_DLEF.event_factuality_identification.task2_model_base import Bert_Model
from EB_DLEF.event_factuality_identification.task2_train_and_eval import *
from torch import optim

device = torch.device("cuda:7") if torch.cuda.is_available() else 'cpu'


def write_res(eval_dict, mode):
    print("    " + mode + " set(CT+): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["ct_pos_p"],
        eval_dict["ct_pos_r"],
        eval_dict["ct_pos_f1"],
    ))
    print("    " + mode + " set(CT-): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["ct_neg_p"],
        eval_dict["ct_neg_r"],
        eval_dict["ct_neg_f1"],
    ))
    print("    " + mode + " set(PS+): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["ps_pos_p"],
        eval_dict["ps_pos_r"],
        eval_dict["ps_pos_f1"],
    ))
    print("    " + mode + " set(PS-): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["ps_neg_p"],
        eval_dict["ps_neg_r"],
        eval_dict["ps_neg_f1"],
    ))
    print("    " + mode + " set(Uu): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["uu_p"],
        eval_dict["uu_r"],
        eval_dict["uu_f1"],
    ))
    print("    " + mode + " set(macro_all): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["macro_p_all"],
        eval_dict["macro_r_all"],
        eval_dict["macro_f1_all"],
    ))
    print("    " + mode + " set(micro_all): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["micro_p_all"],
        eval_dict["micro_r_all"],
        eval_dict["micro_f1_all"],
    ))
    print("    " + mode + " set(macro): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["macro_p"],
        eval_dict["macro_r"],
        eval_dict["macro_f1"],
    ))
    print("    " + mode + " set(micro): P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f}".format(
        eval_dict["micro_p"],
        eval_dict["micro_r"],
        eval_dict["micro_f1"],
    ))
    if mode != "Training":
        print("    " + mode + " set(fever): NoEvi_Acc:{0:.4f} \tEvi_Acc:{1:.4f}".format(
            eval_dict["no_evi_acc"],
            eval_dict["evi_acc"],
        ))
        print("    " + mode + " set(fever): One_NoEvi:{0:.4f} \tOne_Evi:{1:.4f}".format(
            eval_dict["one_no_evi_acc"],
            eval_dict["one_evi_acc"],
        ))
        print("    " + mode + " set(fever): More_NoEvi:{0:.4f} \tMore_Evi:{1:.4f}".format(
            eval_dict["more_no_evi_acc"],
            eval_dict["more_evi_acc"],
        ))
    print("    " + mode + " set(loss): Loss:{0:.4f}".format(
        eval_dict["avg_val_loss"]
    ))


def run_simple(batch_size, learning_rate, epochs):
    # pre-process and dataloader
    train_list, test_list = load_dataset()
    data_dict = get_simple_division(train_list, test_list)
    data_tensor_dict = data_to_tensor(data_dict)

    # model and optimizer
    model = Bert_Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # best result
    best_dict = {}
    best_evi_acc = -1
    best_epoch, best_epoch_time = -1, -1
    start_train_time = time.time()
    # train and test
    for epoch_i in range(epochs):
        train(data_tensor_dict, model, optimizer, device, epoch_i, "NoneFold")
        # train_eval_dict = test_train(None, data_tensor_dict, model, device, mode="train")
        # write_res(train_eval_dict, "Training")
        test_eval_dict = test(data_dict["test_event"], data_tensor_dict, model, device, mode="test")
        write_res(test_eval_dict, "Testing")
        # update the best
        if test_eval_dict["evi_acc"] > best_evi_acc:
            best_dict = test_eval_dict
            # best_epoch, best_train_f1 = epoch_i + 1, train_eval_dict["avg_val_loss"]
            best_epoch = epoch_i + 1
            best_epoch_time = time.time() - start_train_time
            # # save
            # save_path = "task2_en.pth"
            # torch.save(model.state_dict(), save_path)
            best_evi_acc = test_eval_dict["evi_acc"]

    # best result
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("  Best Epoch:", best_epoch)
    print("  Best Epoch Time:", format_time(best_epoch_time))
    write_res(best_dict, "BestRes")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == '__main__':
    batch_size = 4
    learning_rate = 0.00001
    epochs = 5
    run_simple(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)
