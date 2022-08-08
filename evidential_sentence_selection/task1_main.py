import time
from EB_DLEF.evidential_sentence_selection.task1_data_process import *
from EB_DLEF.evidential_sentence_selection.task1_model import Bert_Model
from torch import optim
from EB_DLEF.evidential_sentence_selection.task1_train_and_eval import *

device = torch.device("cuda:6") if torch.cuda.is_available() else 'cpu'


def run_simple(batch_size, learning_rate, epochs):
    # process data，huild dataloader
    data_list = load_dataset()
    data_dict = get_simple_division(data_list)
    data_tensor_dict = data_to_tensor(data_dict)
    train_dataloader, validation_dataloader = get_dataloader(data_tensor_dict, batch_size=batch_size)

    # model and optimizer
    model = Bert_Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # best res
    best_p, best_r, best_f1 = 0, 0, 0
    best_epoch, best_epoch_time = 0, 0
    start_train_time = time.time()
    # train and test
    for epoch_i in range(epochs):
        train(train_dataloader, model, optimizer, device, epoch_i, "NoneFold")
        test_p_1, test_r_1, test_f1_1, test_avg_val_loss = test(validation_dataloader, model, device)
        print(
            "    Testing Set1: P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f} \tLoss:{3:.4f}".format(test_p_1, test_r_1, test_f1_1,
                                                                                         test_avg_val_loss))
        # eval_dict = event_test(data_dict['test_event'], model, device)
        # test_p_2, test_r_2, test_f1_2, test_f1_2_cal = eval_dict['all_p'], eval_dict['all_r'], \
        #                                                eval_dict['all_f1'], eval_dict['all_cal_f1']
        #
        # print("    Testing Set2: P:{0:.4f} \tR:{1:.4f} \tF1:{2:.4f} \tCal F1:{3:.4f}".format(
        #     test_p_2,
        #     test_r_2,
        #     test_f1_2,
        #     test_f1_2_cal,
        # ))

        # update the best result
        if test_f1_1 > best_f1:
            best_epoch = epoch_i + 1
            best_epoch_time = time.time() - start_train_time
            best_p, best_r, best_f1 = test_p_1, test_r_1, test_f1_1
            # # save the model
            # save_path = "../event_factuality_identification/get_task_input/task1_checkpoint/task1_en.pth"
            # torch.save(model.state_dict(), save_path)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("  Best  Result: Epoch:{0} \tTook:{1} \tP:{2:.4f} \tR:{3:.4f} \tF1:{4:.4f}".format(
        best_epoch, format_time(best_epoch_time), best_p, best_r, best_f1))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == '__main__':
    batch_size = 4
    learning_rate = 0.00001
    epochs = 3

    run_simple(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)
