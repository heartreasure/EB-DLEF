import json

"""
for each event, we transform the json file to a python dict
item_0: event id
item_1: doc id
item_2: annotation info
item_3: event
item_4: factuality
item_5: evidential sentences list
item_6: all sentences count
item_7: all sentences list
"""


def get_event_based_dict_for_chinese():

    cn_json_path = '../corpus/DB-DLEF-CN.json'
    with open(cn_json_path, "r", encoding='utf-8') as f:
        all_res = json.load(f)

    all_event_list = []

    for doc_id in all_res.keys():
        event_count = doc_id.split("_")[-1]
        doc = all_res[doc_id]
        each_event_dict = {
            "item_0": event_count,
            "item_1": doc_id,
            "item_2": doc["annotation info"],
            "item_3": doc["event"],
            "item_4": doc["factuality"],
            "item_5": doc["evidential sentences"],
            "item_6": len(doc["all sentences"]),
            "item_7": doc["all sentences"],
        }
        all_event_list.append(each_event_dict)

    # write and save the dict
    filename = '../processed_corpus/cn_event_base.txt'
    with open(filename, 'w', encoding='utf-8') as file_object:
        file_object.write(str(all_event_list))
    return all_event_list


def get_event_based_dict_for_english():

    cn_json_path = '../corpus/DB-DLEF-EN.json'
    with open(cn_json_path, "r", encoding='utf-8') as f:
        all_res = json.load(f)

    all_event_list = []

    for doc_id in all_res.keys():
        event_count = doc_id.split("_")[-1]
        doc = all_res[doc_id]
        each_event_dict = {
            "item_0": event_count,
            "item_1": doc_id,
            "item_2": doc["annotation info"],
            "item_3": doc["event"],
            "item_4": doc["factuality"],
            "item_5": doc["evidential sentences"],
            "item_6": len(doc["all sentences"]),
            "item_7": doc["all sentences"],
        }
        all_event_list.append(each_event_dict)

    # write and save the dict
    filename = '../processed_corpus/en_event_base.txt'
    with open(filename, 'w', encoding='utf-8') as file_object:
        file_object.write(str(all_event_list))
    return all_event_list


if __name__ == '__main__':
    get_event_based_dict_for_chinese()
    get_event_based_dict_for_english()