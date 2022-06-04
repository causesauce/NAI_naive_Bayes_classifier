from data_instance import DataObject


def parse_data(data_source, data_repr: DataObject):
    data_tmp = []
    labels_dict_tmp = dict()

    number = 1
    for i in data_source:
        raw_data = i.strip().split(',')
        tmp_data_row, tmp_label = raw_data, raw_data[-1]

        if tmp_label not in labels_dict_tmp.keys():
            labels_dict_tmp[tmp_label] = 1
        else:
            labels_dict_tmp[tmp_label] += 1

        data_tmp.append(tmp_data_row)

    # for i in labels_dict_tmp.keys():
    #     labels_dict_tmp[i] /= len(data_tmp)

    data_repr.data_list = data_tmp
    data_repr.data_labels_dict = labels_dict_tmp


if __name__ == '__main__':
    training_file_path = 'train/train'  # input('provide training file path: ')
    data_object = DataObject()
    f = open(training_file_path)
    want_test_own_instance = input('provide 1 if you want to test your own data instance: ')
    print('parsing training data...')
    parse_data(f, data_object)
    f.close()

    if want_test_own_instance == '1':
        test_data_row = input('provide your data instance: ')
        predicted, actual = data_object.predict(test_data_row)
        print(predicted, actual)
    else:
        test_file_path = 'test/test'
        f = open(test_file_path)
        total_counter = 0
        true_class_counter = 0
        for plain_data_row in f:
            total_counter += 1
            predicted, actual = data_object.predict(plain_data_row)
            print(f'{predicted=}, {actual=}, {predicted == actual}')
            if predicted == actual:
                true_class_counter += 1

        print(f'accuracy of the classifier according to the given test set is: {(true_class_counter / total_counter)}|')
