class DataObject:

    def __init__(self):
        self.data_list = []
        self.data_labels_dict = dict()

    def predict(self, test_data_row: str):
        test_data = []
        for attribute in test_data_row.strip().split(','):
            test_data.append(attribute)

        # print(test_data)
        corresponding_bayes_probabilities = []

        for one_key in self.data_labels_dict.keys():
            key_probability = self.data_labels_dict[one_key] / len(self.data_list)

            for test_attribute_position in range(len(test_data) - 1):
                attribute_counter = 0
                for data_row in self.data_list:
                    if data_row[-1] == one_key \
                            and data_row[test_attribute_position] == test_data[test_attribute_position]:
                        attribute_counter += 1

                attribute_cond_probability = attribute_counter / self.data_labels_dict[one_key]
                if attribute_cond_probability == 0:
                    # smoothing
                    diff_attr_values = set()
                    for data_row_tmp in self.data_list:
                        diff_attr_values.add(data_row_tmp[test_attribute_position])
                    diff_attr_number = len(diff_attr_values)
                    attribute_cond_probability = 1 / (self.data_labels_dict[one_key] + diff_attr_number)
                key_probability *= attribute_cond_probability
            corresponding_bayes_probabilities.append(key_probability)

        prediction_dict = dict()
        index = 0
        for key in self.data_labels_dict.keys():
            prediction_dict[key] = corresponding_bayes_probabilities[index]
            index += 1

        maximum = -1
        label = 'none'
        for i in prediction_dict.keys():
            if maximum < prediction_dict[i]:
                maximum = prediction_dict[i]
                label = i

        return label, test_data[-1]
