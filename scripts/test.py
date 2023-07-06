import yaml


def init_label(path_):
    with open(path_, 'r') as file:
        data = yaml.safe_load(file)
    index_to_name = {i: name for i, name in enumerate(data['names'])}
    name_to_index = {name: i for i, name in enumerate(data['names'])}
    return index_to_name, name_to_index


if __name__ == '__main__':
    data_path = "/Users/hienhuynhdang/Documents/UIT/kltn/Cen-LeNet5/data/dataset/data.yaml"
    num_to_label, label_to_num = init_label(data_path)
    print(num_to_label)
    print(label_to_num)
