import yaml


def init_label(path_):
    with open(path_, 'r') as file:
        data = yaml.safe_load(file)
    index_to_name = {i: name for i, name in enumerate(data['names'])}
    name_to_index = {name: i for i, name in enumerate(data['names'])}
    return index_to_name, name_to_index


def test_args(a, b, *args):
    print(a)
    print(b)
    print(*args)


if __name__ == '__main__':
    # data_path = "/Users/hienhuynhdang/Documents/UIT/kltn/Cen-LeNet5/data/dataset/data.yaml"
    # num_to_label, label_to_num = init_label(data_path)
    # print(num_to_label)
    # print(label_to_num)
    test_args("thisisa", "thisisb", 31152223965639657336611012608, "0xF8F74aB2320a355027b68648FE04570AC54f328E")
