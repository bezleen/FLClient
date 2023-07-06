# --- Open cmt line bellow if run by cmd: python *.py
# import sys  # nopep8
# sys.path.append(".")  # nopep8
# ----

from torchvision import transforms


class ImageTransform():
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __init__(self, resize: tuple = (32, 32), mean: tuple = (0.1307,), std: tuple = (0.3081,)):
        self.data_transform = {
            self.TRAIN: transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            self.VAL: transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            self.TEST: transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        assert phase in [self.TRAIN, self.VAL, self.TEST]
        return self.data_transform[phase](img)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from PIL import Image
#     img_transform = ImageTransform()
#     img_path = "data/dataset/a/642af01a63436e669d877bbe.jpg"
#     img = Image.open(img_path)
#     img_transformed = img_transform(img, Consts.TRAIN)
#     # (channel, height, width) -> (height, width, channel)
#     img_transformed = img_transformed.numpy().transpose(1, 2, 0)
#     img_transformed = np.clip(img_transformed, 0, 1)
#     print(img_transformed.shape)
#     plt.imshow(img_transformed)
#     plt.show()
