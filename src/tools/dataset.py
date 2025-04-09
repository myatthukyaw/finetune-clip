import os

import clip
from PIL import Image


class ClipDataset:
    def __init__(self, path, classes, descriptions, preprocess) -> None:
        self.path = path
        self.classes = classes
        self.descriptions = descriptions
        self.preprocess = preprocess
        self.list_image_path = []
        self.list_labels = []
        self.list_txt = []
        self._load_data()
        self.title = clip.tokenize(self.list_txt)

    def __len__(self) -> int:
        return len(self.title)

    def __getitem__(self, idx: int) -> tuple:
        image = self.preprocess(Image.open(self.list_image_path[idx]))
        title = self.title[idx]
        label = self.list_labels[idx]
        image_path = self.list_image_path[idx]

        return image, title, label, image_path

    def _load_data(self) -> None:
        categories = os.listdir(self.path)
        for category in categories:
            category_path = os.path.join(self.path, category)
            images = os.listdir(category_path)
            for image in images:
                if (
                    image.endswith(".jpg")
                    or image.endswith(".png")
                    or image.endswith(".JPEG")
                ):
                    img_path = os.path.join(category_path, image)
                    self.list_image_path.append(img_path)
                    self.list_txt.append(self.descriptions[category])
                    self.list_labels.append(self.classes.index(category))


class CustomDataset:
    def __init__(self, path, classes, transform=None) -> None:
        self.path = path
        self.classes = classes
        self.transform = transform
        self.list_image_path = []
        self.list_labels = []
        self._load_data()

    def __len__(self) -> int:
        return len(self.list_image_path)

    def __getitem__(self, idx) -> tuple:
        image = Image.open(self.list_image_path[idx]).convert("RGB")
        label = self.list_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def _load_data(self) -> None:
        categories = os.listdir(self.path)
        for category in categories:
            category_path = os.path.join(self.path, category)
            images = os.listdir(category_path)
            for image in images:
                if (
                    image.endswith(".jpg")
                    or image.endswith(".png")
                    or image.endswith(".JPEG")
                ):
                    img_path = os.path.join(category_path, image)
                    self.list_image_path.append(img_path)
                    self.list_labels.append(self.classes.index(category))
