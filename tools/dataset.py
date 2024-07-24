import os
import clip
from PIL import Image


# Define a custom dataset
class ClipImageNetteDataset():
    def __init__(self, path, classes, descriptions, preprocess):
        self.path = path
        self.classes = classes
        self.descriptions = descriptions
        self.preprocess = preprocess
        self.list_image_path = []
        self.list_labels = []
        self.list_txt = []
        self._load_data()
        self.title = clip.tokenize(self.list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.list_image_path[idx]))
        title = self.title[idx]
        label = self.list_labels[idx]
        image_path = self.list_image_path[idx]

        return image, title, label, image_path

    def _load_data(self):

        type_of_categories = os.listdir(self.path)
        for category in type_of_categories:
            images = os.listdir(self.path+ "/" + category)
            for image in images:
                if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.JPEG'):
                    self.list_image_path.append(self.path + '/' + category + '/' + image)
                    self.list_txt.append(self.descriptions[category])                
                    self.list_labels.append(self.classes.index(category))

class ImageNetteDataset():
    def __init__(self, path, classes, transform=None):
        self.path = path
        self.classes = classes
        self.transform = transform
        self.list_image_path = []
        self.list_labels = []
        self._load_data()

    def __len__(self):
        return len(self.list_image_path)

    def __getitem__(self, idx):
        image = Image.open(self.list_image_path[idx]).convert("RGB")
        label = self.list_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def _load_data(self):
        categories = os.listdir(self.path)
        for category in categories:
            images = os.listdir(self.path+ "/" + category)
            for image in images:
                if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.JPEG'):
                    self.list_image_path.append(self.path + '/' + category + '/' + image)
                    self.list_labels.append(self.classes.index(category))