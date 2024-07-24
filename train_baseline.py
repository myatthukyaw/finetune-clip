import os
import torch
import argparse

from tqdm import tqdm
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms, ToTensor

from models import ResNet18, get_model
from tools.dataset import ImageNetteDataset
from tools.file import get_output_dir
from tools.baseline_utils import train, eval


def main(args):

    writer = SummaryWriter(f"runs/{args.dataset}-{args.model}-baseline-exp")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2470, 0.2435, 0.2616)), ])

    if args.dataset == 'cifar10':
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        training_set = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root="data", train=False, download=False, transform=transform)

    elif args.dataset == 'imagenette':
        train_image_path = os.path.join(args.dataset_path, "train")
        val_image_path = os.path.join(args.dataset_path, "val")
        classes = os.listdir(train_image_path)

        training_set = ImageNetteDataset(train_image_path, classes, transform=transform)
        test_set = ImageNetteDataset(val_image_path, classes, transform=transform)

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model = get_model(args.model, num_classes=len(classes))
    model.to(args.device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)

    # get some random images
    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    writer.add_graph(model, images.to(args.device))

    output_dir = get_output_dir(args)
    best_acc = 0.0

    for epoch in range(args.epochs):

        total_train_loss = train(train_loader, args.device, model, loss_fun, optimizer, epoch, args.epochs)
        total_eval_loss, eval_acc = eval(test_loader, args.device, model, loss_fun)

        writer.add_scalar('Train loss', total_train_loss, epoch+1)
        writer.add_scalar('Eval loss', total_eval_loss, epoch+1)
        writer.add_scalar('Eval accuracy', eval_acc, epoch+1)

        #print(f"Training Loss : {total_train_loss/len(train_loader)}")
        #print(f"Evaluation Loss : {total_eval_loss/len(test_loader)}")
        print(f"Evaluation Accuracy : {eval_acc:.2f}% \n")
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), f"{output_dir}/{args.model}_best_model.pth")
    
    torch.save(model.state_dict(), f"{output_dir}/{args.model}_last_model.pth")
    writer.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenette"])
    parser.add_argument("--dataset_path", type=str, default="/mnt/d/Workspace/Data/datasets/imagenette2/imagenette2", 
                                                    help="set the path to the dataset if you are using imagenette dataset")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()
    main(args)
