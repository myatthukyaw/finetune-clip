import argparse
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms

from src.models import get_model
from src.tools.dataset import CustomDataset
from src.trainer.trainer import Trainer
from src.tools.utils import get_output_dir


def main(args):

    if args.tensorboard:
        writer = SummaryWriter(f"runs/{args.dataset}-{args.model}-baseline-exp")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    if args.dataset == "cifar10":
        classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck",]
        training_set = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform
        )
        val_set = datasets.CIFAR10(
            root="data", train=False, download=False, transform=transform
        )

    elif args.dataset == "custom":
        train_image_path = os.path.join(args.dataset_path, "train")
        val_image_path = os.path.join(args.dataset_path, "val")
        classes = os.listdir(train_image_path)

        training_set = CustomDataset(train_image_path, classes, transform=transform)
        val_set = CustomDataset(val_image_path, classes, transform=transform)

    train_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = get_model(args.model, num_classes=len(classes))

    trainer = Trainer(model, classes, args)

    output_dir = get_output_dir(args)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):

        train_loss = trainer.train(train_loader, epoch)
        eval_metrics, eval_loss = trainer.eval(val_loader, epoch, output_dir)

        if args.tensorboard:
            trainer.write_to_tensorboard(writer, train_loss, eval_loss, epoch)

        if eval_metrics.accuracy > best_acc:
            best_acc = eval_metrics.accuracy
            trainer.save_model(f"{output_dir}/{args.model}_best_model.pth")

    trainer.save_model(f"{output_dir}/{args.model}_last_model.pth")
    
    if args.tensorboard:
        writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--loss", type=str, default="CE", choices=["CE", "BCE"], help="Cross Entropy or Binary Cross Entropy",)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--dataset", type=str, default="custom", choices=["cifar10", "custom"])
    parser.add_argument("--dataset_path", type=str,
        default="/mnt/d/Workspace/Data/datasets/imagenette2",
        help="set the path to the dataset if you are using imagenette dataset",
    )
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--gradient_clipping", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
