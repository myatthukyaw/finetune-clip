import os
import clip
import torch
import argparse
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from tools.file import get_output_dir
from tools.dataset import ClipImageNetteDataset
from tools.clip_utils import train, eval, get_text_descriptions

def main(args):

    writer = SummaryWriter(f"runs/{args.dataset}-{args.model.replace('/','-')}-finetune-exp")

    model, preprocess = clip.load(args.model, device = args.device, jit=False)

    classes = os.listdir(os.path.join(args.dataset_path, 'train'))
    descriptions = get_text_descriptions(classes)
    text_tokens = torch.cat([clip.tokenize(c) for c in descriptions.values()]).to(args.device)

    train_image_path = os.path.join(args.dataset_path, "train")
    eval_image_path = os.path.join(args.dataset_path, "val")

    train_dataset = ClipImageNetteDataset(train_image_path, classes, descriptions, preprocess)
    eval_dataset = ClipImageNetteDataset(eval_image_path, classes, descriptions, preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False) 

    best_acc = 0.0
    output_dir = get_output_dir(args)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-7, 
                                 betas=(0.9, 0.98),
                                 eps=1e-6, 
                                 weight_decay=0.2) 


    for epoch in range(args.epochs):

        model.train()
        train_loss = train(train_dataloader, args.device, model, optimizer, loss_img, loss_txt, epoch, args.epochs)

        model.eval()
        eval_acc = eval(eval_dataloader, args.device, model, text_tokens, classes)

        writer.add_scalar('Train loss', train_loss, epoch+1)
        writer.add_scalar('Eval accuracy', eval_acc, epoch+1)

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), f"{output_dir}/{args.model.replace('/','-')}_best_model.pth")
    
    torch.save(model.state_dict(), f"{output_dir}/{args.model.replace('/','-')}_last_model.pth")
    writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B/32", choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="imagenette", choices=["imagenette"])
    parser.add_argument("--dataset_path", type=str, default="/mnt/d/Workspace/Data/datasets/imagenette2/imagenette2", 
                                                    help="set the path to the dataset if you are using imagenette dataset")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()
    main(args)