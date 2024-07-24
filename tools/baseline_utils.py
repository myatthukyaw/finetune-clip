import torch
from tqdm import tqdm

def train(train_loader, device, model, loss_fun, optimizer, epoch, n_epochs):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_loader)) as pbar:
        for i, (images, labels)  in enumerate(train_loader):

            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)

            loss = loss_fun(preds, labels)

            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.item())
                        
            pbar.set_description(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
            pbar.update()
    return total_loss

def eval(test_loader, device, model, loss_fun):

    model.eval()
    total_eval_loss = 0
    correct = 0
    total = 0

    with tqdm(total=len(test_loader)) as pbar:
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):

                images = images.to(device)
                labels = labels.to(device)
                
                preds = model(images)
                loss = loss_fun(preds, labels)
                _, predicted = torch.max(preds.data, 1)
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()
                
                total_eval_loss += float(loss.item())
    
                pbar.set_description(f"Evaluation Loss: {loss.item():.2f}")
                pbar.update()
    acc = 100 * correct / total

    return total_eval_loss, acc