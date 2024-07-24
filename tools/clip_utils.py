import torch
from tqdm import tqdm

def get_text_descriptions(classes):
    descriptions = {}
    for cls in classes:
        descriptions[cls] = f"This is a photo containing a {cls}."
    return descriptions

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def train(train_loader, device, model, optimizer, loss_img, loss_txt, epoch, n_epochs):
    
    total_loss = 0
    with tqdm(total=len(train_loader)) as pbar:
        for batch in train_loader:
            optimizer.zero_grad()
            images, texts, _, _ = batch
            images, texts = images.to(device), texts.to(device)
            
            logits_per_image, logits_per_text = model(images, texts)
            
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2
            
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)
            optimizer.step()
            
            total_loss += float(loss.item())

            pbar.set_description(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
            pbar.update()
    return total_loss

def get_cls_total_and_cls_correct(classes):
    cls_total, cls_correct = {}, {}
    for i, cls in enumerate(classes):
        cls_total[i] = 0
        cls_correct[i] = 0
    return cls_total, cls_correct

def eval(eval_loader, device, model, text_tokens, classes):
    
    correct, total = 0, 0
    cls_total, cls_correct = get_cls_total_and_cls_correct(classes)

    with tqdm(total=len(eval_loader)) as pbar:
        for batch in eval_loader:
            images, _, labels, _ = batch
            images = images.to(device)

            with torch.no_grad():
                # Calculate similarity scores between image and text
                logits_per_image, logits_per_text = model(images, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            indices = probs.argmax(axis=-1)  # Get the indices of the max probability for each image
            preds = indices.tolist()  # Convert to list
            labels = labels.tolist()
            
            for label, pred in zip(labels, preds):
                if label == pred:
                    correct += 1
                    cls_correct[label] += 1
                total += 1
                cls_total[label] += 1
            pbar.set_description(f"Running Evaluation")
            pbar.update()

    eval_acc = correct/total*100
    for cls, corr in cls_correct.items():
        print(f"Eval accuracy on {classes[cls]} : {cls_correct[cls]/cls_total[cls]*100:.2f}%")

    print(f"Total eval accuracy : {eval_acc:.2f}%\n")
    return eval_acc

# def eval_2(eval_loader, device, model, text_tokens):

#     correct, total = 0, 0
#     cls_total = {0 : 0, 1 : 0, 2 : 0, }
#     cls_correct = {0 : 0, 1 : 0, 2 : 0,}

#     with tqdm(total=len(eval_loader)) as pbar:
#         for batch in eval_loader:
#             images, texts, labels, images_paths = batch
#             images = images.to(device)

#             with torch.no_grad():
#                 image_features = model.encode_image(images)
#                 text_features = model.encode_text(text_tokens)

#             # Normalize image and text features, normalized to have unit length
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)

#             similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#             values, indices = similarity.topk(1)
#             preds = [pred[0] for pred in indices.tolist()]
#             labels = labels.tolist()
            
#             for label, pred in zip(labels, preds):
#                 if label == pred:
#                     correct += 1
#                     cls_correct[label] += 1
#                 total += 1
#                 cls_total[label] += 1
#             pbar.set_description(f"Running Evaluation")
#             pbar.update()

#     eval_acc = correct/total
#     for cls, corr in cls_correct.items():
#         print(f"Eval accuracy on class {cls} : {str(cls_correct[cls]/cls_total[cls])}")

#     print(f"Eval accuracy on all is : {eval_acc}\n")
#     return eval_acc