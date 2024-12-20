import os

import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

descriptions = {
    'cassette_player': "This is a photo containing a cassette player",
    'chain_saw': "This is a photo containing a chain saw",
    'church': "This is a photo containing a church",
    'english_springer': "This is a photo containing a english springer", }

def run_inference():

    # Construct the full path to the image file using the given 'image_path'
    image_path = "/mnt/d/Workspace/Data/datasets/imagenette2/imagenette2/val/chain_saw/n03000684_440.JPEG"
    image_class = image_path.split("/")[1]
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = torch.cat([clip.tokenize(c) for c in descriptions.values()]).to(device)
    
    with torch.no_grad():
        # Encode image and text
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        
    # Normalize image and text features
    # normalized to have unit length
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores
    # dot product measures how similar the image embedding is to each text embedding.
    # softmax converts it into probabilities
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    print(similarity)
    print(values)
    print(indices)
    # Print the top predictions
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{list(descriptions.keys())[index]}: {100 * value.item():.2f}%")


# Load the CLIP model architecture
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Load the saved state dictionary
model.load_state_dict(torch.load('trained_clip_model_9.pth'))
model.eval()

run_inference()
