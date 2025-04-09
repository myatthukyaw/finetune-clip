import argparse
import os
from pathlib import Path

import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Model Inference")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the fine-tuned CLIP model"
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="Path to a single image or directory of images for inference"
    )
    parser.add_argument(
        "--clip_model", 
        type=str, 
        default="ViT-B/32",
        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
        help="CLIP model architecture"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=3,
        help="Number of top predictions to show"
    )
    parser.add_argument(
        "--classes", 
        type=str, 
        nargs='+',
        help="Classes to use for inference. If not provided, will infer from image directory structure"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="inference-results",
        help="Directory to save visualization results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for inference with multiple images"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run inference on (cuda or cpu)"
    )
    parser.add_argument(
        "--visualization", 
        action="store_true",
        help="Enable visualization of results"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Confidence threshold for predictions"
    )
    return parser.parse_args()


def generate_descriptions(classes):
    """Generate text descriptions for classes."""
    return {cls: f"This is a photo containing a {cls}." for cls in classes}


def get_classes_from_image_path(image_path):
    """Try to infer classes from image directory structure."""
    if os.path.isdir(image_path):
        classes = [d for d in os.listdir(image_path) 
                  if os.path.isdir(os.path.join(image_path, d))]
        if classes:
            print(f"Found classes: {', '.join(classes)}")
            return classes
    
    # Default classes if we can't determine from directory structure
    default_classes = ['cassette_player', 'chain_saw', 'church', 'english_springer',
                      'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball',
                      'parachute', 'tench']
    print(f"Using default classes: {', '.join(default_classes)}")
    return default_classes


def get_images_from_path(image_path):
    """Get list of images from path (file or directory)."""
    image_files = []
    if os.path.isfile(image_path):
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_files.append(image_path)
    else:
        for root, _, files in os.walk(image_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_files.append(os.path.join(root, file))
    
    if not image_files:
        raise ValueError(f"No valid image files found at {image_path}")
    
    return image_files


def load_model(model_path, clip_model, device):
    """Load CLIP model with fine-tuned weights."""
    print(f"\n{'='*70}")
    print(f"Loading CLIP model: {clip_model}")
    print(f"{'='*70}")
    
    # Load the base CLIP model
    model, preprocess = clip.load(clip_model, device=device, jit=False)
    
    # Load fine-tuned weights if provided
    if model_path:
        print(f"Loading fine-tuned weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    return model, preprocess


def predict_batch(model, images, text_tokens, preprocess, device, top_k=3):
    """Run inference on a batch of images."""
    # Preprocess images
    preprocessed_images = torch.stack([preprocess(img) for img in images])
    preprocessed_images = preprocessed_images.to(device)
    
    with torch.no_grad():
        # Get image features
        image_features = model.encode_image(preprocessed_images)
        
        # Normalize image features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Get text features (assumed already normalized)
        text_features = text_tokens
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get top k predictions for each image
        values, indices = similarity.topk(top_k, dim=1)
    
    return values, indices


def visualize_prediction(image, predictions, class_names, output_path):
    """Create a visualization of the prediction results on the image."""
    # Make a copy of the image for drawing
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to get a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw a semi-transparent overlay at the top
    overlay_height = 30 * len(predictions) + 40
    draw.rectangle([(0, 0), (img_draw.width, overlay_height)], 
                  fill=(0, 0, 0, 180))
    
    # Add title
    draw.text((10, 10), "CLIP Prediction Results:", fill=(255, 255, 255), font=font)
    
    # Add each prediction
    for i, (class_idx, conf) in enumerate(predictions):
        # Color based on confidence (red to green)
        color_val = min(255, int(conf * 255 / 100))
        text_color = (255 - color_val, color_val, 0)
        
        # Draw the prediction text
        draw.text(
            (20, 40 + i * 30),
            f"{class_names[class_idx]}: {conf:.2f}%",
            fill=text_color,
            font=font
        )
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_draw.save(output_path)
    
    return img_draw


def run_inference(args):
    """Main inference function."""
    # Set up the device
    device = args.device
    print(f"Using device: {device}")
    
    # Determine classes
    classes = args.classes if args.classes else get_classes_from_image_path(args.image_path)
    descriptions = generate_descriptions(classes)
    
    # Load the model
    model, preprocess = load_model(args.model_path, args.clip_model, device)
    
    # Prepare text tokens
    with torch.no_grad():
        text_tokens = torch.cat([clip.tokenize(desc) for desc in descriptions.values()])
        text_tokens = text_tokens.to(device)
        
        # Pre-compute text features
        text_features = model.encode_text(text_tokens)
        # Normalize text features
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Get image paths
    image_files = get_images_from_path(args.image_path)
    print(f"Found {len(image_files)} images for inference")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images in batches
    results = []
    
    print(f"\n{'='*70}")
    print(f"Running inference on {len(image_files)} images")
    print(f"{'='*70}")
    
    # Process images in batches
    for i in tqdm(range(0, len(image_files), args.batch_size)):
        batch_files = image_files[i:i+args.batch_size]
        batch_images = [Image.open(f).convert("RGB") for f in batch_files]
        
        # Run prediction
        values, indices = predict_batch(
            model, batch_images, text_features, preprocess, device, args.top_k
        )
        
        # Process results
        for j, (image_path, image) in enumerate(zip(batch_files, batch_images)):
            image_values = values[j].cpu().numpy()
            image_indices = indices[j].cpu().numpy()
            
            # Store results for this image
            image_results = list(zip(image_indices, image_values))
            results.append((image_path, image_results))
            
            # Visualize if requested
            if args.visualization:
                output_path = os.path.join(
                    args.output_dir, 
                    f"result_{Path(image_path).stem}.jpg"
                )
                visualize_prediction(image, image_results, classes, output_path)
    
    # Print tabulated results
    print(f"\n{'='*70}")
    print(f"{'INFERENCE RESULTS':^70}")
    print(f"{'='*70}")
    print(f"{'Image':<40} {'Class':<20} {'Confidence':<10}")
    print(f"{'-'*70}")
    
    for image_path, preds in results:
        # Get base filename
        filename = os.path.basename(image_path)
        # Get true class if available from directory structure
        path_parts = Path(image_path).parts
        true_class = "unknown"
        for part in path_parts:
            if part in classes:
                true_class = part
                break
        
        # Print the top prediction with highlight if it matches true class
        top_class_idx, top_conf = preds[0]
        top_class = classes[top_class_idx]
        
        if top_class == true_class:
            print(f"{filename[:38]:<40} {top_class[:18]:<20} \033[1m{top_conf:.2f}%\033[0m ✓")
        else:
            print(f"{filename[:38]:<40} {top_class[:18]:<20} {top_conf:.2f}%")
    
    print(f"\n{'='*70}")
    
    # Calculate and print overall accuracy if ground truth available
    correct = 0
    total_with_truth = 0
    
    for image_path, preds in results:
        path_parts = Path(image_path).parts
        true_class = None
        for part in path_parts:
            if part in classes:
                true_class = part
                break
        
        if true_class is not None:
            total_with_truth += 1
            top_class_idx = preds[0][0]
            if classes[top_class_idx] == true_class:
                correct += 1
    
    if total_with_truth > 0:
        accuracy = 100 * correct / total_with_truth
        print(f"Overall Accuracy: \033[1m{accuracy:.2f}%\033[0m ({correct}/{total_with_truth})")
        print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
