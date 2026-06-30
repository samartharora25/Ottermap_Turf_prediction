"""
Standalone inference CLI for advanced turf/grass segmentation.
Architecture: 4-Channel (RGB+NDVI) ResNet50 U-Net.
"""
import argparse, os, glob, json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape

PATCH = 256

def add_ndvi_channel(img_arr):
    img_float = img_arr.astype(np.float32)
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    ndvi = (G - R) / (G + R + 1e-7)
    ndvi_norm = ((ndvi + 1) / 2.0 * 255.0).astype(np.uint8)
    return np.dstack((img_arr, ndvi_norm))

def build_model(weights_path, device):
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=4, classes=1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_full_image(model, img_arr, device, patch=PATCH, stride=PATCH):
    h, w = img_arr.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1 = min(y + patch, h)
                x1 = min(x + patch, w)
                y0 = max(0, y1 - patch)
                x0 = max(0, x1 - patch)
                
                tile = img_arr[y0:y1, x0:x1]
                
                # THE FIX: Removed the / 255.0 division so it matches training data!
                t = torch.from_numpy(tile.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
                
                out = torch.sigmoid(model(t)).cpu().numpy()[0, 0]
                pred_mask[y0:y1, x0:x1] += out
                count[y0:y1, x0:x1] += 1
                
    count[count == 0] = 1
    return pred_mask / count

def identity_pixel_transform(h, w):
    return Affine(1, 0, 0, 0, -1, h)

def mask_to_geojson(pred_mask, transform, out_path, thresh=0.3, simplify_tol=0.5):
    binary = (pred_mask > thresh).astype("uint8")
    features = []
    for geom, val in shapes(binary, mask=binary.astype(bool), transform=transform):
        if val != 1: continue
        geom_shape = shape(geom).simplify(simplify_tol, preserve_topology=True)
        if geom_shape.is_empty: continue
        features.append({"type": "Feature", "properties": {"class": "turf"},
                         "geometry": geom_shape.__geo_interface__})
    fc = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(fc, f)
    return len(features)

def save_overlay(img_arr, pred_mask, out_path, thresh=0.3):
    overlay = img_arr.copy()
    m = pred_mask > thresh
    overlay[m] = (0.4 * overlay[m] + 0.6 * np.array([255, 0, 0])).astype("uint8")
    Image.fromarray(overlay).save(out_path)

def run_on_image(path, model, device, out_dir, bbox=None):
    img = np.array(Image.open(path).convert("RGB"))
    img_4c = add_ndvi_channel(img)
    
    pred = predict_full_image(model, img_4c, device)
    
    # DEBUG: Print the absolute maximum confidence the model output
    print(f"DEBUG -> Max model confidence before threshold: {pred.max():.4f}")
    
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)

    overlay_path = os.path.join(out_dir, f"{base}_overlay.png")
    save_overlay(img, pred, overlay_path)

    h, w = img.shape[:2]
    transform = identity_pixel_transform(h, w)
    geojson_path = os.path.join(out_dir, f"{base}_pred.geojson")
    
    n_polys = mask_to_geojson(pred, transform, geojson_path)

    print(f"[{base}] Turf coverage: {(pred > 0.3).mean()*100:.2f}% "
          f"-> {overlay_path}, {geojson_path} ({n_polys} polygons)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--weights", type=str, default="weights/best_model.pt")
    ap.add_argument("--out_dir", type=str, default="results/test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading weights from {args.weights} to {device}...")
    model = build_model(args.weights, device)

    if args.image: run_on_image(args.image, model, device, args.out_dir)
    elif args.input:
        for f in sorted(glob.glob(os.path.join(args.input, "*.*"))):
            run_on_image(f, model, device, args.out_dir)
    else: raise SystemExit("Provide --image or --input")

if __name__ == "__main__": main()
