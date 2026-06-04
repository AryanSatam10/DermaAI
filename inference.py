# """
# inference.py
# ============
# All model loading and bias-aware prediction logic for the HAM10000 Streamlit app.
# Imported by app.py — never run directly.

# Integrates all Colab phases:
#   Phase 3  → EfficientNet-B4 backbone
#   Phase 4  → DenseNet-121, ViT-B, MetaMLP ensemble
#   Phase 5  → Temperature scaling + MC Dropout uncertainty
#   Phase 6  → Mahalanobis + Energy OOD detection (OR-gate secured)
#   Phase 8  → Dynamic Grad-CAM Explainability (Visual Trust)
#   Active   → MetaMLP V8 (Human-Corrected Checkpoint)
# """

# import os
# import json
# import time
# import numpy as np
# import torch
# import torch.nn as nn
# import timm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from PIL import Image
# import cv2

# # ──────────────────────────────────────────────────────────────
# # PATHS (all relative to project root)
# # ──────────────────────────────────────────────────────────────
# BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# CKPT_DIR   = os.path.join(BASE_DIR, 'checkpoints')
# EMB_DIR    = os.path.join(BASE_DIR, 'embeddings')
# DATA_DIR   = os.path.join(BASE_DIR, 'data')
# RESULT_DIR = os.path.join(BASE_DIR, 'results')

# CLASS_NAMES  = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# CLASS_LABELS = {
#     'nv'   : 'Melanocytic Nevi',
#     'mel'  : 'Melanoma',
#     'bkl'  : 'Benign Keratosis',
#     'bcc'  : 'Basal Cell Carcinoma',
#     'akiec': 'Actinic Keratosis',
#     'vasc' : 'Vascular Lesion',
#     'df'   : 'Dermatofibroma',
# }
# NUM_CLASSES  = 7
# IMG_SIZE     = 224
# MEAN         = [0.7630, 0.5456, 0.5700]
# STD          = [0.1409, 0.1520, 0.1692]

# # ──────────────────────────────────────────────────────────────
# # DEVICE (CPU for local; MPS for Apple Silicon; CUDA for Nvidia)
# # ──────────────────────────────────────────────────────────────
# def get_device():
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#         return torch.device('mps')
#     return torch.device('cpu')

# DEVICE = get_device()

# # ──────────────────────────────────────────────────────────────
# # SAFE CHECKPOINT LOADER (PyTorch 2.6 compatible)
# # ──────────────────────────────────────────────────────────────
# def _register_safe_globals():
#     try:
#         import numpy._core.multiarray
#         torch.serialization.add_safe_globals([
#             numpy._core.multiarray.scalar,
#             np.dtype, np.float64, np.float32, np.int64, np.int32,
#         ])
#     except Exception:
#         pass

# _register_safe_globals()

# def safe_load(path: str) -> dict:
#     try:
#         return torch.load(path, map_location=DEVICE, weights_only=True)
#     except Exception:
#         return torch.load(path, map_location=DEVICE, weights_only=False)

# # ──────────────────────────────────────────────────────────────
# # MetaMLP (must match Phase 4/8 architecture exactly)
# # ──────────────────────────────────────────────────────────────
# class MetaMLP(nn.Module):
#     def __init__(self, in_dim: int = 3584, num_cls: int = 7):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
#             nn.Linear(512, 128),    nn.BatchNorm1d(128),  nn.GELU(), nn.Dropout(0.3),
#             nn.Linear(128, num_cls),
#         )

#     def forward(self, x):
#         return self.net(x)

# # ──────────────────────────────────────────────────────────────
# # PREPROCESSING (hair removal + resize + normalise)
# # ──────────────────────────────────────────────────────────────
# def _remove_hair(img_bgr: np.ndarray) -> np.ndarray:
#     """DullRazor morphological hair removal."""
#     gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
#     blackhat= cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
#     _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
#     return cv2.inpaint(img_bgr, mask, 6, cv2.INPAINT_TELEA)

# _val_transform = A.Compose([
#     A.Resize(IMG_SIZE, IMG_SIZE),
#     A.Normalize(mean=MEAN, std=STD),
#     ToTensorV2(),
# ])

# def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
#     """PIL Image → preprocessed tensor [1, 3, 224, 224]."""
#     img_bgr   = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
#     img_clean = _remove_hair(img_bgr)
#     img_rgb   = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
#     tensor    = _val_transform(image=img_rgb)['image']
#     return tensor.unsqueeze(0)  # [1, 3, 224, 224]

# # ──────────────────────────────────────────────────────────────
# # MODEL LOADER & PIPELINE
# # ──────────────────────────────────────────────────────────────
# class HAM10000Pipeline:
#     def __init__(self, t_passes: int = 50):
#         self.t_passes = t_passes
#         self._load_configs()
#         self._load_backbones()
#         self._load_ensemble()
#         self._load_ood_params()
#         print(f"✓ Pipeline ready on {DEVICE}")

#     def _load_configs(self):
#         self.label_map  = json.load(open(os.path.join(DATA_DIR, 'label_map.json')))
#         thresholds      = json.load(open(os.path.join(RESULT_DIR, 'uncertainty_thresholds.json')))
#         ood_cfg         = json.load(open(os.path.join(RESULT_DIR, 'ood_detectors.json')))
#         temp_data       = safe_load(os.path.join(CKPT_DIR, 'temperature_scalar.pt'))

#         self.T_OPT      = float(temp_data['temperature'])
#         self.U_HIGH     = thresholds['U_high_var']
#         self.U_LOW      = thresholds['U_low_var']
#         self.MAHA_THR   = ood_cfg['mahalanobis_threshold']
#         self.ENERGY_THR = ood_cfg['energy_threshold']

#     def _load_backbones(self):
#         def _load(name, ckpt_file):
#             m     = timm.create_model(name, pretrained=False, num_classes=0).to(DEVICE)
#             state = {k: v for k, v in safe_load(os.path.join(CKPT_DIR, ckpt_file))['model_state'].items()
#                      if not k.startswith(('classifier', 'head'))}
#             m.load_state_dict(state, strict=False)
#             m.eval()
#             return m

#         self.eff_model   = _load('efficientnet_b4',      'effb4_phase3_best.pt')
#         self.dense_model = _load('densenet121',           'densenet121_phase4_best.pt')
#         self.vit_model   = _load('vit_base_patch16_224', 'vit_phase4_best.pt')

#         dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
#         with torch.no_grad():
#             e = self.eff_model(dummy).shape[1]
#             d = self.dense_model(dummy).shape[1]
#             v = self.vit_model(dummy).shape[1]
#         self.emb_dim = e + d + v

#     def _load_ensemble(self):
#         ckpt_path = os.path.join(CKPT_DIR, 'ensemble_meta_v10.pt')
        
#         # ──────────────────────────────────────────────────────────
#         # MLOps DIAGNOSTIC CHECK (Proves which file is actually loading)
#         # ──────────────────────────────────────────────────────────
#         if os.path.exists(ckpt_path):
#             mod_time = time.ctime(os.path.getmtime(ckpt_path))
#             print("\n" + "="*60)
#             print("🚨 CHECKPOINT VERIFICATION 🚨")
#             print(f"Target Path: {ckpt_path}")
#             print(f"File Last Modified: {mod_time}")
#             print("If this is NOT today's date/time, you have the wrong file!")
#             print("="*60 + "\n")
#         else:
#             print(f"❌ ERROR: Cannot find the checkpoint at {ckpt_path}")

#         self.meta = MetaMLP(in_dim=self.emb_dim).to(DEVICE)
#         self.meta.load_state_dict(safe_load(ckpt_path)['model_state'])
#         self.meta.eval()

#     def _load_ood_params(self):
#         self.maha_means  = np.load(os.path.join(EMB_DIR, 'mahalanobis_class_means.npy'))
#         self.maha_covinv = np.load(os.path.join(EMB_DIR, 'mahalanobis_cov_inv.npy'))

#     def _extract_embedding(self, tensor: torch.Tensor) -> np.ndarray:
#         tensor = tensor.to(DEVICE)
#         with torch.no_grad():
#             e = self.eff_model(tensor)
#             d = self.dense_model(tensor)
#             v = self.vit_model(tensor)
#         return torch.cat([e, d, v], dim=1).cpu().numpy()[0]   

#     def _mahalanobis(self, emb: np.ndarray) -> float:
#         dists = [float((emb - mu) @ self.maha_covinv @ (emb - mu))
#                  for mu in self.maha_means]
#         return float(min(dists))

#     def _energy(self, emb: np.ndarray) -> float:
#         x = torch.FloatTensor(emb[None, :]).to(DEVICE)
#         self.meta.eval()
#         with torch.no_grad():
#             logits = self.meta(x)
#         return float(-torch.logsumexp(logits, dim=1).item())

#     def _enable_mc(self):
#         self.meta.eval()
#         for m in self.meta.modules():
#             if isinstance(m, nn.Dropout):
#                 m.train()

#     def _mc_inference(self, emb: np.ndarray):
#         x = torch.FloatTensor(emb[None, :]).to(DEVICE)
#         self._enable_mc()
#         passes = []
#         with torch.no_grad():
#             for _ in range(self.t_passes):
#                 logits = self.meta(x)
#                 probs  = torch.softmax(logits / self.T_OPT, dim=-1)
#                 passes.append(probs.cpu().numpy()[0])
#         mc_arr   = np.stack(passes)                           
#         mean_p   = mc_arr.mean(axis=0)                       
#         variance = float(mc_arr.var(axis=0).sum())
#         entropy  = float(-(mean_p * np.log(mean_p + 1e-9)).sum())
#         return mean_p, variance, entropy

#     # ──────────────────────────────────────────────────────────────
#     # GRAD-CAM EXPLAINABILITY LOGIC (Final Ensemble Correction)
#     # ──────────────────────────────────────────────────────────────
#     def generate_gradcam(self, tensor: torch.Tensor, target_class: int) -> np.ndarray:
#         """Dynamically traces gradients from the MetaMLP back to the DenseNet backbone."""
#         tensor = tensor.to(DEVICE).requires_grad_(True)
        
#         # Turn OFF Dropout so gradients aren't shredded during backprop!
#         self.meta.eval()
        
#         features = []
#         gradients = []
        
#         def fw_hook(module, input, output):
#             features.append(output)
#             output.register_hook(lambda grad: gradients.append(grad))
            
# # Hook DenseNet instead of EfficientNet for a stronger spatial signal
#         if not hasattr(self.dense_model, 'features'):
#             return None
            
#         h = self.dense_model.features.register_forward_hook(fw_hook)
        
#         try:
#             self.eff_model.zero_grad()
#             self.dense_model.zero_grad()
#             self.vit_model.zero_grad()
#             self.meta.zero_grad()
            
#             # Funnel all gradients strictly into DenseNet
#             with torch.no_grad():
#                 e = self.eff_model(tensor)
#                 v = self.vit_model(tensor)
                
#             with torch.set_grad_enabled(True):
#                 d = self.dense_model(tensor)
                
#             emb = torch.cat([e, d, v], dim=1)
            
#             with torch.set_grad_enabled(True):
#                 logits = self.meta(emb)
#                 score = logits[0, target_class]
                
#             score.backward()
            
#         finally:
#             # GUARANTEED CLEANUP: This runs even if score.backward() fails
#             h.remove()
        
#         if not features or not gradients:
#             return None
            
#         fmap = features[0].squeeze().cpu().detach().numpy()
#         grad = gradients[0].squeeze().cpu().detach().numpy()
        
#         weights = np.mean(grad, axis=(1, 2))
        
#         cam = np.zeros(fmap.shape[1:], dtype=np.float32)
#         for i, w in enumerate(weights):
#             cam += w * fmap[i]
            
#         cam = np.maximum(cam, 0) # ReLU
        
#         cam_min = np.min(cam)
#         cam_max = np.max(cam)
#         if cam_max - cam_min > 1e-8:
#             cam = (cam - cam_min) / (cam_max - cam_min)
#         else:
#             return None 
            
#         cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        
#         cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#         cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        
#         return cam_heatmap

#     def _overlay_heatmap(self, pil_img: Image.Image, heatmap_np: np.ndarray) -> Image.Image:
#         """Mixes the generated heatmap with the original image."""
#         img_np = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
#         overlay = cv2.addWeighted(img_np, 0.5, heatmap_np, 0.7, 0)
#         return Image.fromarray(overlay)

#     # ──────────────────────────────────────────────────────────────
#     # CORE PREDICTION ENGINE
#     # ──────────────────────────────────────────────────────────────
#     def predict(self, pil_img: Image.Image) -> dict:
#         tensor = preprocess_image(pil_img)
#         emb    = self._extract_embedding(tensor)

#         # OOD scores 
#         ood_maha   = self._mahalanobis(emb)
#         ood_energy = self._energy(emb)
        
#         # CORRECTED LOGIC: OR-gate for clinical safety 
#         is_ood     = (ood_maha > self.MAHA_THR) or (ood_energy < self.ENERGY_THR)

#         # MC Dropout uncertainty
#         mean_p, variance, entropy = self._mc_inference(emb)

#         pred_idx   = int(mean_p.argmax())
#         confidence = float(mean_p.max())

#         if is_ood:
#             decision = 'REJECT'
#         elif variance > self.U_HIGH:
#             decision = 'FLAG'
#         elif variance <= self.U_LOW:
#             decision = 'ACCEPT'
#         else:
#             decision = 'FLAG'

#         result = {
#             'class'       : CLASS_NAMES[pred_idx],
#             'class_full'  : CLASS_LABELS[CLASS_NAMES[pred_idx]],
#             'confidence'  : round(confidence, 4),
#             'uncertainty' : round(variance, 6),
#             'entropy'     : round(entropy, 4),
#             'ood_maha'    : round(ood_maha, 2),
#             'ood_energy'  : round(ood_energy, 4),
#             'is_ood'      : bool(is_ood),
#             'decision'    : decision,
#             'all_probs'   : {CLASS_NAMES[i]: round(float(mean_p[i]), 4)
#                              for i in range(NUM_CLASSES)},
#         }

#         # Generate Explainability Heatmap if it's not a rejected Out-of-Distribution image
#         if decision in ['ACCEPT', 'FLAG']:
#             try:
#                 heatmap_np = self.generate_gradcam(tensor, pred_idx)
#                 if heatmap_np is not None:
#                     result['gradcam_img'] = self._overlay_heatmap(pil_img, heatmap_np)
#             except Exception as e:
#                 # Silently catch CAM failure to ensure prediction still routes to UI
#                 print(f"Grad-CAM generation failed: {e}")
#                 pass 

#         return result
    



"""
inference.py
============
All model loading and bias-aware prediction logic for the HAM10000 Streamlit app.
Imported by app.py — never run directly.

Integrates all Colab phases:
  Phase 3  → EfficientNet-B4 backbone
  Phase 4  → DenseNet-121, ViT-B, MetaMLP ensemble
  Phase 5  → Temperature scaling + MC Dropout uncertainty
  Phase 6  → Mahalanobis + Energy OOD detection (OR-gate secured)
  Phase 8  → Spatial Feature Activation Mapping (Visual Trust)
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2

# ──────────────────────────────────────────────────────────────
# PATHS (all relative to project root)
# ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR   = os.path.join(BASE_DIR, 'checkpoints')
EMB_DIR    = os.path.join(BASE_DIR, 'embeddings')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

CLASS_NAMES  = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_LABELS = {
    'nv'   : 'Melanocytic Nevi',
    'mel'  : 'Melanoma',
    'bkl'  : 'Benign Keratosis',
    'bcc'  : 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc' : 'Vascular Lesion',
    'df'   : 'Dermatofibroma',
}
NUM_CLASSES  = 7
IMG_SIZE     = 224
MEAN         = [0.7630, 0.5456, 0.5700]
STD          = [0.1409, 0.1520, 0.1692]

# ──────────────────────────────────────────────────────────────
# DEVICE (CPU for local; MPS for Apple Silicon; CUDA for Nvidia)
# ──────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()

# ──────────────────────────────────────────────────────────────
# SAFE CHECKPOINT LOADER (PyTorch 2.6 compatible)
# ──────────────────────────────────────────────────────────────
def _register_safe_globals():
    try:
        import numpy._core.multiarray
        torch.serialization.add_safe_globals([
            numpy._core.multiarray.scalar,
            np.dtype, np.float64, np.float32, np.int64, np.int32,
        ])
    except Exception:
        pass

_register_safe_globals()

def safe_load(path: str) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except Exception:
        return torch.load(path, map_location=DEVICE, weights_only=False)

# ──────────────────────────────────────────────────────────────
# MetaMLP 
# ──────────────────────────────────────────────────────────────
class MetaMLP(nn.Module):
    def __init__(self, in_dim: int = 3584, num_cls: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 128),    nn.BatchNorm1d(128),  nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_cls),
        )

    def forward(self, x):
        return self.net(x)

# ──────────────────────────────────────────────────────────────
# PREPROCESSING (hair removal + resize + normalise)
# ──────────────────────────────────────────────────────────────
def _remove_hair(img_bgr: np.ndarray) -> np.ndarray:
    """DullRazor morphological hair removal."""
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat= cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, 6, cv2.INPAINT_TELEA)

_val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """PIL Image → preprocessed tensor [1, 3, 224, 224]."""
    img_bgr   = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    img_clean = _remove_hair(img_bgr)
    img_rgb   = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
    tensor    = _val_transform(image=img_rgb)['image']
    return tensor.unsqueeze(0)

# ──────────────────────────────────────────────────────────────
# MODEL LOADER & PIPELINE
# ──────────────────────────────────────────────────────────────
class HAM10000Pipeline:
    def __init__(self, t_passes: int = 50):
        self.t_passes = t_passes
        self._load_configs()
        self._load_backbones()
        self._load_ensemble()
        self._load_ood_params()
        print(f"✓ Pipeline ready on {DEVICE}")

    def _load_configs(self):
        self.label_map  = json.load(open(os.path.join(DATA_DIR, 'label_map.json')))
        thresholds      = json.load(open(os.path.join(RESULT_DIR, 'uncertainty_thresholds.json')))
        ood_cfg         = json.load(open(os.path.join(RESULT_DIR, 'ood_detectors.json')))
        temp_data       = safe_load(os.path.join(CKPT_DIR, 'temperature_scalar.pt'))

        self.T_OPT      = float(temp_data['temperature'])
        self.U_HIGH     = thresholds['U_high_var']
        self.U_LOW      = thresholds['U_low_var']
        self.MAHA_THR   = ood_cfg['mahalanobis_threshold']
        self.ENERGY_THR = ood_cfg['energy_threshold']

    def _load_backbones(self):
        def _load(name, ckpt_file):
            m     = timm.create_model(name, pretrained=False, num_classes=0).to(DEVICE)
            state = {k: v for k, v in safe_load(os.path.join(CKPT_DIR, ckpt_file))['model_state'].items()
                     if not k.startswith(('classifier', 'head'))}
            m.load_state_dict(state, strict=False)
            m.eval()
            return m

        self.eff_model   = _load('efficientnet_b4',      'effb4_phase3_best.pt')
        self.dense_model = _load('densenet121',           'densenet121_phase4_best.pt')
        self.vit_model   = _load('vit_base_patch16_224', 'vit_phase4_best.pt')

        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        with torch.no_grad():
            e = self.eff_model(dummy).shape[1]
            d = self.dense_model(dummy).shape[1]
            v = self.vit_model(dummy).shape[1]
        self.emb_dim = e + d + v

    def _load_ensemble(self):
        ckpt_path = os.path.join(CKPT_DIR, 'ensemble_meta_v10.pt')
        
        if not os.path.exists(ckpt_path):
            print(f"❌ ERROR: Cannot find the checkpoint at {ckpt_path}")

        self.meta = MetaMLP(in_dim=self.emb_dim).to(DEVICE)
        self.meta.load_state_dict(safe_load(ckpt_path)['model_state'])
        self.meta.eval()

    def _load_ood_params(self):
        self.maha_means  = np.load(os.path.join(EMB_DIR, 'mahalanobis_class_means.npy'))
        self.maha_covinv = np.load(os.path.join(EMB_DIR, 'mahalanobis_cov_inv.npy'))

    def _extract_embedding(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.to(DEVICE)
        with torch.no_grad():
            e = self.eff_model(tensor)
            d = self.dense_model(tensor)
            v = self.vit_model(tensor)
        return torch.cat([e, d, v], dim=1).cpu().numpy()[0]   

    def _mahalanobis(self, emb: np.ndarray) -> float:
        dists = [float((emb - mu) @ self.maha_covinv @ (emb - mu))
                 for mu in self.maha_means]
        return float(min(dists))

    def _energy(self, emb: np.ndarray) -> float:
        x = torch.FloatTensor(emb[None, :]).to(DEVICE)
        self.meta.eval()
        with torch.no_grad():
            logits = self.meta(x)
        return float(-torch.logsumexp(logits, dim=1).item())

    def _enable_mc(self):
        self.meta.eval()
        for m in self.meta.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _mc_inference(self, emb: np.ndarray):
        x = torch.FloatTensor(emb[None, :]).to(DEVICE)
        self._enable_mc()
        passes = []
        with torch.no_grad():
            for _ in range(self.t_passes):
                logits = self.meta(x)
                probs  = torch.softmax(logits / self.T_OPT, dim=-1)
                passes.append(probs.cpu().numpy()[0])
        mc_arr   = np.stack(passes)                           
        mean_p   = mc_arr.mean(axis=0)                       
        variance = float(mc_arr.var(axis=0).sum())
        entropy  = float(-(mean_p * np.log(mean_p + 1e-9)).sum())
        return mean_p, variance, entropy

    # ──────────────────────────────────────────────────────────────
    # SPATIAL FEATURE ACTIVATION MAPPING (Replaces Grad-CAM)
    # ──────────────────────────────────────────────────────────────
    def generate_gradcam(self, tensor: torch.Tensor, target_class: int, use_specialist: bool = False) -> np.ndarray:
        """
        Bypasses noisy MetaMLP ensemble gradients.
        Extracts raw Spatial Feature Activation Maps directly from DenseNet.
        """
        self.dense_model.eval()
        features = []
        
        def fw_hook(module, input, output):
            features.append(output)
            
        # Hook the final spatial layer of DenseNet
        if hasattr(self.dense_model, 'features'):
            h = self.dense_model.features.register_forward_hook(fw_hook)
        else:
            return None
            
        try:
            # Forward pass ONLY (no .backward() gradient starvation)
            with torch.no_grad():
                _ = self.dense_model(tensor.to(DEVICE))
        finally:
            h.remove()
            
        if not features:
            return None
            
        # Extract the 2D spatial feature maps [Channels, Height, Width]
        fmap = features[0].squeeze().cpu().numpy()
        
        # 🚨 THE FIX: Use Max Pooling across all filter channels to find the most dominant structural activations (lesions)
        cam = np.max(fmap, axis=0)
        
        # Normalize strictly to 0-1 range
        cam = np.maximum(cam, 0)
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            return None
            
        # Smooth upscale using INTER_CUBIC to eliminate blocky artifacts
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        
        # Apply Heatmap palette
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        
        return cam_heatmap

    def _overlay_heatmap(self, pil_img: Image.Image, heatmap_np: np.ndarray) -> Image.Image:
        """Mixes the generated heatmap with the original image."""
        img_np = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
        overlay = cv2.addWeighted(img_np, 0.5, heatmap_np, 0.7, 0)
        return Image.fromarray(overlay)

    # ──────────────────────────────────────────────────────────────
    # CORE PREDICTION ENGINE
    # ──────────────────────────────────────────────────────────────
    def predict(self, pil_img: Image.Image) -> dict:
        tensor = preprocess_image(pil_img)
        emb    = self._extract_embedding(tensor)

        # OOD scores 
        ood_maha   = self._mahalanobis(emb)
        ood_energy = self._energy(emb)
        
        is_ood     = (ood_maha > self.MAHA_THR) or (ood_energy < self.ENERGY_THR)

        # MC Dropout uncertainty
        mean_p, variance, entropy = self._mc_inference(emb)

        pred_idx   = int(mean_p.argmax())
        confidence = float(mean_p.max())

        if is_ood:
            decision = 'REJECT'
        elif variance > self.U_HIGH:
            decision = 'FLAG'
        elif variance <= self.U_LOW:
            decision = 'ACCEPT'
        else:
            decision = 'FLAG'

        result = {
            'class'       : CLASS_NAMES[pred_idx],
            'class_full'  : CLASS_LABELS[CLASS_NAMES[pred_idx]],
            'confidence'  : round(confidence, 4),
            'uncertainty' : round(variance, 6),
            'entropy'     : round(entropy, 4),
            'ood_maha'    : round(ood_maha, 2),
            'ood_energy'  : round(ood_energy, 4),
            'is_ood'      : bool(is_ood),
            'decision'    : decision,
            'all_probs'   : {CLASS_NAMES[i]: round(float(mean_p[i]), 4)
                             for i in range(NUM_CLASSES)},
        }

        # Generate Explainability Heatmap 
        if decision in ['ACCEPT', 'FLAG']:
            try:
                heatmap_np = self.generate_gradcam(tensor, pred_idx)
                if heatmap_np is not None:
                    result['gradcam_img'] = self._overlay_heatmap(pil_img, heatmap_np)
            except Exception as e:
                pass 

        return result