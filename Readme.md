## Assignment 2 (Computer Vision) — Problem Statement 4  
### Fine-Tuning Stable Diffusion v1.5 with LoRA for Readable Text Images (OCR Evaluation)

### Submission header (fill this in)
- **Course / Subject**: \<Course name\>  
- **Assignment**: Assignment 2 — Problem Statement 4  
- **Topic**: Fine-tuning a text-to-image model (Stable Diffusion v1.5) using LoRA  
- **Group**: Group 13  
- **Submitted by**: \<Name (ID)\>, \<Name (ID)\>, ...  
- **Professor / Instructor**: \<Name\>  
- **Institute**: \<College/University\>  
- **Date**: \<YYYY-MM-DD\>  

---

### TL;DR (what this project does)
- **Goal**: Adapt a pre-trained text-to-image model so it can generate **simple, readable text images** (like signs/labels) from text prompts.  
- **Model**: `runwayml/stable-diffusion-v1-5` fine-tuned using **LoRA** (train only small attention adapters; base model stays frozen).  
- **Data**: A small synthetic dataset that simulates “crowd-sourced operational text images.”  
- **Evaluation**: Generate images → run **Tesseract OCR** → compute **Exact Match Rate** + **Character Accuracy**.

### Key results (from the notebook run)
- **Validation size**: 10 prompts  
- **Exact Match Rate**: **0.00%**  
- **Average Character Accuracy**: **6.58%** (min **0.00%**, max **31.82%**)  
- **Trainable parameters (LoRA only)**: **797,184** (≈ **0.0927%** of full UNet parameters)
- **Training epochs**: 20  
- **Best training loss**: **0.0183** (saved as best LoRA adapter)  
- **Final training loss (last epoch)**: **0.0416**  

> Note: OCR metrics depend strongly on image quality (font size/contrast). The notebook also includes **qualitative visual comparison** (“before vs after fine-tuning”) showing improved text-like structure/placement.

### Professor quick-check (30 seconds)
If you are grading quickly, these files prove the full pipeline ran end-to-end:
- **LoRA weights exist**: `./lora_weights/best_model/`  
- **Training curve image**: `./outputs/training_curves.png` (loss decreasing)  
- **Evaluation CSV**: `./outputs/evaluation_results.csv` (exact match + character accuracy per sample)  
- **Visual proof**:
  - `./outputs/generated_images_with_ocr.png` (generated images + OCR text)
  - `./outputs/before_after_comparison.png` (base vs fine-tuned comparison)
- **One-page summary (text)**: `./outputs/summary_report.txt`

---

## Table of Contents
- [0. Rubric Alignment (15 pts)](#0-rubric-alignment-15-pts)
- [1. Introduction](#1-introduction)
- [2. Environment Setup](#2-environment-setup)
- [3. Data Preprocessing & Augmentation](#3-data-preprocessing--augmentation)
- [4. Model Development (Stable Diffusion + LoRA)](#4-model-development-stable-diffusion--lora)
- [5. Training](#5-training)
- [6. Evaluation (OCR-based)](#6-evaluation-ocr-based)
- [7. Results & Analysis](#7-results--analysis)
- [8. Deliverables (What gets saved)](#8-deliverables-what-gets-saved)
- [9. How to Run / Reproduce](#9-how-to-run--reproduce)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Limitations & Future Work](#11-limitations--future-work)
- [12. References](#12-references)
- [13. Viva / Exam Q&A (Common Questions)](#13-viva--exam-qa-common-questions)
- [Appendix A. Notebook Cell-by-Cell Walkthrough](#appendix-a-notebook-cell-by-cell-walkthrough)

---

## 0. Rubric Alignment (15 pts)
This section is written to directly match the **college marking rubric** (Total: **15 pts**).  
For each criterion, I list **what we did**, **why it is relevant**, and **what we can improve**.

### 0.1 Data Preprocessing (2.5 pts)
**Rubric expects**: preprocessing such as **normalization**, **resizing**, etc. (rubric mentions “semantic segmentation” as an example).

**What we did (in this notebook):**
- **Resize + crop** images to a fixed model resolution (**512×512**) using PyTorch transforms.
- **Normalize** pixel values to the range required by Stable Diffusion training (**[-1, 1]** via `Normalize([0.5],[0.5])`).
- **Tokenize prompts** using CLIP tokenizer (padding/truncation).
- **OCR preprocessing** (evaluation stage):
  - grayscale → contrast increase → Otsu thresholding (this is effectively a binary “segmentation” for OCR).

**Why it matches the task:**
- Diffusion training requires consistent image sizes and normalized tensors.
- OCR benefits strongly from thresholding/contrast to separate characters from background.

**Improvements (what we can do next):**
- Add a dedicated **text-region detector/segmentation** (CRAFT/EAST) to crop text before OCR and compute metrics on the text region.
- Add more realistic augmentations: blur, rotation, perspective warp, JPEG artifacts, uneven lighting.

### 0.2 Model Development (5 pts)
**Rubric expects**: implement the model and integrate techniques for improved performance (examples: contextual awareness, multi-task learning).

**What we did (in this notebook):**
- Implemented Stable Diffusion v1.5 fine-tuning pipeline:
  - CLIP Text Encoder + UNet + VAE + DDPM scheduler.
- Integrated **LoRA** (parameter-efficient fine-tuning):
  - trains only attention projections: `to_q`, `to_k`, `to_v`, `to_out.0`
  - freezes base weights (VAE, text encoder, base UNet).
- Used **mixed precision (fp16)** and **gradient accumulation** to fit Colab GPU memory.
- Used prompts with **contextual style**:  
  `a <style> image with the text '<TEXT>' in clear, readable font`

**Why it matches the task:**
- Text rendering is a domain shift. LoRA adapts the model efficiently with limited compute.
- Context words (“sign/label/badge”) guide the model to the correct domain.

**Improvements (what we can do next):**
- Train with more data (and ideally some real crowd-sourced samples).
- Increase LoRA rank / apply LoRA to additional blocks or also the text encoder.
- Add an auxiliary **multi-task** signal (OCR-guided loss) to enforce spelling correctness.

### 0.3 Evaluation Metrics (2.5 pts)
**Rubric expects**: use appropriate metrics and justify them.

**What we did (in this notebook):**
- **Exact Match Rate** (strict readability)
- **Character Accuracy** (soft readability)
- Qualitative evaluation:
  - generated samples with OCR text
  - base vs fine-tuned comparison

**Why these metrics are relevant:**
- The assignment target is “readable text.” OCR is a practical automatic proxy for readability.

**Improvements (what we can do next):**
- Add CER/WER (edit-distance-based metrics).
- Track inference **speed** (seconds per image) and memory usage.
- Add CLIPScore / human study rating for perceptual quality.

### 0.4 Justification (2.5 pts)
**Rubric expects**: explain success/poor performance (underfitting/overfitting, model choices).

**What we did (in this notebook + expanded in Sections 7 and 11):**
- Explained why OCR is low:
  - tiny dataset (50 samples),
  - font/contrast dependence,
  - diffusion text difficulty (“artistic” text-like strokes).
- Justified LoRA choice for limited compute.

**Improvements (what we can do next):**
- Run ablations: LoRA rank/alpha, LR, epochs, guidance scale, inference steps.
- Compare with/without certain augmentations (noise, borders, fonts).

### 0.5 Documentation / Presentation / Code Quality (2.5 pts)
**What we did:**
- Notebook is structured (Setup → Data → Model → Training → Evaluation → Results).
- This README provides:
  - beginner explanations,
  - config and hyperparameters,
  - saved outputs/deliverables,
  - reproducibility instructions.

---

## 1. Introduction

### 1.1 Problem statement (Problem Statement 4)
Fine-tune a **pre-trained text-to-image model (Stable Diffusion v1.5)** using **LoRA** to generate **simple, readable text images** from **crowd-sourced text–image pairs**.

### 1.2 Objectives (why we do this)
- **Domain shift understanding**: generative models trained on broad internet images often struggle with *precise* text rendering.
- **Model adaptation with limited/noisy samples**: show how LoRA can adapt a large model using a small dataset.
- **Practical constraints**: keep training feasible on Colab GPU (memory/runtime limits).

### 1.3 Background for beginners (quick explanations)
- **Stable Diffusion**: a *diffusion* model that learns to remove noise step-by-step in a compressed “latent” space (via a VAE).  
- **LoRA (Low-Rank Adaptation)**: instead of updating all weights, we insert tiny low-rank matrices into selected layers (here: attention projections).  
  - This makes fine-tuning **fast**, **memory-efficient**, and practical with small GPUs.
- **Why OCR evaluation?** If the model truly renders the requested text, OCR should be able to read it. We use this as an automatic readability proxy.

### 1.4 What we did vs. what we did not do (important for exam clarity)
**What we did (this notebook proves):**
- Generated a dataset of text-image pairs and stored it with metadata.
- Fine-tuned Stable Diffusion v1.5 using LoRA adapters (UNet attention layers).
- Evaluated generated images using OCR-based metrics + visual comparisons.
- Saved all results to disk (`./data`, `./lora_weights`, `./outputs`) so grading is easy.

**What we did not do (and why):**
- We did **not** train Stable Diffusion from scratch (too large; not required for the assignment).
- We did **not** use real crowd-sourced images because a real dataset was not provided; instead we **simulate** “crowd-sourced operational text images” with synthetic generation.
- We did **not** apply semantic segmentation as in classic CV segmentation problems.  
  Our task is **generative text rendering**; however, OCR preprocessing includes threshold-based text/background separation.

### 1.5 Why text rendering is hard for diffusion models (short explanation)
Text needs **exact discrete structure** (characters, spacing, spelling).  
Diffusion models optimize for realistic images, so they may create “text-like strokes” that look plausible to humans but fail OCR.  
That is why we include OCR metrics and discuss limitations + improvements.

---

## 2. Environment Setup

### 2.1 Hardware/GPU check (from the notebook)
The notebook checks CUDA availability and prints:
- **Python**: 3.12.12  
- **PyTorch**: 2.10.0 + CUDA 12.8  
- **GPU**: Tesla T4 (~15.6 GB)

### 2.2 Libraries installed in the notebook
The notebook installs:
- **Core training**: `diffusers`, `transformers`, `accelerate`, `peft`, `datasets`
- **Vision + utilities**: `torchvision`, `pillow`, `matplotlib`, `opencv-python`
- **Efficiency**: `bitsandbytes`, `xformers`
- **OCR**: `pytesseract`
- **Compatibility fix**: `numpy==1.24.4`
- **Tesseract (Colab)**: `apt-get install tesseract-ocr libtesseract-dev`

### 2.3 Reproducibility settings (seed)
The notebook sets:
- `SEED = 42`
- seeds for Python `random`, NumPy, and PyTorch
- deterministic cuDNN flags (`cudnn.deterministic=True`, `benchmark=False`)

**Why it matters (exam justification):**
- With seeds fixed, training curves and generated samples are more reproducible, which supports fair evaluation.

### 2.4 Directories created automatically
The notebook creates these folders if missing:
- `./data` → dataset (images + metadata.json)
- `./lora_weights` → best LoRA adapter + checkpoints
- `./outputs` → plots, generated samples, CSV results, summary report

---

## 3. Data Preprocessing & Augmentation

### 3.1 Dataset idea (what “crowd-sourced” means here)
The notebook **simulates** crowd-sourced text-image pairs by generating images that resemble:
- signs (STOP, EXIT, NO PARKING, etc.)
- labels/badges/stamps
- short multi-word instructions (KEEP CLEAN, TURN OFF LIGHTS, etc.)

This is a controlled way to test LoRA fine-tuning under limited data conditions.

### 3.1.1 Text vocabulary (numerics)
The generator uses a fixed list of **57 unique text strings** (examples: `STOP`, `EXIT`, `NO PARKING`, `CUSTOMER SERVICE`, `TURN OFF LIGHTS`).

**Category breakdown (count of unique texts):**
- Signs & labels: **16**
- Business/retail: **8**
- Informational: **8**
- Directions: **7**
- Generic UI-like words: **12**
- Multi‑word instructions: **6**
- **Total**: 16 + 8 + 8 + 7 + 12 + 6 = **57**

**Sampling logic (important):**
- Each training sample picks one text with `random.choice(texts)` → approximately **uniform probability \(1/57\)** per text.
- With only **50** training images, some texts may appear multiple times and some may not appear at all.  
  This is part of the “limited data” challenge.

### 3.1.2 Style labels used (numerics)
If `style="random"`, the generator chooses one of:
- `sign`, `label`, `simple`, `badge`, `stamp`

Because it is `random.choice([...])`, each style has ~**20%** probability.

Additional randomness:
- **Border (sign style)**: added with probability ~**50%** (condition `random.random() > 0.5`).
- **Noise**: added with probability ~**50%** (condition `random.random() > 0.5`).

### 3.2 Synthetic data generation (TextImageGenerator)
For each sample:
- **Choose text** from a prompt list (signs/labels/retail/informational/directions/generic).
- **Create a 512×512 RGB image** with a **random background color**.
- Choose a **contrasting text color** (black text on bright background, white on dark).
- Render text with font:
  - tries: `DejaVuSans-Bold.ttf` (size randomly 40–80)
  - fallback: default font if not found
- Center the text on the image.
- Optional style variation:
  - **border rectangle** for “sign” style (sometimes)
- Optional realism:
  - add **Gaussian noise** (mean 0, std ~5)

### 3.2.1 Why these design choices simulate “crowd-sourced” images
Real crowd-sourced text images (phones, CCTV, screenshots, posters) vary a lot.  
Our synthetic generator adds controlled variations in:
- **Background**: random colors emulate different surfaces/lighting.
- **Contrast**: choosing contrasting text color makes text visible (and learnable).
- **Font size**: random size teaches robustness to scale.
- **Layout**: centered text + optional border teaches “sign-like” composition.
- **Noise**: approximates camera sensor noise/compression artifacts.

### 3.2.2 Prompt template used for training (important)
Each image is paired with a descriptive text prompt. Example:
- `a sign image with the text 'STOP' in clear, readable font`

**Why it matters:** Stable Diffusion is trained to map **prompt → image**.  
If prompts contain domain/style cues (“sign”, “label”, “badge”), the model learns to generate in that domain.

### 3.2.3 Generation parameters (numerics)
These are the exact numeric choices used in the generator:
- **Image size**: \(512 \times 512\) RGB
- **Background color**: each RGB channel sampled as an integer in **[0, 255]**
- **Text color**: chosen as black/white based on background brightness (contrast rule)
- **Font size**: random integer in **[40, 80]** (when DejaVu font is available)
- **Border (optional)**:
  - only for some `sign` samples
  - border margin: **20 px**
  - outline width: **5 px**
- **Noise (optional)**:
  - Gaussian noise sampled from \( \mathcal{N}(0, 5)\)
  - added to pixels and clipped back to **[0, 255]**

### 3.2.4 Dataset generation pseudocode (easy to understand)
```text
for i in range(num_samples):
  text  <- random choice from 57 texts
  style <- random choice from {sign, label, simple, badge, stamp}

  bg_color   <- random RGB in [0..255]
  text_color <- black if bg is bright else white
  font_size  <- random integer 40..80

  draw centered text on a 512x512 image
  optionally draw border (for sign style)
  optionally add Gaussian noise

  prompt <- "a <style> image with the text '<text>' in clear, readable font"
  save {id, image, text, prompt}
```

### 3.3 Training/validation split sizes (from Config)
- **Training samples**: 50  
- **Validation samples**: 10

### 3.4 Saved dataset format on disk
The notebook saves the dataset to:
- `./data/train/images/0000.png`, `0001.png`, ...  
- `./data/train/metadata.json`
- `./data/val/images/...`
- `./data/val/metadata.json`

Each metadata entry contains:
- `id`
- `image_path`
- `text` (ground-truth label, e.g., `"STOP"`)
- `prompt` (the generation prompt used for training, e.g., `a sign image with the text 'STOP' in clear, readable font`)

### 3.4.1 Example `metadata.json` entry (exact structure)
Each `metadata.json` file is a **list of objects** like:
```json
{
  "id": 0,
  "image_path": "./data/train/images/0000.png",
  "text": "STOP",
  "prompt": "a sign image with the text 'STOP' in clear, readable font"
}
```

**How it is used later:**
- During training, the dataset class reads:
  - image pixels (from `image` in memory or `image_path` on disk),
  - `prompt` → tokenization for conditioning,
  - `text` is kept as the ground-truth label (used mainly for evaluation/reporting).

### 3.5 PyTorch Dataset preprocessing (TextImageDataset)
The training dataset class performs:
- **Transforms**
  - Resize to `resolution` (512)
  - Center crop to 512
  - Convert to tensor
  - Normalize with `Normalize([0.5], [0.5])` → values mapped to **[-1, 1]**
- **Prompt tokenization**
  - CLIP tokenizer
  - pad to max length
  - truncation enabled

### 3.5.1 Tensor shapes + normalization math (numerics)
This is what happens numerically to each image:
- Start: PIL image (RGB) with size **512×512×3**
- After `ToTensor()`: tensor shape **(3, 512, 512)** with values in **[0, 1]**
- After normalization `Normalize([0.5],[0.5])`:
  - \(x_{norm} = (x - 0.5) / 0.5 = 2x - 1\)
  - Since \(x = \text{pixel}/255\), we get:
    - \(x_{norm} = 2(\text{pixel}/255) - 1\) → approximately **[-1, 1]**

DataLoader batch shapes:
- `pixel_values`: **(B, 3, 512, 512)** where \(B=1\)
- `input_ids`: **(B, L)** where \(L = tokenizer.model_max_length\)  
  (CLIP in Stable Diffusion commonly uses **77 tokens**, and the notebook reads this from the tokenizer.)

### 3.5.2 Why CenterCrop (and when RandomCrop helps)
- We use **CenterCrop** because synthetic text is centered, and we want consistent training signals.
- In real-world data, **RandomCrop** can be a good augmentation to teach robustness to framing.

### 3.6 Why preprocessing is required (exam explanation)
Stable Diffusion fine-tuning expects:
- **fixed resolution** (we choose 512 for SD v1.5 standard training)
- **normalized tensors** (we map pixel values to approximately [-1, 1])
- **tokenized prompts** with consistent length for CLIP conditioning

If we skip these, training becomes unstable or incompatible with the model components.

### 3.7 Improvements (data + preprocessing)
To make the dataset closer to real crowd-sourced images and improve OCR:
- add perspective distortion (phone camera angle),
- add blur / motion blur,
- add stronger noise + compression artifacts,
- add multiple fonts and font weights,
- generate multi-line text (labels/posters),
- segment/crop text region before OCR for more fair evaluation.

---

## 4. Model Development (Stable Diffusion + LoRA)

### 4.1 Base model
- **Model ID**: `runwayml/stable-diffusion-v1-5`
- Components loaded:
  - `CLIPTokenizer`
  - `CLIPTextModel` (text encoder)
  - `AutoencoderKL` (VAE)
  - `UNet2DConditionModel` (denoiser)
  - `DDPMScheduler` (noise schedule)

### 4.1.1 Stable Diffusion components (deep explanation)
Stable Diffusion uses a **text-conditioned diffusion** process in a compressed latent space:
- **CLIP Text Encoder**: converts the prompt into embeddings (conditioning \(c\)).
- **VAE (AutoencoderKL)**: encodes an image into a smaller latent tensor \(z\). Training happens in this latent space (faster than pixel space).
- **UNet**: predicts the noise residual at each timestep, conditioned on \(c\).
- **Noise Scheduler (DDPM)**: defines how noise is added/removed over timesteps.

In simple terms: the model learns to start from noise and gradually “denoise” into an image that matches the prompt.

### 4.1.2 Latent-space numerics (important shapes)
Stable Diffusion v1.5 works in **latent space** instead of pixel space.

For our resolution **512×512**:
- Pixel tensor: **(B, 3, 512, 512)**
- VAE downsampling factor is **8×**, so latent spatial size becomes:
  - \(512 / 8 = 64\)
- Latent tensor (typical for SD v1.x):
  - **(B, 4, 64, 64)**  (4 latent channels)

In the training code:
- We sample latents from the VAE distribution:
  - `latents = vae.encode(...).latent_dist.sample()`
- Then scale them:
  - `latents = latents * vae.config.scaling_factor`
  - (For SD v1.x this factor is commonly **0.18215**, and the notebook reads it from the model config.)

**Why scaling matters:** it keeps the latent magnitude consistent with what the UNet was originally trained on.

### 4.1.3 Where the text prompt affects the image (cross-attention)
Inside the UNet, **cross-attention** layers combine:
- image/latent features \(X\)
- text embeddings \(C\) (from CLIP)

The core attention computation is:
\[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]
where:
- \(Q = XW_Q\) (queries from latents)
- \(K = CW_K\), \(V = CW_V\) (keys/values from text embeddings)
- output goes through \(W_O\)

**This is exactly why training `to_q`, `to_k`, `to_v`, and `to_out` is powerful:**  
it directly changes how the model “pays attention” to the prompt words when generating images.

### 4.1.4 Numerics for text conditioning (CLIP embeddings)
For Stable Diffusion v1.5:
- Token sequence length \(L\) is typically **77 tokens** (the notebook uses `tokenizer.model_max_length`)
- CLIP text embedding dimension is **768**

So `encoder_hidden_states` is typically shaped like:
- **(B, L, 768)** → e.g., **(1, 77, 768)**

### 4.1.5 UNet denoiser architecture (what the UNet “is”)
The UNet used in Stable Diffusion is a **multi‑scale encoder–decoder** network with skip connections:
- **Down blocks**: progressively reduce spatial resolution and increase channels (extract features).
- **Mid block**: processes the smallest-resolution features (often includes attention).
- **Up blocks**: progressively increase resolution while merging skip connections from down blocks.
- **Cross‑attention blocks**: inserted at multiple resolutions so the model can condition image features on the text embeddings.

You can think of it like this (high level):

```text
noisy_latents z_t (B,4,64,64)
   │ + time embedding(t)
   ▼
[Down blocks: ResNet + (Self/Cross-Attn)]  ──┐  (skip connections)
   ▼                                        │
[Mid block: ResNet + Attn]                  │
   ▼                                        │
[Up blocks: ResNet + (Self/Cross-Attn)]  ◀──┘
   ▼
predicted_noise εθ (B,4,64,64)
```

**Why this architecture is appropriate**
- The UNet can model both **global layout** (where text should go) and **local details** (stroke shapes).
- Skip connections help keep fine details (important for readable text).

### 4.1.6 Noise scheduler (DDPM) — the numerics behind timesteps
Diffusion defines a sequence of noise levels using \(\beta_t\) (noise variance schedule):
- \(\alpha_t = 1 - \beta_t\)
- \(\bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i\)

Forward process (corrupting a clean latent \(z_0\)):
\[
z_t=\sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
\]

In the notebook:
- \(t\) is sampled uniformly as an integer timestep (often total timesteps ≈ 1000 in SD schedulers).
- `noise_scheduler.add_noise(latents, noise, timesteps)` implements the equation above.

Reverse process (sampling / generation):
- Start from pure noise \(z_T\sim \mathcal{N}(0,I)\)
- For \(t = T \rightarrow 0\), repeatedly:
  - predict noise with UNet: \(\epsilon_\theta(z_t,t,c)\)
  - scheduler computes the next latent \(z_{t-1}\)

<details>
<summary><b>Click: inference algorithm pseudocode (how an image is generated)</b></summary>

```text
z_T ~ Normal(0, I)
for t = T ... 1:
  eps = UNet(z_t, t, text_embeddings)
  z_{t-1} = scheduler.step(eps, t, z_t)
image = VAE.decode(z_0)
```

</details>

### 4.2 LoRA configuration (what is trained)
The notebook freezes all base parameters and trains only LoRA adapters on the UNet attention modules:
- **Target modules**: `["to_k", "to_q", "to_v", "to_out.0"]`
- **Rank (r)**: 4  
- **Alpha**: 4  
- **Dropout**: 0.0  
- **Init**: gaussian  

Trainable parameter summary printed in the notebook:
- **trainable params**: 797,184  
- **all params**: 860,318,148  
- **trainable%**: 0.0927%

### 4.2.1 LoRA intuition (simple math)
In a normal linear layer, we have weights \(W\). LoRA adds a low-rank update:
\[
W' = W + \Delta W,\quad \Delta W = B A
\]
where \(A\) and \(B\) are small matrices with rank \(r\) (here \(r=4\)).  
We train only \(A\) and \(B\), not the full \(W\).

**Why this is important for Colab:** training all SD weights is too heavy; LoRA makes it feasible.

### 4.2.2 What are `to_q`, `to_k`, `to_v`, and `to_out.0`? (attention numerics)
In a cross-attention block, we compute linear projections:
- **Query**: \(Q = XW_Q\)  → implemented as `to_q`
- **Key**: \(K = CW_K\)    → implemented as `to_k`
- **Value**: \(V = CW_V\)  → implemented as `to_v`
- **Output projection**: \(O = \text{Attention}(Q,K,V)W_O\) → implemented as `to_out.0`

LoRA adds trainable low-rank updates to these matrices, so we adapt how text and image features interact.

### 4.2.3 LoRA parameter count intuition (why 797,184 is “small”)
For a linear layer \(W \in \mathbb{R}^{d_{out}\times d_{in}}\), LoRA adds:
- \(A \in \mathbb{R}^{r \times d_{in}}\)
- \(B \in \mathbb{R}^{d_{out} \times r}\)

So LoRA parameters per layer are:
- \(r(d_{in} + d_{out})\)

Because we use **r = 4**, this is much smaller than \(d_{out}\times d_{in}\).  
Applying this across all UNet attention projections results in **797,184 trainable parameters** total, which is:
- \(797,184 / 860,318,148 \approx 0.000927 = 0.0927\%\)

### 4.2.4 What does LoRA “alpha” do?
LoRA uses a scaling factor. In common LoRA implementations (including PEFT), the update is scaled roughly by:
- \(\alpha / r\)

Here:
- \(\alpha = 4\), \(r = 4\) → scaling ≈ **1.0**

This means the adapter update magnitude is in a reasonable range relative to the base weights.

### 4.2.5 Rank/alpha trade-offs (how to tune LoRA intelligently)
This is the simplest tuning intuition:
- **Rank \(r\)** controls *capacity* (how much the adapter can change the model).
  - Higher \(r\) → more trainable parameters → can learn sharper text, but increases VRAM/time and overfitting risk.
  - Lower \(r\) → very efficient, but may underfit text spelling details.
- **Alpha \(\alpha\)** controls *update scale* relative to rank (roughly \(\alpha/r\)).
  - If \(\alpha/r\) is large, updates can be strong (sometimes unstable unless LR is reduced).
  - If \(\alpha/r\) is small, updates may be too weak (may under-adapt).

**Practical tuning suggestion (exam-ready):**
- First try increasing \(r\) (4 → 8) while keeping \(\alpha/r\) ≈ 1 (e.g., \(\alpha=8\)).
- If training becomes unstable, reduce LR (e.g., 1e-4 → 5e-5).

### 4.3 Device and precision
- Device: CUDA if available
- Mixed precision: fp16 on GPU (VAE and Text Encoder are cast to fp16; UNet handled by `accelerate`)

### 4.4 Improvements (model-side)
Below is a more detailed improvement scope with **theory**, **expected impact**, and **trade-offs**.

| Improvement | Why it helps (theory) | Expected effect | Cost / trade-off |
|---|---|---|---|
| Increase LoRA rank (4 → 8/16) | More adapter capacity in attention projections → better text conditioning | Higher character clarity + potentially better OCR | More VRAM + time; overfitting risk with tiny data |
| Expand target modules / layers | Adapts more of UNet, not only attention projections | Stronger domain shift adaptation | More parameters; may need more data |
| LoRA on text encoder | Adapts prompt embeddings → better mapping from words to strokes | Better prompt-text alignment | Risk: overfit prompt style; needs careful LR |
| Higher resolution training (512 → 768) | More pixels per character; latent becomes larger (96×96) | Sharper characters | Much higher VRAM/time |
| OCR-guided (multi‑task) training | Adds recognition objective (forces spelling correctness) | Biggest expected OCR jump | Requires additional model + more code complexity |
| OCR‑based selection (no training) | Generate N candidates and keep best OCR score | Improves measured OCR without retraining | Slower inference (N×) |

<details>
<summary><b>Click: OCR-guided training idea (how it works conceptually)</b></summary>

1. Generate image with SD (fine-tuned or base).
2. Run an OCR/recognition model (e.g., TrOCR/CRNN) to predict text.
3. Compute a loss between predicted text and ground truth (cross-entropy / edit-distance proxy).
4. Backprop into LoRA adapters (or use reinforcement/gradient estimators if OCR is non-differentiable).

**Why it helps:** it turns the task into a multi-task system: image realism + text correctness.

</details>

---

## 5. Training

### 5.1 Hyperparameters (Config)
| Category | Parameter | Value |
|---|---:|---:|
| Model | `MODEL_ID` | `runwayml/stable-diffusion-v1-5` |
| Data | `RESOLUTION` | 512 |
| Data | `NUM_TRAINING_SAMPLES` | 50 |
| Data | `NUM_VALIDATION_SAMPLES` | 10 |
| LoRA | `LORA_RANK` | 4 |
| LoRA | `LORA_ALPHA` | 4 |
| LoRA | `LORA_DROPOUT` | 0.0 |
| LoRA | `LORA_TARGET_MODULES` | `to_k,to_q,to_v,to_out.0` |
| Train | `BATCH_SIZE` | 1 |
| Train | `GRADIENT_ACCUMULATION_STEPS` | 4 (effective batch = 4) |
| Train | `NUM_EPOCHS` | 20 |
| Train | `LEARNING_RATE` | 1e-4 |
| Optim | AdamW betas | (0.9, 0.999) |
| Optim | Weight decay | 1e-2 |
| Optim | Epsilon | 1e-8 |
| Train | `MAX_GRAD_NORM` | 1.0 |
| Scheduler | `LR_SCHEDULER` | `constant_with_warmup` |
| Scheduler | `LR_WARMUP_STEPS` | 0 |
| Inference | `INFERENCE_STEPS` | 50 |
| Inference | `GUIDANCE_SCALE` | 7.5 |
| Paths | `OUTPUT_DIR` | `./outputs` |
| Paths | `DATA_DIR` | `./data` |
| Paths | `MODEL_SAVE_DIR` | `./lora_weights` |
| Repro | `SEED` | 42 |

### 5.2 Data loaders
- Train dataloader: 50 batches (batch size 1, shuffled)
- Val dataloader: 10 batches (batch size 1)
- `num_workers=0` for Colab compatibility

### 5.2.1 How many training iterations happen? (numerics)
From the notebook settings:
- Batches per epoch: **50**
- Epochs: **20**

So total forward/backward iterations are approximately:
- \(50 \times 20 = 1000\) training iterations

With gradient accumulation = 4, the effective number of optimizer updates is roughly:
- \(1000 / 4 \approx 250\) updates  
(conceptually; this is why accumulation helps simulate larger batches).

### 5.3 Training objective (diffusion loss)
For each batch:
1. Encode image → **latent** using VAE (no grad)
2. Sample random noise + random timestep
3. Add noise using scheduler
4. Get CLIP text embeddings (no grad)
5. UNet predicts noise residual
6. Compute **MSE loss** between predicted noise and target noise
7. Backprop only through **LoRA parameters**
8. Gradient clipping + optimizer step + scheduler step

### 5.3.1 Loss definition (exam-level clarity)
The notebook uses the standard diffusion objective (noise prediction). In simplified form:
\[
\mathcal{L} = \|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2
\]
where:
- \(z_t\) = noised latent at timestep \(t\),
- \(c\) = text embedding from CLIP,
- \(\epsilon_\theta\) = UNet prediction.

### 5.3.2 Why gradient accumulation is required (Colab constraint)
We set:
- `BATCH_SIZE = 1` (fits GPU memory)
- `GRADIENT_ACCUMULATION_STEPS = 4`

This means the optimizer updates after 4 steps → **effective batch size = 4** without needing the VRAM of batch size 4.

### 5.3.3 Why mixed precision (fp16) is used
Mixed precision:
- reduces GPU memory usage,
- speeds up training on GPUs like Tesla T4,
- is a standard technique for diffusion fine-tuning.

### 5.3.4 Forward diffusion equation (concept + numerics)
Diffusion training works by corrupting the clean latent \(z_0\) into a noisy latent \(z_t\).

The “forward process” is:
\[
z_t = \sqrt{\bar{\alpha}_t}\,z_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
\]

In the notebook:
- \(t\) is sampled randomly per image:
  - `timesteps = randint(0, noise_scheduler.config.num_train_timesteps)`
  - (For SD v1.5 schedulers, `num_train_timesteps` is commonly **1000**.)
- Noise is added using:
  - `noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)`

### 5.3.5 “Epsilon prediction” vs “v-prediction” (why the code checks this)
Different diffusion parameterizations exist:
- **epsilon prediction**: UNet predicts \(\epsilon\) directly (very common for SD v1.x)
- **v-prediction**: UNet predicts a “velocity” \(v\)

The notebook handles both:
- if `prediction_type == "epsilon"` → target = `noise`
- if `prediction_type == "v_prediction"` → target = `noise_scheduler.get_velocity(...)`

Velocity is typically defined using:
\[
v = \alpha_t \epsilon - \sigma_t z_0
\]
where \(\alpha_t = \sqrt{\bar{\alpha}_t}\), \(\sigma_t = \sqrt{1-\bar{\alpha}_t}\).

### 5.3.6 Tensor shapes during one training step (numerics)
For one batch (here \(B=1\)):
- `pixel_values`: **(1, 3, 512, 512)**
- `latents` (after VAE): **(1, 4, 64, 64)**
- `noise`: **(1, 4, 64, 64)**
- `timesteps`: **(1,)**
- `encoder_hidden_states`: **(1, L, 768)** (typically \(L=77\))
- `model_pred` (UNet output): **(1, 4, 64, 64)**
- loss: scalar (single float)

This “shape trace” is a strong exam explanation of *what actually flows through the network*.

### 5.4 Checkpointing strategy
The notebook saves:
- **Best LoRA weights** to: `./lora_weights/best_model` (whenever training loss improves)
- **Checkpoint every 10 epochs**: `./lora_weights/checkpoint_epoch_10`, `checkpoint_epoch_20`, ...

### 5.5 What to look for in training curves (how to justify training)
Open `./outputs/training_curves.png`:
- **Loss should generally decrease** (indicates the model is learning).
- Sudden spikes can indicate instability or too high learning rate.

### 5.6 Improvements (training-side)
If performance is weak, we can:
- increase dataset size and train longer,
- tune LR, epochs, LoRA rank/alpha,
- run periodic validation sampling (not just training loss),
- try different schedulers (cosine, linear warmup).

---

## 6. Evaluation (OCR-based)

### 6.1 Load best model for inference
The notebook loads a `StableDiffusionPipeline`:
- base pipeline from `runwayml/stable-diffusion-v1-5`
- replaces pipeline UNet
- applies LoRA adapter weights from `./lora_weights/best_model`
- sets `safety_checker=None` (for simplicity in this assignment setting)

### 6.2 Generate images
- Generates images for **all validation prompts**  
- Uses:
  - `num_inference_steps = 50`
  - `guidance_scale = 7.5`
  - seeded generator (`SEED = 42`)

### 6.2.1 What do “inference steps” mean? (numerics)
Stable Diffusion generates an image by running a denoising process for a fixed number of steps:
- **More steps** → usually cleaner images but slower
- **Fewer steps** → faster but can reduce quality

We use **50 steps**, which is a common “high quality” setting for SD v1.5.

### 6.2.2 What does “guidance scale” mean? (classifier-free guidance)
Stable Diffusion often uses **classifier-free guidance (CFG)** to enforce prompt alignment.

Conceptually, the guided prediction is:
```text
pred = pred_uncond + s * (pred_cond - pred_uncond)
```
where:
- `pred_uncond` = prediction without the prompt
- `pred_cond` = prediction with the prompt
- \(s\) = **guidance_scale** (here **7.5**)

**Interpretation:**
- If \(s\) is too low → the model may ignore the prompt (weak conditioning)
- If \(s\) is too high → can introduce artifacts or distort characters

### 6.3 OCR preprocessing + extraction
OCR function steps:
- convert to grayscale
- increase contrast (`cv2.convertScaleAbs`, alpha=1.5)
- Otsu thresholding
- `pytesseract.image_to_string(..., config='--psm 6')`

**Why these steps help OCR (simple explanation):**
- grayscale reduces color complexity and focuses OCR on intensity patterns
- contrast increase makes strokes darker/lighter and improves separation
- Otsu threshold converts to a clean black/white mask (helps character segmentation)
- `--psm 6` tells Tesseract to assume a single uniform block of text (good for signs/labels)

### 6.4 Metrics computed
1. **Exact Match** (boolean): normalized predicted text equals normalized ground truth  
2. **Character Accuracy** (0–1):
   - lowercase + strip both strings
   - count matching characters in `zip(gt, pred)`
   - divide by `max(len(gt), len(pred))`

Results are saved to `./outputs/evaluation_results.csv`.

### 6.4.1 Metric formulas (clear + exam-ready)
**Exact Match Rate**
- For each sample:
  - normalize ground truth and OCR text (lowercase + strip)
  - exact match if they are identical
- Exact Match Rate = average of exact match over all validation samples.

**Character Accuracy (as implemented in notebook)**
- Let `gt` and `pred` be normalized strings.
- Count aligned matches:
  - `matches = sum(1 for a,b in zip(gt,pred) if a==b)`
- Final score:
  - `matches / max(len(gt), len(pred))`

**Important note:** This is a simple metric (not full edit-distance). It is still valid for this assignment, but CER/WER would be more standard.

### 6.5 What we could additionally measure (improvements)
To strengthen the evaluation section for full marks:
- **CER / WER** using edit distance (more rigorous than zip-based matching).
- OCR confidence scores (Tesseract provides confidence per word/line).
- **Speed**: average inference time per image (metric often required in rubrics).
- Quality metrics:
  - CLIPScore for prompt-image alignment,
  - FID for distribution similarity (if a real dataset is available),
  - human evaluation (readability rating).

---

## 7. Results & Analysis

### 7.1 Quantitative OCR results (from the notebook run)
- **Total samples**: 10  
- **Exact Match Rate**: 0.00%  
- **Average Character Accuracy**: 6.58%  
- **Min Character Accuracy**: 0.00%  
- **Max Character Accuracy**: 31.82%

### 7.2 Qualitative analysis (printed in the notebook)
The notebook highlights:
- **Domain adaptation**: model shifts from general image generation toward operational text-like images (signs/labels/badges)
- **Challenges**:
  - OCR accuracy depends on **font size** and **contrast**
  - some images show artistic interpretation rather than literal text
  - dataset is small (50 train samples) → limits generalization
- **LoRA efficiency**: trains only 797,184 parameters; feasible on Colab GPU

### 7.3 Why OCR scores are low (detailed justification)
Low OCR does **not** necessarily mean the model learned nothing. Common reasons:
- **Diffusion text problem**: SD often produces “text-like” shapes that look plausible but are not exact letters.
- **Small training set** (50 samples): the model sees limited diversity and may underfit.
- **Sampling randomness**: diffusion generation can distort characters even with fixed prompts.
- **OCR sensitivity**: slight blur/contrast issues cause big drops in OCR output quality.

### 7.4 What improved after fine-tuning (what to show to professor)
Open these outputs:
- `./outputs/before_after_comparison.png`
- `./outputs/generated_images_with_ocr.png`

Look for:
- more consistent centered placement of text,
- more sign/label-like composition,
- clearer “text intention” compared to base model.

---

## 8. Deliverables (What gets saved)

### 8.1 Main folders
After a full run, the notebook produces:
- **`./data/`**: synthetic dataset + metadata
- **`./lora_weights/`**: saved LoRA adapters (best + checkpoints)
- **`./outputs/`**: plots, generated samples, CSV results, summary report

### 8.2 Expected output files (from notebook “Project Completion Summary”)
- **LoRA weights**: `./lora_weights/`
- **Training curves**: `./outputs/training_curves.png`
- **Evaluation metrics plot**: `./outputs/evaluation_metrics.png`
- **Generated samples with OCR text**: `./outputs/generated_images_with_ocr.png`
- **Before/After comparison**: `./outputs/before_after_comparison.png`
- **Evaluation CSV**: `./outputs/evaluation_results.csv`
- **Final summary report**: `./outputs/summary_report.txt`

Optional (also present in the notebook):
- dataset sample visualization: `./outputs/dataset_samples.png`
- custom prompt test images: `./outputs/custom_test_images.png`

### 8.4 How to interpret each deliverable (exam-friendly)
- `training_curves.png`: verifies the training process (loss trend over epochs).
- `evaluation_results.csv`: per-sample OCR output + metrics (proof of quantitative evaluation).
- `evaluation_metrics.png`: metric visualization (helps professor grade quickly).
- `generated_images_with_ocr.png`: qualitative results + OCR text side-by-side.
- `before_after_comparison.png`: direct evidence of improvement vs base model.
- `summary_report.txt`: executive summary (one-page report with config + key results).

### 8.3 Suggested project tree (after running)
```text
CVassignment/
├── CV_assignment2_group13_problemstatement4 (1).ipynb
├── README.md
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── metadata.json
│   └── val/
│       ├── images/
│       └── metadata.json
├── lora_weights/
│   ├── best_model/
│   ├── checkpoint_epoch_10/
│   └── checkpoint_epoch_20/
└── outputs/
    ├── dataset_samples.png
    ├── training_curves.png
    ├── evaluation_metrics.png
    ├── generated_images_with_ocr.png
    ├── before_after_comparison.png
    ├── evaluation_results.csv
    ├── summary_report.txt
    └── custom_test_images.png
```

---

## 9. How to Run / Reproduce

### Option A (Recommended): Run in Google Colab
1. Upload/open `CV_assignment2_group13_problemstatement4 (1).ipynb` in Colab
2. Set runtime:
   - **Runtime → Change runtime type → GPU**
3. Run notebook from top to bottom:
   - GPU check
   - install libraries
   - generate dataset → save to `./data/`
   - load model → apply LoRA
   - train → save `./lora_weights/`
   - inference + OCR evaluation → save `./outputs/`

### Option B: Run locally (macOS/Linux)
This notebook is written for Colab-style installs, but you can run locally if you have:
- a compatible **CUDA GPU** (training without GPU is very slow)
- Python environment with the same libraries
- Tesseract installed and available in PATH

Minimal local steps (example):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -U diffusers transformers accelerate peft datasets pytesseract pillow matplotlib opencv-python torchvision bitsandbytes xformers
pip install numpy==1.24.4 pandas
```

Install tesseract:
- **macOS (Homebrew)**:
```bash
brew install tesseract
```
- **Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev
```

---

## 10. Troubleshooting

### 10.1 CUDA/GPU not detected
- The notebook prints a warning if CUDA is unavailable.
- For best results: use Colab GPU or a local CUDA GPU.

### 10.2 Stable Diffusion model download issues
`runwayml/stable-diffusion-v1-5` may require:
- internet access for downloading weights
- accepting model terms on Hugging Face (depending on account settings)

### 10.3 OCR not working
If `pytesseract` fails:
- ensure **Tesseract** is installed (system package)
- ensure it is discoverable in PATH

### 10.4 Poor OCR scores
This is expected with tiny datasets and stylized generations.
Main factors:
- small font size
- low contrast
- model “artistic” text-like strokes rather than crisp characters

### 10.5 Common Colab issues (practical tips)
- **First run is slow**: the SD model downloads multiple components the first time.
- **Out-of-memory (VRAM)**:
  - keep batch size = 1,
  - reduce inference steps during testing,
  - clear cache (`torch.cuda.empty_cache()`).
- **NumPy/Pandas mismatch**: notebook pins `numpy==1.24.4` to avoid binary compatibility errors.

---

## 11. Limitations & Future Work (as stated in notebook)
- Limited training data (50 samples) affects generalization
- OCR depends on image quality + preprocessing
- More diverse fonts/styles would help
- Possible **underfitting**: dataset is too small for the model to learn exact character spelling reliably
- Possible **overfitting risk**: training loss can keep decreasing without OCR improvement; needs validation sampling/human checks
- Future work:
  - larger dataset
  - longer training
  - hyperparameter tuning (rank/alpha, LR, resolution, inference steps)

### 11.1 Improvement Roadmap (detailed, with theory + “what to change”)
This is a practical plan that a professor can use to see you understand *how to improve* the system.

#### Quick wins (no retraining)
| Change | Where to change | Why it helps | Expected impact |
|---|---|---|---|
| Generate 5 images per prompt and pick best by OCR | In evaluation: loop N times per prompt | OCR is noisy; best-of-N increases chance of readable text | Higher measured OCR, slower inference |
| Tune CFG guidance scale (e.g., 5.0, 7.5, 9.0) | `config.GUIDANCE_SCALE` | Too high CFG can distort letters; too low ignores prompt | Can improve readability at same model |
| Increase inference steps (50 → 75/100) for final results | `config.INFERENCE_STEPS` | More denoising steps reduces artifacts | Higher quality, slower sampling |

#### Data improvements (most important for text)
| Change | Where to change | Why it helps (theory) | Expected impact |
|---|---|---|---|
| Increase dataset size (50 → 5k+) | `NUM_TRAINING_SAMPLES` | More examples reduce underfitting and improve character learning | Stronger generalization + higher OCR |
| Add more fonts and weights | generator font selection | Model learns invariant character shapes across fonts | Better OCR robustness |
| Add perspective/blur/compression | generator augmentations | Matches real photo distortions; reduces domain gap | Better real-world performance |

#### Model/training improvements (capacity + stability)
| Change | Where to change | Why it helps | Expected impact |
|---|---|---|---|
| Increase LoRA rank (4 → 8) | `LORA_RANK`, `LORA_ALPHA` | More adapter capacity | Better adaptation, more VRAM |
| Train longer with more data | `NUM_EPOCHS` | More optimization steps after data scale-up | Higher quality, risk overfit if data still small |
| LoRA on text encoder | apply PEFT to text encoder too | Better prompt embedding alignment | Better spelling alignment, needs tuning |

#### Evaluation improvements (better scoring + proof)
| Change | Why it helps | Notes |
|---|---|---|
| Add CER/WER (edit distance) | standard OCR evaluation | stronger than zip-based accuracy |
| Track OCR confidence + speed | matches rubric “speed/metrics” | add mean inference time per image |

> **Exam tip**: If asked “what would you do next?”, pick one improvement from each category and justify with theory.

---

## 12. References
- **Stable Diffusion / Latent Diffusion**: Rombach et al. (Latent Diffusion Models)  
- **LoRA**: Hu et al. (Low-Rank Adaptation of Large Language Models)  
- **Hugging Face Diffusers** documentation (Stable Diffusion + training utilities)  
- **Tesseract OCR** documentation

---

## 13. Viva / Exam Q&A (Common Questions)

### Q1. Why use LoRA instead of full fine-tuning?
Full fine-tuning of Stable Diffusion updates hundreds of millions of parameters and needs large GPU memory + long runtime.  
LoRA updates only ~**0.09%** parameters (here **797,184**) while freezing the base model, making training feasible on Colab.

### Q2. Why freeze VAE and text encoder?
We want to adapt the **image generation behavior** (UNet attention) without destabilizing the full pipeline.  
Freezing VAE/text encoder reduces compute and keeps the base model’s general knowledge.

### Q3. What is “gradient accumulation” and why did we use it?
It accumulates gradients across multiple mini-steps before updating weights.  
Here it simulates an effective batch size of 4 while keeping memory usage like batch size 1.

### Q4. Why OCR evaluation?
The output goal is a **readable text image**. OCR provides an automatic, objective proxy of readability.

### Q5. Why is the exact match rate 0% but you still claim improvement?
Exact match is extremely strict (one wrong character = failure).  
The model can still improve *visual text structure* and placement, which is visible in `before_after_comparison.png` even if OCR remains low.

### Q6. What would you do to improve accuracy if given more time?
- Train with more data (and preferably real crowd-sourced samples).
- Improve augmentation (blur, perspective, lighting).
- Use CER/WER metrics and OCR confidence.
- Add OCR-guided training (multi-task idea).

---

### Exam / Viva-ready summary (one paragraph)
This assignment fine-tunes Stable Diffusion v1.5 using LoRA adapters applied to UNet attention layers, training only ~0.09% of parameters. A small synthetic dataset of operational text images (signs/labels) is generated and used for training under Colab constraints (batch size 1 + gradient accumulation). The fine-tuned model is evaluated by generating images for validation prompts and measuring readability using Tesseract OCR with exact match and character accuracy metrics, producing reproducible outputs (CSV, plots, and a full summary report) saved under `./outputs/` and LoRA weights under `./lora_weights/`.

---

## Appendix A. Notebook Cell-by-Cell Walkthrough
This appendix explains **each notebook cell** (markdown + code) in order of the notebook’s own section numbering.  
Goal: a professor can quickly verify correctness, and a beginner can understand *why each cell exists*.

> Important: the notebook is organized as “Markdown heading cell → Code cell”.  
> Below, I treat each heading and its following code cell(s) as separate “cells”.

### A0. Notebook Cell Index (65 cells) — click to expand

<details>
<summary><b>Cell index table (Cell 000 → Cell 064)</b></summary>

| Cell | Type | Notebook heading / first line | Main output / purpose |
|---:|---|---|---|
| 000 | markdown | Assignment 2: Fine-Tuning a Text-to-Image Model Using Crowd-Sourced Text–Image Pairs | Title + intro + ToC |
| 001 | markdown | 2. Environment Setup | Section heading |
| 002 | code | import torch | GPU/CUDA availability + device info print |
| 003 | markdown | 2.2 Install Required Libraries | Install section heading |
| 004 | code | %%capture | Installs Python packages + system Tesseract |
| 005 | markdown | 2.3 Import Libraries | Import section heading |
| 006 | code | %%capture | Imports all training/OCR/plotting libraries |
| 007 | markdown | 2.4 Set Random Seeds for Reproducibility | Reproducibility heading |
| 008 | code | def set_seed(seed: int = 42): | Sets random seeds + deterministic flags |
| 009 | markdown | 2.5 Configuration and Hyperparameters | Config heading |
| 010 | code | class Config: | Central hyperparameters + output folders |
| 011 | markdown | 3. Data Preprocessing & Augmentation | Data section heading |
| 012 | code | class TextImageGenerator: | Synthetic text-image generator |
| 013 | markdown | 3.2 Create Training and Validation Datasets | Split heading |
| 014 | code | # Initialize generator | Generate 50 train + 10 val samples |
| 015 | markdown | 3.3 Save Dataset to Disk | Save heading |
| 016 | code | def save_dataset(dataset: List[Dict], split: str = "train"): | Save images + metadata.json |
| 017 | markdown | 3.4 Visualize Sample Images | Visualization heading |
| 018 | code | def visualize_samples(dataset: List[Dict], num_samples: int = 6): | Save `outputs/dataset_samples.png` |
| 019 | markdown | 3.5 Custom Dataset Class for PyTorch | Dataset heading |
| 020 | code | class TextImageDataset(Dataset): | Preprocess tensors + tokenize prompts |
| 021 | markdown | 4. Model Development | Model section heading |
| 022 | code | print(f"Loading Stable Diffusion model: {config.MODEL_ID}") | Load tokenizer, text encoder, VAE, UNet, scheduler |
| 023 | markdown | 4.2 Configure LoRA for Fine-tuning | LoRA heading |
| 024 | code | # Freeze all model parameters (we only want to train LoRA weights) | Freeze + apply LoRA adapters |
| 025 | markdown | 4.3 Move Models to Device | Device heading |
| 026 | code | # Move models to device | `.to(cuda)` + fp16 setup |
| 027 | markdown | 5. Training | Training section heading |
| 028 | code | # Create datasets | Dataloaders (train/val) |
| 029 | markdown | 5.2 Setup Optimizer and Scheduler | Optim heading |
| 030 | code | # Only optimize LoRA parameters | AdamW + LR scheduler |
| 031 | markdown | 5.3 Initialize Accelerator for Mixed Precision Training | Accelerator heading |
| 032 | code | # Initialize accelerator | `Accelerator()` prepare |
| 033 | markdown | 5.4 Training Loop | Loop heading |
| 034 | code | def train_one_epoch( | Core diffusion training function |
| 035 | markdown | 5.5 Execute Training | Run heading |
| 036 | code | # Training history | Run epochs + save best/checkpoints |
| 037 | markdown | 5.6 Plot Training Curves | Plot heading |
| 038 | code | def plot_training_curves(history: Dict): | Save `outputs/training_curves.png` |
| 039 | markdown | 6. Evaluation Metrics | Evaluation heading |
| 040 | code | # Load the complete pipeline with LoRA weights | Load pipeline + load adapter |
| 041 | markdown | 6.2 Generate Images for Evaluation | Generation heading |
| 042 | code | def generate_images(pipeline, prompts: List[str], ...) | Generate validation images |
| 043 | markdown | 6.3 OCR-based Evaluation | OCR heading |
| 044 | code | def extract_text_from_image(image: Image.Image) -> str: | OCR preprocessing + metrics functions |
| 045 | markdown | 6.4 Perform Evaluation on Generated Images | Eval run heading |
| 046 | code | # Evaluate generated images | Run OCR + save CSV |
| 047 | markdown | 6.5 Visualize Evaluation Results | Eval plot heading |
| 048 | code | def visualize_evaluation_results(results_df: pd.DataFrame): | Save `outputs/evaluation_metrics.png` |
| 049 | markdown | 7. Results & Analysis | Results heading |
| 050 | code | def visualize_generated_images_with_ocr( | Save `outputs/generated_images_with_ocr.png` |
| 051 | markdown | 7.2 Compare Before and After Fine-tuning | Comparison heading |
| 052 | code | # Load base model without LoRA for comparison | Load base pipeline |
| 053 | code | def visualize_before_after_comparison( | Save `outputs/before_after_comparison.png` |
| 054 | markdown | 7.3 Qualitative Analysis | Qualitative heading |
| 055 | code | print("=" * 70) | Prints qualitative analysis text |
| 056 | markdown | 7.4 Save Final Summary Report | Report heading |
| 057 | code | summary_report = f""" | Writes `outputs/summary_report.txt` |
| 058 | markdown | 8. Conclusion | Conclusion heading |
| 059 | code | print("="*70) | Prints completion summary + deliverables |
| 060 | markdown | 8.2 Generate Test Images with Custom Prompts (Bonus) | Bonus heading |
| 061 | code | # Generate some custom test images | Saves `outputs/custom_test_images.png` |
| 062 | code | print("\\nTeam Member Contributions:\\n") | Prints contributions section |
| 063 | code | from datetime import datetime | Timestamp helper import |
| 064 | markdown | End of Assignment | End marker |

</details>

> **How to use this appendix**: press `Ctrl/Cmd + F` and search for `Cell 034`, `Cell 040`, etc.  
> Each cell below is **collapsible** (click-to-expand) so the README stays readable but still detailed.

### A0.1 Cell-by-cell explanations (click to expand)

<details>
<summary><b>Cell 000 (markdown)</b> — Assignment title + intro + ToC</summary>

- **What this cell contains**: the assignment title, problem statement, objectives, and a table of contents.
- **Why it exists**: establishes the report structure (important for grading and for readers).
- **Theory connection**: explains “domain shift” and why LoRA + OCR are used in this task.

</details>

<details>
<summary><b>Cell 001 (markdown)</b> — Section 2 heading (Environment Setup)</summary>

- **Purpose**: separates the report into a clear “setup” phase before implementation.

</details>

<details>
<summary><b>Cell 002 (code)</b> — GPU/CUDA availability check</summary>

- **Goal**: verify training can run on GPU (and show the GPU model + memory).
- **What the code does**:
  - prints Python + PyTorch versions
  - checks `torch.cuda.is_available()`
  - prints CUDA version, GPU name, and VRAM
- **Theory**:
  - diffusion fine-tuning is compute/VRAM heavy; VRAM limits batch size/resolution.
- **Expected output**: e.g., `CUDA Available: True`, `GPU Device: Tesla T4`, etc.

</details>

<details>
<summary><b>Cell 003 (markdown)</b> — “Install Required Libraries” heading</summary>

- **Purpose**: introduces the dependency installation step (so the notebook is reproducible).

</details>

<details>
<summary><b>Cell 004 (code)</b> — Install packages + install Tesseract</summary>

- **What it does**:
  - installs `diffusers`, `transformers`, `accelerate`, `peft` (training stack)
  - installs `pytesseract`, `opencv-python`, `pillow` (OCR + imaging)
  - installs `bitsandbytes`, `xformers` (efficiency)
  - pins `numpy==1.24.4` to avoid binary compatibility issues
  - installs system OCR engine: `tesseract-ocr`
- **Theory**:
  - Python `pytesseract` is only a wrapper; system Tesseract is required for OCR.
  - `peft` provides LoRA adapters; `accelerate` manages fp16 + accumulation.

> **Exam tip**: “Why pin NumPy?” → binary compatibility errors are common in Colab; pinning stabilizes the environment.

</details>

<details>
<summary><b>Cell 005 (markdown)</b> — “Import Libraries” heading</summary>

- **Purpose**: separates “installation” from “imports” (clean notebook style).

</details>

<details>
<summary><b>Cell 006 (code)</b> — Import training/OCR/plotting libraries</summary>

- **What it does**: imports:
  - PyTorch + DataLoader + transforms
  - `diffusers` components (UNet, VAE, scheduler, pipeline)
  - `peft` LoRA utilities
  - OCR libraries (pytesseract + OpenCV)
  - plotting (matplotlib)
- **Theory**: SD training needs text encoder + UNet + VAE; OCR evaluation needs image preprocessing + Tesseract.

</details>

<details>
<summary><b>Cell 007 (markdown)</b> — “Set Random Seeds” heading</summary>

- **Purpose**: introduces reproducibility (important for exam submissions).

</details>

<details>
<summary><b>Cell 008 (code)</b> — Set seeds (reproducibility)</summary>

- **What it does**:
  - sets seeds for `random`, NumPy, PyTorch (CPU + CUDA)
  - sets deterministic cuDNN flags
- **Theory**:
  - diffusion generation is stochastic; fixed seeds make results more repeatable.
- **Trade-off**: deterministic cuDNN may reduce speed slightly.

</details>

<details>
<summary><b>Cell 009 (markdown)</b> — “Configuration and Hyperparameters” heading</summary>

- **Purpose**: centralizes all numeric settings so the experiment is auditable.

</details>

<details>
<summary><b>Cell 010 (code)</b> — `Config` class (all hyperparameters)</summary>

- **What it defines (key values)**:
  - base model: `runwayml/stable-diffusion-v1-5`
  - resolution: 512
  - LoRA: rank=4, alpha=4, target modules in attention
  - training: batch=1, accum=4, epochs=20, LR=1e-4
  - data: train=50, val=10
  - inference: steps=50, guidance=7.5
  - output folders: `./data`, `./outputs`, `./lora_weights`
- **Theory**:
  - batch=1 + accumulation=4 is a memory strategy for Colab.
  - LoRA rank=4 is small but enables adaptation.

</details>

<details>
<summary><b>Cell 011 (markdown)</b> — Section 3 heading (Data)</summary>

- **Purpose**: begins dataset generation and preprocessing section.

</details>

<details>
<summary><b>Cell 012 (code)</b> — `TextImageGenerator` (synthetic dataset generator)</summary>

- **Goal**: simulate “crowd-sourced operational text images” using controlled randomness.
- **What it generates**:
  - 512×512 RGB images
  - centered text (font size 40–80, if font exists)
  - random background RGB in [0..255]
  - contrasting text color (black/white by brightness)
  - optional border + optional Gaussian noise \(\\mathcal{N}(0,5)\)
- **Why (theory)**:
  - variability helps generalization (domain shift simulation).
  - style words in prompt (sign/label/badge) teach contextual domain.

</details>

<details>
<summary><b>Cell 013 (markdown)</b> — “Create Training and Validation Datasets” heading</summary>

- **Purpose**: introduces train/val split creation.

</details>

<details>
<summary><b>Cell 014 (code)</b> — Generate 50 train + 10 val samples</summary>

- **What it does**:
  - creates generator with resolution 512
  - runs `generate_dataset(50)` and `generate_dataset(10)`
- **Theory**: small dataset demonstrates LoRA under limited-data conditions.
- **Output**: in-memory lists of dicts used in later steps.

</details>

<details>
<summary><b>Cell 015 (markdown)</b> — “Save Dataset to Disk” heading</summary>

- **Purpose**: ensures dataset is stored and reproducible for grading.

</details>

<details>
<summary><b>Cell 016 (code)</b> — Save images + `metadata.json`</summary>

- **What it does**:
  - saves images to `./data/<split>/images/{id:04d}.png`
  - writes `./data/<split>/metadata.json` with `id,image_path,text,prompt`
- **Why (theory)**:
  - separates data from code; anyone can inspect the dataset without rerunning generation.

</details>

<details>
<summary><b>Cell 017 (markdown)</b> — “Visualize Sample Images” heading</summary>

- **Purpose**: prompts a sanity-check step before training.

</details>

<details>
<summary><b>Cell 018 (code)</b> — Visualize dataset samples</summary>

- **What it does**:
  - plots sample train images in a grid
  - saves `./outputs/dataset_samples.png`
- **Why (theory)**:
  - visual data inspection prevents training on bad/blank images.

</details>

<details>
<summary><b>Cell 019 (markdown)</b> — “Custom Dataset Class for PyTorch” heading</summary>

- **Purpose**: introduces the Dataset wrapper that feeds tensors to the model.

</details>

<details>
<summary><b>Cell 020 (code)</b> — `TextImageDataset` (transforms + tokenization)</summary>

- **What it does**:
  - image → tensor (3,512,512)
  - normalize to [-1,1] using: \(x_{norm} = 2\\cdot(pixel/255) - 1\)
  - prompt → CLIP `input_ids` (typically length 77)
- **Theory**:
  - SD expects normalized images and tokenized prompts for text conditioning.

</details>

<details>
<summary><b>Cell 021 (markdown)</b> — Section 4 heading (Model Development)</summary>

- **Purpose**: begins SD loading + LoRA application.

</details>

<details>
<summary><b>Cell 022 (code)</b> — Load SD components (tokenizer, text encoder, VAE, UNet, scheduler)</summary>

- **What it does**: loads pretrained components from `runwayml/stable-diffusion-v1-5`.
- **Theory (how SD works)**:
  - CLIP encodes text → embeddings \(c\)
  - VAE encodes image → latent \(z\)
  - UNet predicts noise residual in latent space
- **Numerics**:
  - pixels: (B,3,512,512) → latents: (B,4,64,64) (downsample factor ~8)

</details>

<details>
<summary><b>Cell 023 (markdown)</b> — “Configure LoRA” heading</summary>

- **Purpose**: introduces parameter-efficient fine-tuning.

</details>

<details>
<summary><b>Cell 024 (code)</b> — Freeze base + apply LoRA adapters</summary>

- **What it does**:
  - freezes VAE + text encoder + UNet base
  - applies LoRA to `to_q,to_k,to_v,to_out.0`
  - prints trainable params (797,184) and percent (~0.0927%)
- **Theory**:
  - these modules implement cross-attention projections; adapting them changes prompt-image alignment.

</details>

<details>
<summary><b>Cell 025 (markdown)</b> — “Move Models to Device” heading</summary>

- **Purpose**: prepares models for GPU + mixed precision.

</details>

<details>
<summary><b>Cell 026 (code)</b> — Move models to CUDA + fp16 setup</summary>

- **What it does**:
  - `.to(cuda)` for VAE/text encoder/UNet
  - casts VAE + text encoder to float16 (fp16)
  - sets eval mode for frozen parts, train mode for UNet
- **Theory**: fp16 reduces VRAM and speeds up training on Colab GPUs.

</details>

<details>
<summary><b>Cell 027 (markdown)</b> — Section 5 heading (Training)</summary>

- **Purpose**: begins the training pipeline section.

</details>

<details>
<summary><b>Cell 028 (code)</b> — Create DataLoaders</summary>

- **What it does**:
  - wraps lists into `TextImageDataset`
  - makes train loader (shuffle) and val loader
  - prints batch counts (50 train, 10 val)
- **Theory**:
  - each epoch processes all train samples once (batch size 1).

</details>

<details>
<summary><b>Cell 029 (markdown)</b> — “Setup Optimizer and Scheduler” heading</summary>

- **Purpose**: introduces optimization configuration for LoRA parameters only.

</details>

<details>
<summary><b>Cell 030 (code)</b> — AdamW + LR scheduler (LoRA params only)</summary>

- **What it does**:
  - filters trainable parameters (LoRA)
  - sets AdamW with weight decay
  - configures scheduler (constant_with_warmup)
- **Theory**: AdamW is standard for diffusion fine-tuning; scheduler stabilizes LR.

</details>

<details>
<summary><b>Cell 031 (markdown)</b> — “Initialize Accelerator” heading</summary>

- **Purpose**: introduces mixed precision + gradient accumulation support.

</details>

<details>
<summary><b>Cell 032 (code)</b> — `Accelerator()` prepare</summary>

- **What it does**:
  - sets `gradient_accumulation_steps=4`
  - sets mixed precision mode (fp16 if CUDA)
  - wraps UNet/optimizer/dataloader/scheduler
- **Theory**:
  - accumulation simulates batch size 4 without VRAM of batch 4.

</details>

<details>
<summary><b>Cell 033 (markdown)</b> — “Training Loop” heading</summary>

- **Purpose**: introduces the main diffusion training function.

</details>

<details>
<summary><b>Cell 034 (code)</b> — `train_one_epoch(...)` (core diffusion training)</summary>

- **What it does (step-by-step)**:
  1. VAE encodes image → latents (B,4,64,64)
  2. sample noise + timestep \(t\)
  3. add noise: \(z_t = \\sqrt{\\bar\\alpha_t}z_0 + \\sqrt{1-\\bar\\alpha_t}\\epsilon\)
  4. get CLIP text embeddings (B,77,768)
  5. UNet predicts noise residual
  6. compute MSE loss and backprop through LoRA
- **Why it works (theory)**:
  - diffusion training teaches the UNet to denoise conditioned on prompt embeddings.
- **Numerics**:
  - timesteps sampled from `[0, num_train_timesteps)` (often 1000)

```text
LOSS = MSE(model_pred, target_noise)
target_noise = noise  (epsilon prediction case)
```

</details>

<details>
<summary><b>Cell 035 (markdown)</b> — “Execute Training” heading</summary>

- **Purpose**: introduces the actual training run and checkpointing logic.

</details>

<details>
<summary><b>Cell 036 (code)</b> — Run epochs + save best model/checkpoints</summary>

- **What it does**:
  - runs 20 epochs
  - tracks loss + learning rate history
  - saves best LoRA adapter to `./lora_weights/best_model`
  - checkpoints every 10 epochs
  - cleans memory periodically
- **Theory**:
  - “best model” saving prevents losing the best adapter if later epochs degrade.

</details>

<details>
<summary><b>Cell 037 (markdown)</b> — “Plot Training Curves” heading</summary>

- **Purpose**: introduces visual proof of training convergence.

</details>

<details>
<summary><b>Cell 038 (code)</b> — Plot loss + LR curves</summary>

- **What it does**:
  - plots training loss and learning rate over epochs
  - saves `./outputs/training_curves.png`
- **Theory**:
  - decreasing loss indicates learning; spikes indicate instability.

</details>

<details>
<summary><b>Cell 039 (markdown)</b> — Section 6 heading (Evaluation Metrics)</summary>

- **Purpose**: begins inference + OCR evaluation.

</details>

<details>
<summary><b>Cell 040 (code)</b> — Load pipeline + load LoRA adapter for inference</summary>

- **What it does**:
  - loads base StableDiffusionPipeline
  - loads UNet and attaches LoRA adapter weights from `./lora_weights/best_model`
  - disables safety checker for simplicity
- **Theory**:
  - inference uses the same SD components, but UNet attention is adapted by LoRA.

</details>

<details>
<summary><b>Cell 041 (markdown)</b> — “Generate Images for Evaluation” heading</summary>

- **Purpose**: introduces sample generation for validation prompts.

</details>

<details>
<summary><b>Cell 042 (code)</b> — `generate_images(...)` (validation generation)</summary>

- **What it does**:
  - generates images for val prompts
  - uses steps=50, guidance=7.5, fixed seed=42
- **Theory**:
  - CFG (classifier-free guidance) trades prompt adherence vs artifacts.

</details>

<details>
<summary><b>Cell 043 (markdown)</b> — “OCR-based Evaluation” heading</summary>

- **Purpose**: introduces OCR functions and readability metrics.

</details>

<details>
<summary><b>Cell 044 (code)</b> — OCR + metric functions</summary>

- **What it does**:
  - preprocess image: grayscale → contrast → Otsu threshold
  - run Tesseract OCR (`--psm 6`)
  - metrics: exact match + character accuracy
- **Theory**:
  - thresholding improves separation of strokes vs background for OCR.

</details>

<details>
<summary><b>Cell 045 (markdown)</b> — “Perform Evaluation” heading</summary>

- **Purpose**: introduces evaluation loop across generated images.

</details>

<details>
<summary><b>Cell 046 (code)</b> — Evaluate generated images + save CSV</summary>

- **What it does**:
  - loops through 10 generated images
  - computes OCR text + exact match + char accuracy per sample
  - prints per-sample results
  - saves `./outputs/evaluation_results.csv`
- **Theory**:
  - exact match is strict; char accuracy captures partial correctness.

</details>

<details>
<summary><b>Cell 047 (markdown)</b> — “Visualize Evaluation Results” heading</summary>

- **Purpose**: introduces metric visualization and best/worst sample inspection.

</details>

<details>
<summary><b>Cell 048 (code)</b> — Plot evaluation metrics</summary>

- **What it does**:
  - visualizes distribution and best/worst samples (as implemented in notebook)
  - saves `./outputs/evaluation_metrics.png`

</details>

<details>
<summary><b>Cell 049 (markdown)</b> — Section 7 heading (Results & Analysis)</summary>

- **Purpose**: begins qualitative visualization and interpretation.

</details>

<details>
<summary><b>Cell 050 (code)</b> — Visualize generated images with OCR overlays</summary>

- **What it does**:
  - shows generated image + ground truth + OCR text
  - saves `./outputs/generated_images_with_ocr.png`
- **Theory**:
  - qualitative evidence complements OCR metrics (important when OCR is harsh).

</details>

<details>
<summary><b>Cell 051 (markdown)</b> — “Compare Before and After Fine-tuning” heading</summary>

- **Purpose**: introduces baseline comparison (base SD vs LoRA fine-tuned).

</details>

<details>
<summary><b>Cell 052 (code)</b> — Load base model (without LoRA) for comparison</summary>

- **What it does**:
  - loads a base StableDiffusionPipeline without adapter weights
  - generates baseline images for same prompts
- **Theory**:
  - comparison isolates the effect of LoRA fine-tuning.

</details>

<details>
<summary><b>Cell 053 (code)</b> — Visualize base vs fine-tuned side-by-side</summary>

- **What it does**:
  - plots base outputs vs fine-tuned outputs
  - saves `./outputs/before_after_comparison.png`
- **What to look for**:
  - improved text placement/clarity after fine-tuning.

</details>

<details>
<summary><b>Cell 054 (markdown)</b> — “Qualitative Analysis” heading</summary>

- **Purpose**: introduces written reasoning section (important for marks).

</details>

<details>
<summary><b>Cell 055 (code)</b> — Print qualitative analysis</summary>

- **What it does**:
  - prints: model performance, domain adaptation, challenges, LoRA efficiency, applications
- **Theory**:
  - connects observed results to causes (data size, OCR sensitivity, diffusion behavior).

</details>

<details>
<summary><b>Cell 056 (markdown)</b> — “Save Final Summary Report” heading</summary>

- **Purpose**: introduces writing the final executive summary.

</details>

<details>
<summary><b>Cell 057 (code)</b> — Create + save `summary_report.txt`</summary>

- **What it does**:
  - formats an executive report (metrics + config + loss)
  - saves `./outputs/summary_report.txt`
- **Why it matters**:
  - professors can grade quickly using one file.

</details>

<details>
<summary><b>Cell 058 (markdown)</b> — Section 8 heading (Conclusion)</summary>

- **Purpose**: final wrap-up section of the assignment report.

</details>

<details>
<summary><b>Cell 059 (code)</b> — Print “Project Completion Summary”</summary>

- **What it does**:
  - prints checklist of completed tasks
  - prints paths to deliverables (weights, plots, CSV, report)
- **Theory**:
  - auditable output list supports exam-style submission quality.

</details>

<details>
<summary><b>Cell 060 (markdown)</b> — Bonus: custom prompts heading</summary>

- **Purpose**: optional extension to test generalization on new prompts.

</details>

<details>
<summary><b>Cell 061 (code)</b> — Generate custom test images</summary>

- **What it does**:
  - generates images for custom prompts (not in validation set)
  - saves `./outputs/custom_test_images.png`
- **Theory**:
  - shows whether the fine-tuned adapter generalizes beyond the tiny val set.

</details>

<details>
<summary><b>Cell 062 (code)</b> — Print team member contributions</summary>

- **Purpose**: documentation cell for group submissions (who did what).

</details>

<details>
<summary><b>Cell 063 (code)</b> — Import datetime</summary>

- **Purpose**: used for timestamps in reports (e.g., summary report generation time).

</details>

<details>
<summary><b>Cell 064 (markdown)</b> — End of Assignment</summary>

- **Purpose**: explicit end marker; useful when exporting PDF/printing.

</details>

> After these per-cell explanations, the next section continues with a more narrative (section-by-section) explanation for readability.

### A1. Title + Introduction

- **Cell 1 (Markdown) — Title + Table of Contents**
  - **What it contains**: assignment title, problem statement number, and a ToC linking to sections 1–8.
  - **Why it matters**: shows the work is structured like a report (good for exam presentation).

- **Cell 2 (Markdown) — Introduction**
  - **What it contains**: problem statement, objectives, and approach (SD v1.5 + LoRA + OCR metrics).
  - **Theory**:
    - Text rendering is a **domain shift** for diffusion models.
    - LoRA is used because full fine-tuning is too expensive.
    - OCR metrics provide an objective readability proxy.

---

### A2. Section 2 — Environment Setup

- **Cell 2.1 (Markdown) — “Check GPU Availability”**
  - **Purpose**: informs that training needs GPU.

- **Cell 2.1 (Code) — CUDA/GPU check**
  - **What it does**:
    - prints Python version, PyTorch version
    - checks `torch.cuda.is_available()`
    - prints CUDA version + GPU name + total VRAM
  - **Theory**:
    - Diffusion fine-tuning is compute-heavy; without GPU it is impractical.
    - VRAM determines feasible batch size/resolution.
  - **Expected output**: e.g., Tesla T4, ~15–16GB VRAM.

- **Cell 2.2 (Markdown) — “Install Required Libraries”**
  - **Purpose**: lists packages needed for SD + LoRA + OCR.

- **Cell 2.2 (Code) — pip/apt installs**
  - **What it does**:
    - installs `diffusers transformers accelerate peft`
    - installs OCR + imaging libs (`pytesseract`, `opencv`, `pillow`, etc.)
    - installs `bitsandbytes`, `xformers` for efficiency
    - pins `numpy==1.24.4` (binary compatibility fix)
    - installs system tesseract (`apt-get install tesseract-ocr`)
  - **Theory**:
    - `diffusers` provides SD components and schedulers.
    - `peft` provides LoRA injection and adapter saving/loading.
    - OCR requires a system-level tesseract binary.

- **Cell 2.3 (Markdown) — “Import Libraries”**
  - **Purpose**: begins the “implementation” part (exam-style).

- **Cell 2.3 (Code) — imports**
  - **What it does**:
    - imports PyTorch + torchvision transforms
    - imports diffusers models (UNet, VAE, scheduler, pipeline)
    - imports `LoraConfig`, `get_peft_model`
    - imports OCR + plotting libs
  - **Theory**:
    - Stable Diffusion training uses (Text Encoder + UNet + VAE + Scheduler).
    - Evaluation uses OCR + matplotlib for quantitative/qualitative outputs.

- **Cell 2.4 (Code) — Set random seed**
  - **What it does**:
    - seeds Python `random`, NumPy, PyTorch (CPU + CUDA)
    - sets deterministic cuDNN flags
  - **Theory**:
    - reduces randomness so training curves and generated images are reproducible.
    - determinism can reduce speed slightly but improves repeatability.

- **Cell 2.5 (Markdown) — “Configuration and Hyperparameters”**
  - **Purpose**: central place for all numeric settings (good exam practice).

- **Cell 2.5 (Code) — `Config` class**
  - **What it does**:
    - defines SD model ID, resolution, LoRA rank/alpha, training params
    - defines dataset sizes (50 train, 10 val)
    - defines inference params (steps=50, guidance=7.5)
    - creates directories (`./data`, `./outputs`, `./lora_weights`)
  - **Theory (why these numbers)**:
    - 512 resolution is standard for SD v1.5.
    - batch size 1 + gradient accumulation 4 is a Colab memory strategy.
    - LoRA rank 4 is small enough to fit, but still allows adaptation.

---

### A3. Section 3 — Data Preprocessing & Augmentation

- **Cell 3.1 (Markdown) — “Generate Synthetic Text-Image Dataset”**
  - **Purpose**: describes why synthetic data is used to simulate crowd-sourced pairs.

- **Cell 3.1 (Code) — `TextImageGenerator` class**
  - **What it does**:
    - builds a list of 57 operational texts (STOP/EXIT/etc.)
    - generates random background + contrasting text color
    - draws centered text (font size 40–80)
    - optional sign border and optional Gaussian noise \( \mathcal{N}(0,5) \)
    - constructs prompt: `a <style> image with the text '<TEXT>' in clear, readable font`
  - **Theory**:
    - introduces controlled variation (color, noise, style) to mimic real-world photos.
    - prompt style words help the model learn domain context.
  - **Numerics**:
    - output images: 512×512 RGB
    - font: DejaVuSans-Bold (if available)

- **Cell 3.2 (Markdown) — “Create Training and Validation Datasets”**
  - **Purpose**: defines split sizes for training/testing.

- **Cell 3.2 (Code) — generate datasets**
  - **What it does**:
    - creates generator at `(512,512)`
    - generates 50 train + 10 val samples
  - **Theory**:
    - small val set is enough to demonstrate OCR evaluation pipeline.
    - small train set demonstrates LoRA under low-data constraints.

- **Cell 3.3 (Markdown) — “Save Dataset to Disk”**
  - **Purpose**: ensures reproducibility and easy grading.

- **Cell 3.3 (Code) — `save_dataset()`**
  - **What it does**:
    - saves each image to `./data/<split>/images/{id:04d}.png`
    - writes `metadata.json` with `id`, `image_path`, `text`, `prompt`
  - **Theory**:
    - clear dataset structure is essential for exam submissions.
    - `metadata.json` makes the dataset readable without code.

- **Cell 3.4 (Markdown) — “Visualize Sample Images”**
  - **Purpose**: sanity-check data quality before training.

- **Cell 3.4 (Code) — visualization**
  - **What it does**:
    - plots multiple samples in a grid
    - saves `./outputs/dataset_samples.png`
  - **Theory**:
    - visual inspection is critical; garbage-in → garbage-out.

- **Cell 3.5 (Markdown) — “Custom Dataset Class for PyTorch”**
  - **Purpose**: converts stored samples into tensors + tokenized prompts.

- **Cell 3.5 (Code) — `TextImageDataset(Dataset)`**
  - **What it does**:
    - transforms image → tensor (3,512,512), normalized to [-1,1]
    - tokenizes prompt → `input_ids` (shape (L,), typically L=77)
    - returns dict: `pixel_values`, `input_ids`, plus `text/prompt` for reporting
  - **Theory**:
    - Stable Diffusion training expects normalized image tensors and tokenized prompts.
    - Normalization: \(x_{norm} = 2(\text{pixel}/255) - 1\).

---

### A4. Section 4 — Model Development (Stable Diffusion + LoRA)

- **Cell 4.1 (Markdown) — “Load Pre-trained Stable Diffusion Model”**
  - **Purpose**: loads SD pipeline components separately for training control.

- **Cell 4.1 (Code) — load tokenizer/text encoder/VAE/UNet/scheduler**
  - **What it does**:
    - loads from `runwayml/stable-diffusion-v1-5` subfolders
  - **Theory**:
    - Text encoder produces embeddings \(c\).
    - VAE maps image → latent \(z\).
    - UNet predicts noise residual.
    - Scheduler defines diffusion timesteps.
  - **Numerics**:
    - pixels: (B,3,512,512)
    - latents: (B,4,64,64) because downsample factor ≈ 8.

- **Cell 4.2 (Markdown) — “Configure LoRA for Fine-tuning”**
  - **Purpose**: parameter-efficient adaptation plan.

- **Cell 4.2 (Code) — freeze base + apply LoRA**
  - **What it does**:
    - freezes VAE/text encoder/UNet base weights
    - creates LoRA config (r=4, alpha=4, target attention projections)
    - wraps UNet with PEFT LoRA and prints trainable param count
  - **Theory**:
    - LoRA updates attention projections \(W_Q,W_K,W_V,W_O\) so the model learns to “attend” to text tokens for rendering.
    - Cross-attention math:
      \( \text{softmax}(QK^T/\sqrt{d})V \)

- **Cell 4.3 (Markdown) — “Move Models to Device”**
  - **Purpose**: GPU + fp16 setup.

- **Cell 4.3 (Code) — `.to(device)` + fp16**
  - **What it does**:
    - moves models to CUDA
    - casts VAE/text encoder to fp16
    - sets eval/train modes
  - **Theory**:
    - fp16 reduces VRAM and speeds up training.

---

### A5. Section 5 — Training

- **Cell 5.1 (Markdown) — “Prepare Data Loaders”**
  - **Purpose**: build iterators for training.

- **Cell 5.1 (Code) — dataloaders**
  - **What it does**:
    - wraps data into `TextImageDataset`
    - makes DataLoaders (shuffle for train)
    - prints 50 train batches, 10 val batches
  - **Theory**:
    - each epoch processes 50 samples (batch size 1).

- **Cell 5.2 (Markdown) — “Setup Optimizer and Scheduler”**
  - **Purpose**: configure training updates for LoRA only.

- **Cell 5.2 (Code) — AdamW + scheduler**
  - **What it does**:
    - optimizes only parameters with `requires_grad=True` (LoRA adapters)
    - uses AdamW with weight decay
    - uses constant-with-warmup scheduler
  - **Theory**:
    - AdamW is standard for diffusion fine-tuning.

- **Cell 5.3 (Markdown) — “Initialize Accelerator”**
  - **Purpose**: mixed precision + gradient accumulation wrapper.

- **Cell 5.3 (Code) — `Accelerator(...)`**
  - **What it does**:
    - sets `gradient_accumulation_steps=4`
    - prepares UNet/optimizer/dataloader/scheduler
  - **Theory**:
    - accumulation simulates batch size 4 with memory of batch size 1.

- **Cell 5.4 (Markdown) — “Training Loop”**
  - **Purpose**: define core diffusion training step.

- **Cell 5.4 (Code) — `train_one_epoch(...)`**
  - **What it does (step-by-step)**:
    1. image → latent via VAE
    2. sample noise and timestep
    3. add noise to latent
    4. get text embeddings via CLIP
    5. UNet predicts noise residual
    6. compute MSE loss
    7. backprop through LoRA
    8. clip grads, optimizer + scheduler step
  - **Theory (diffusion equation)**:
    \( z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon \)

- **Cell 5.5 (Markdown) — “Execute Training”**
  - **Purpose**: run epochs, save best model.

- **Cell 5.5 (Code) — training driver**
  - **What it does**:
    - runs epochs 1..20
    - saves best adapter to `./lora_weights/best_model`
    - checkpoints every 10 epochs
    - logs training history and cleans GPU memory periodically
  - **Theory**:
    - best-loss saving prevents losing the best adapter due to later instability.

- **Cell 5.6 (Markdown) — “Plot Training Curves”**
  - **Purpose**: show learning trend.

- **Cell 5.6 (Code) — plot loss + LR**
  - **What it does**:
    - saves `./outputs/training_curves.png`
  - **Theory**:
    - decreasing loss indicates adaptation is happening.

---

### A6. Section 6 — Evaluation Metrics

- **Cell 6.1 (Markdown) — “Load Best Model for Inference”**
  - **Purpose**: load base pipeline + apply LoRA adapter for generation.

- **Cell 6.1 (Code) — pipeline load + adapter load**
  - **What it does**:
    - loads `StableDiffusionPipeline`
    - loads UNet and applies LoRA adapter weights
  - **Theory**:
    - inference uses the same SD components but with adapted UNet attention.

- **Cell 6.2 (Markdown) — “Generate Images for Evaluation”**
  - **Purpose**: generate val images.

- **Cell 6.2 (Code) — `generate_images(...)`**
  - **What it does**:
    - generates 10 images using:
      - inference steps = 50
      - guidance scale = 7.5
      - fixed seed = 42
  - **Theory**:
    - CFG formula: `pred = uncond + s*(cond-uncond)` controls prompt adherence.

- **Cell 6.3 (Markdown) — “OCR-based Evaluation”**
  - **Purpose**: define readability measurement.

- **Cell 6.3 (Code) — OCR + metric functions**
  - **What it does**:
    - preprocess image for OCR (grayscale, contrast, threshold)
    - extracts text via Tesseract
    - computes exact match + character accuracy
  - **Theory**:
    - OCR preprocessing increases separation of text strokes from background.

- **Cell 6.4 (Markdown) — “Perform Evaluation”**
  - **Purpose**: run OCR and compute dataset statistics.

- **Cell 6.4 (Code) — evaluation loop + CSV**
  - **What it does**:
    - loops over 10 generated images
    - prints per-sample OCR results
    - saves `./outputs/evaluation_results.csv`
  - **Theory**:
    - exact match is strict; char accuracy provides softer correctness measure.

- **Cell 6.5 (Markdown) — “Visualize Evaluation Results”**
  - **Purpose**: show distribution and best/worst cases.

- **Cell 6.5 (Code) — evaluation plots**
  - **What it does**:
    - saves `./outputs/evaluation_metrics.png`

---

### A7. Section 7 — Results & Analysis

- **Cell 7.1 (Markdown) — “Visualize Generated Images with OCR Results”**
  - **Purpose**: qualitative evaluation.

- **Cell 7.1 (Code) — image grid with OCR**
  - **What it does**:
    - saves `./outputs/generated_images_with_ocr.png`
  - **Theory**:
    - lets a human grader verify whether text is visually readable even if OCR fails.

- **Cell 7.2 (Markdown) — “Compare Before and After Fine-tuning”**
  - **Purpose**: show improvement vs base model.

- **Cell 7.2 (Code) — base vs LoRA comparison**
  - **What it does**:
    - generates images with base pipeline (no LoRA)
    - generates images with fine-tuned pipeline
    - saves side-by-side figure: `./outputs/before_after_comparison.png`

- **Cell 7.3 (Markdown) — “Qualitative Analysis”**
  - **Purpose**: written reasoning for exam marks.

- **Cell 7.3 (Code) — printed analysis**
  - **What it does**:
    - prints model performance, domain adaptation, challenges, LoRA efficiency, applications
  - **Theory**:
    - connects observations to reasons (data size, OCR sensitivity, LoRA efficiency).

- **Cell 7.4 (Markdown) — “Save Final Summary Report”**
  - **Purpose**: create one-page exam report automatically.

- **Cell 7.4 (Code) — summary report writer**
  - **What it does**:
    - composes a formatted report (results + config + loss)
    - saves `./outputs/summary_report.txt`

---

### A8. Section 8 — Conclusion

- **Cell 8.1 (Markdown) — “Project Summary”**
  - **Purpose**: final closure section.

- **Cell 8.1 (Code) — completion summary**
  - **What it does**:
    - prints completed tasks checklist
    - prints deliverables paths (weights, plots, CSV, report)
  - **Theory**:
    - a clean, auditable “what was produced” list helps grading.

### A9. (Optional) Custom prompt testing cell (if executed)
The notebook also generates images for custom prompts and saves:
- `./outputs/custom_test_images.png`

**Purpose**: demonstrate generalization on new text prompts beyond the small validation set.

