
# HW1 - Visual Captioning & Stylized Image Generation

## 👨‍🔬 Author

- 姓名：王獻霆  
- 學號：R13945041

---

## 🧩 Tasks Overview

- **Task 1**: Compare BLIP and Phi-4 on MSCOCO / Flickr30k using captioning + evaluation (BLEU/ROUGE/METEOR)
- **Task 2-1**: Use Phi-4 to generate "Snoopy-style" captions → use SD3 to synthesize stylized images
- **Task 2-2**: Same as Task 2-1, but use SD v1.5 instead of SD3

---

## 🧪 Environment Details

- Platform: Google Colab  
- Python: 3.10+  
- Runtime: GPU（A100 / L4 ）
- Run each Cell in order
### 🔧 Required Packages

- torch==2.6.0
- torchvision==0.21.0
- torchaudio==2.6.0
- transformers==4.48.2
- accelerate==1.3.0
- diffusers
- evaluate
- datasets
- flash-attn==2.7.4.post1
- peft==0.13.2
- pillow==11.1.0
- soundfile, scipy, backoff, tqdm 等

> 所有套件皆已於 notebook 中安裝（使用 `!pip install`），無需額外安裝。
> 安裝環境與登入 Hugging Face：
   - 依照開頭 Cell 執行 pip 安裝。
   - 並於 `login("hf_...")` 處貼上 Hugging Face Token 以存取模型。
---

## 🚀 How to Run the Code

### 🟠 Task 1 - Captioning & Evaluation

1. 使用以下資料集：
   - MSCOCO (5k test) ➜ `ds_mscoco_Test`
   - Flickr30k ➜ `ds_flicker30k`

2. 使用模型：
   - `BLIP (Salesforce/blip-image-captioning-base)`
   - `Phi-4 (microsoft/Phi-4-multimodal-instruct)`

3. 每張圖像皆使用兩模型產生描述 → 進行比較與評估：
   - BLEU, ROUGE, METEOR 分數透過 `evaluate` 套件計算
   - 用 `compare_models(index)` 可視覺化模型對照與 Ground Truth

---

### 🟡 Task 2-1 - Snoopy Prompt + Stylization (SD3)

1. 從 `/content/drive/MyDrive/content_image/` 讀入原始圖片  
2. 使用 `Phi-4` 對每張圖片生成 comic-style caption  
3. 將 caption 傳入 `Stable Diffusion 3 Medium` 模型生成風格圖  
4. 儲存圖像為 224x224 JPEG 到：
   ```
   /content/drive/MyDrive/AI_tast2-1/snoopy_prompt/
   ```

5. 將 caption 對應結果儲存成：
   ```
   /content/drive/MyDrive/AI_tast2-1/snoopy_prompt.json
   ```

---

### 🟢 Task 2-2 - Stylization (SD v1.5)

1. 與 2-1 流程幾乎相同，但使用 `Stable Diffusion v1.5` 模型進行圖像生成  
2. 儲存資料夾如下：
   ```
   /content/drive/MyDrive/AI_tast2-2/snoopy_prompt/
   /content/drive/MyDrive/AI_tast2-2/snoopy_prompt.json
   ```

---

## 🗂️ Folder Structure (壓縮前)

```
hw1_<student-id>/
├── hw1_<student-id>.pdf                   # 報告 (Task 1 + 2-1 + 2-2)
├── hw1_<student-id>_stylized_images.zip   # Task 2-1 圖片輸出 (snoopy_prompt/*.jpg)
├── hw1_<student-id>_code.zip              # Notebook 與程式碼（包含三個任務）
└── README.md                              # 說明文件（本檔）
```

---
