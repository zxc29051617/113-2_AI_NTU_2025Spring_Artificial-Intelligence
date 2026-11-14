# NTU Artificial Intelligence 2025 Spring

這個 repository 收錄臺大 113-2 人工智慧（鄭文皇老師）課程的三份作業與期末專題。

作業內容圍繞大型語言模型的應用：多模態模型的圖像描述與風格轉換、檢索增強生成（RAG），以及多代理人系統。

## 專案結構

```text
.
├── README.md
├── hw1/                 # 多模態大型語言模型：圖像描述評估與風格轉換
├── hw2/                 # 檢索增強生成（RAG）
├── hw3/                 # 多代理人餐廳評分系統（AutoGen）
└── AI_finalproject/     # 期末專題：用 UniMuMo 評估音樂與舞蹈風格的主觀感受，以及節拍對齊實驗
```

## HW1：多模態大型語言模型

題目說明：[`hw1/ai_hw1.pdf`](hw1/ai_hw1.pdf)  
報告：[`hw1/hw1_R13945041/hw1_R13945041.pdf`](hw1/hw1_R13945041/hw1_R13945041.pdf)

- Task 1：用 BLIP 和 Phi-4 為 MSCOCO、Flickr30k 的圖片產生描述，以 BLEU、ROUGE、METEOR 評估。
- Task 2-1：用 Phi-4 產生 Snoopy 風格的描述，再用 Stable Diffusion 3 生成風格化圖片。
- Task 2-2：同樣的流程，改用 Stable Diffusion v1.5 做 image-to-image。

程式是三個 Colab notebook，放在 `hw1/hw1_R13945041/hw1_R13945041_code.zip`，產生的圖片在
`hw1_R13945041_stylized_images.zip`。執行方式見 [`hw1/hw1_R13945041/README.md`](hw1/hw1_R13945041/README.md)。

## HW2：檢索增強生成（RAG）

題目說明：[`hw2/ai_hw2.pdf`](hw2/ai_hw2.pdf)  
繳交檔案：[`hw2/hw2_R13945041.zip`](hw2/hw2_R13945041.zip)（報告、notebook、說明）

- Task 1：實作 RAG：解析 PDF 文件、建立向量資料庫，檢索相關段落後交給語言模型回答。
- Task 2：RAG-based Page Retrieval，找出問題對應的 PDF 頁面（Kaggle 競賽）。

## HW3：多代理人系統

題目說明：[`hw3/AI_hw3.pptx.pdf`](hw3/AI_hw3.pptx.pdf)  
繳交檔案：[`hw3/hw3_r13945041.zip`](hw3/hw3_r13945041.zip)（報告、程式、說明）

用 AutoGen（pyautogen 0.9.0）和 GPT-4o-mini 建立多個代理人，根據文字評論替餐廳評分。

- `hw3_0508_original/`：助教提供的原始程式
- `hw3_0508/`：完成後的程式，`python test.py` 執行公開測資
- `test_all_past/`：測試用的腳本

執行前需要設定 `OPENAI_API_KEY` 環境變數。

## 期末專題：音樂與舞蹈風格的主觀感受

第 25 組，組員：Ke-Yang Wu、Chih-Yang Hsiao、Cai-jie Fan、Yu-Shan Lin、Kai-Yuan Cheng、Hsien-Ting Wang。

論文：[`AI_finalproject/final_25/final_25.pdf`](AI_finalproject/final_25/final_25.pdf)  
簡報：[`AI_finalproject/AI Report.pptx`](<AI_finalproject/AI Report.pptx>)

使用 [UniMuMo](https://github.com/hanyangclarence/UniMuMo) 從文字產生音樂與舞蹈動作，比較不同組合產生的結果，評估人們對音樂與舞蹈風格的主觀感受。
各實驗的執行步驟見 [`AI_finalproject/final_25/README.md`](AI_finalproject/final_25/README.md)。

節拍對齊實驗：[`AI_finalproject/beat_rerank/`](AI_finalproject/beat_rerank)。重現專題的評估（UniMuMo、CLAP、
曲風分類器、BAS），加入對照組、真人舞者參考與多個 seed，並用 16 選 1 的候選挑選和節拍重定時，
讓 UniMuMo 生成的舞步對上音樂的拍子。
