# 📝 Text Summarization Web App (T5 + FastAPI)

An end-to-end **NLP text summarization application** built using a **fine-tuned T5 transformer model**, served through a **FastAPI backend** and integrated with a **simple frontend UI** for real-time inference.

---

## 🚀 Project Overview

This project focuses on **dialogue summarization**, where long conversational text is converted into a short, meaningful summary.

### 🔹 What this project does
- Fine-tunes a **T5 model** on the **SAMSum dialogue summarization dataset**
- Exposes the trained model via a **FastAPI REST API**
- Provides a **frontend UI** (HTML/CSS/JS) to interact with the model
- Generates summaries in **real time**

---

## 🧠 Model Details

- **Model:** T5-Small (Transformer-based sequence-to-sequence model)
- **Dataset:** SAMSum (dialogue → summary pairs)
- **Task:** Abstractive Text Summarization
- **Frameworks:** Hugging Face Transformers, PyTorch

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI
- Hugging Face Transformers
- PyTorch

### Frontend
- HTML
- CSS
- JavaScript (Fetch API)

### ML / NLP
- T5 (Text-to-Text Transformer)
- SentencePiece tokenizer
- Fine-tuned on SAMSum dataset

---

## 📂 Project Structure
```
text_summarization_project/
├── app.py                 # FastAPI backend
├── static/
│   └── index.html         # Frontend UI
├── t5_samsum/             # Fine-tuned model & tokenizer
├── venv/                  # Virtual environment
└── README.md
```

---

## ⚙️ How It Works

1. User enters dialogue text in the UI
2. Frontend sends a POST request to `/summarize`
3. FastAPI backend:
   - Tokenizes input text
   - Runs inference using the fine-tuned T5 model
   - Decodes generated tokens
4. Summary is returned and displayed in the UI

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone <your-github-repo-url>
cd text_summarization_project
```

### 2️⃣ Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies
```bash
pip install fastapi uvicorn transformers torch sentencepiece protobuf tiktoken
```

### 4️⃣ Start the server
```bash
uvicorn app:app --reload
```

### 5️⃣ Open in browser
```
http://127.0.0.1:8000
```

---

## 🧪 Example Input
```
Amanda: Are we meeting tomorrow?
John: Yes, at 10 AM.
Amanda: Can you pick me up?
John: Sure, I will be there by 9:45.
```

### ✅ Output
```
Amanda and John are meeting tomorrow at 10 AM. John will pick Amanda up by 9:45.
```

---

## 📊 Training Summary

- **Epochs:** 2
- **Final Training Loss:** ~0.41
- **Validation Loss:** ~0.35
- **Hardware:** Kaggle GPU (Tesla P100)

---

## 💡 Key Learnings

- Fine-tuning transformer models for NLP tasks
- Efficient preprocessing and tokenization
- Serving ML models using FastAPI
- Handling CORS and frontend-backend integration
- Building production-style ML applications

---

## 📌 Future Improvements

- Add authentication
- Improve UI design
- Deploy on Hugging Face Spaces
- Add ROUGE score visualization
- Support long-document summarization

---

## 👤 Author

**Siddharth Kumar Pandey**  
BTech CSE (AI/ML)  
Aspiring Machine Learning Engineer

---

## ⭐ Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SAMSum Dataset](https://huggingface.co/datasets/samsum)
- FastAPI community
```

---

**Commit message to use:**
```
📝 Add comprehensive README documentation
