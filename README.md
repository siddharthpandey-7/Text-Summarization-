# 📝 Text Summarization Web App (T5 + FastAPI)

An end-to-end **NLP text summarization application** built using a **fine-tuned T5 transformer model**, served through a **FastAPI backend** and integrated with a **simple frontend UI** for real-time inference.  
The application is **deployed for free on Hugging Face Spaces**.

---

## 🚀 Project Overview

This project focuses on **dialogue summarization**, where long conversational text is converted into a short, meaningful summary using a transformer-based deep learning model.

### 🔹 What this project does
- Fine-tunes a **T5 (Text-to-Text Transformer)** model on the **SAMSum dialogue summarization dataset**
- Uploads the trained model to the **Hugging Face Model Hub**
- Serves the model via a **FastAPI REST API**
- Provides a **frontend UI** (HTML/CSS/JavaScript) for user interaction
- Generates summaries in **real time**
- Fully deployed on **Hugging Face Spaces**

---

## 🧠 Model Details

- **Model:** T5-Small (Transformer-based sequence-to-sequence model)
- **Dataset:** SAMSum (dialogue → summary pairs)
- **Task:** Abstractive Text Summarization
- **Frameworks:** Hugging Face Transformers, PyTorch

📌 **Model hosted on Hugging Face Hub:**  
[siddharthpandey7/t5-samsum-summarizer](https://huggingface.co/siddharthpandey7/t5-samsum-summarizer)

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

### Deployment
- Hugging Face Spaces
- Docker

---

## 📂 Project Structure
```
text_summarization_project/
├── app.py                 # FastAPI backend
├── static/
│   └── index.html         # Frontend UI
├── requirements.txt       # Python dependencies
├── Dockerfile             # Deployment configuration
└── README.md
```

📌 **Note:**  
The trained model weights are **not stored in this repository**.  
They are loaded dynamically from the **Hugging Face Model Hub**.

---

## 🔽 Model Loading Strategy (Important)

To avoid GitHub file size limits and ensure scalable deployment:

- The trained T5 model is uploaded separately to **Hugging Face Hub**
- The FastAPI app loads the model directly from the hub at runtime
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "siddharthpandey7/t5-samsum-summarizer"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
```

---

## ⚙️ How It Works

1. User enters dialogue text in the web UI
2. Frontend sends a POST request to `/summarize`
3. FastAPI backend:
   - Tokenizes the input text
   - Runs inference using the fine-tuned T5 model
   - Decodes the generated tokens
4. The generated summary is returned as JSON
5. The frontend displays the summary to the user

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
pip install -r requirements.txt
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

## 🧪 Example

### Input
```
Amanda: Are we meeting tomorrow?
John: Yes, at 10 AM.
Amanda: Can you pick me up?
John: Sure, I will be there by 9:45.
```

### Output
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
- Working with dialogue summarization datasets
- Efficient tokenization and inference pipelines
- Serving ML models using FastAPI
- Frontend–backend integration
- Deploying ML applications using Docker and Hugging Face Spaces

---

## 📌 Future Improvements

- Add ROUGE score evaluation
- Improve UI/UX design
- Support long-document summarization
- Add user authentication
- Enable batch summarization

---

## 👤 Author

**Siddharth Kumar Pandey**  
BTech CSE (AI/ML)  
Aspiring Machine Learning Engineer

---

## ⭐ Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SAMSum Dataset](https://huggingface.co/datasets/samsum)
- [FastAPI](https://fastapi.tiangolo.com/)
```

---

**Commit message:**
```
📝 Add comprehensive README with deployment info
