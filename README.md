# Question-Answer Bot and Text Summarizer

A PDF-based Question-Answering and Summarization tool built with Hugging Face `transformers`. It fine-tunes a small causal language model (`distilgpt2`) on the text extracted from a user-supplied PDF, then lets you interactively ask questions about the document or generate a multi-page extractive summary of any topic within it.

## How It Works

1. **PDF Upload & Extraction** — The script uploads a PDF (via Google Colab's file picker) and extracts its text using a fallback chain of parsers: `pdfplumber` first, then `PyPDF2`, then `pypdf`, so extraction still succeeds if one library fails on a given file.
2. **Cleaning & Chunking** — Extracted text is normalized (whitespace/newlines collapsed) and split into ~200-word chunks.
3. **Fine-Tuning** — `distilgpt2` is fine-tuned on the chunks using Hugging Face `Trainer` with a causal language modeling objective (`DataCollatorForLanguageModeling`), so the model adapts to the vocabulary and content of the uploaded document.
4. **Retrieval** — A lightweight keyword-overlap retriever (`retrieve_relevant_text`) scores each chunk against the query/topic and returns the most relevant ones — a simple, dependency-free stand-in for a vector-based retriever.
5. **Summarization** — `summarize_topic` retrieves the most relevant chunks for a topic and concatenates them into an extractive summary of roughly 3 pages (~1800 words).
6. **Question Answering** — `answer_question` builds a prompt from the retrieved context and the question, then uses the fine-tuned model's `generate()` to produce an answer, with a safe-truncation fallback if generation fails on a long context.
7. **Interactive Menu** — Once training completes, the script drops into a console loop where you choose to (1) summarize a topic, (2) ask a question, or type `exit` to quit.

## Tech Stack

- Python
- PyTorch
- Hugging Face `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`, `Trainer`, `TrainingArguments`)
- Hugging Face `datasets`
- `pdfplumber`, `PyPDF2`, `pypdf` (PDF text extraction)
- Google Colab (`google.colab.files` for PDF upload)

## Project Structure

```
question-answer-bot-and-text-summarizer/
└── genai_project (1).py   # Full pipeline: PDF extraction, fine-tuning, retrieval, summarization, Q&A
```

The script was exported from a Google Colab notebook (`genai_project.ipynb`) and is intended to be run there, since it relies on `google.colab.files.upload()` for the PDF input and benefits from Colab's free GPU.

## Setup & Installation

Install the required Python packages:

```bash
pip install pdfplumber pypdf PyPDF2 transformers datasets accelerate torch
```

## How to Run

This script is designed to run in **Google Colab**:

1. Open a new Colab notebook and paste the contents of `genai_project (1).py` into a cell (or upload/open the file directly).
2. Run the cell. You will be prompted to upload a PDF file when `files.upload()` executes.
3. Wait for the fine-tuning step (`trainer.train()`) to complete — this trains `distilgpt2` for 1 epoch on the PDF's text chunks.
4. Once training finishes, use the interactive menu printed to the console:
   - `1` — enter a topic to get a ~3-page extractive summary.
   - `2` — enter a question to get an answer generated from the most relevant chunks of the PDF.
   - `exit` — stop the program.

**Note:** To run outside Colab, replace the `google.colab.files.upload()` call with a local file path, since that API is only available in the Colab environment.

## Notes

- The fine-tuned model and tokenizer are saved locally to a `model_out/` directory after training.
- GPU is used automatically if available (`torch.cuda.is_available()`); otherwise the script falls back to CPU, which will be significantly slower for training.
