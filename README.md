# 🧠 RAG Chatbot with LLaMA 3 & Gradio

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4.svg)](https://www.gradio.app/)
[![LLaMA 3](https://img.shields.io/badge/Model-LLaMA3-yellow.svg)](https://ai.meta.com/llama/)
[![RAG](https://img.shields.io/badge/Type-RAG-informational.svg)](https://huggingface.co/papers/2301.03922)

A lightweight and interactive chatbot powered by **Meta’s LLaMA 3** and enhanced through **Retrieval-Augmented Generation (RAG)** for domain-specific answers. Comes with a clean **Gradio** UI for easy access and testing.

---

## 🚀 Features

- 🔍 **RAG architecture** for better factual accuracy
- 🤖 Powered by **Meta’s LLaMA 3** model
- 🎛️ Easy-to-use **Gradio interface**
- 📚 Plug in your own data (PDFs, text files, etc.)
- 🌐 Local or cloud-hosted deployment

---

## 📸 Demo Screenshot

> *Case Study:* We provide specific details in plain text of an object and ask questions regarding its properties. The LLM must reply only based on the provided context.
![image](https://github.com/user-attachments/assets/33b1c1bd-777c-45b5-9986-1b101f23b9c4)

---

## 🛠️ Installation

```bash
git clone https://github.com/raf-init/rag-llama3-gradio.git
cd rag-llama3-gradio
pip install -r requirements.txt
