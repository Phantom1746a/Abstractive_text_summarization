# 🤖 Urdu Text Summarization with Mistral 7B + LoRA + Unsloth

A lightweight, efficient fine-tuning pipeline to summarize **Urdu text** using **Mistral 7B**, optimized with **LoRA** and **Unsloth** for fast 4-bit training on consumer GPUs.

🎯 **Goal**: Fine-tune Mistral 7B to generate concise summaries of Urdu news articles, blogs, or documents.  
⚡ **Speed**: 2x faster training with **Unsloth**'s optimized kernels.  
💾 **Efficiency**: Uses only **~15GB VRAM** via 4-bit quantization and LoRA.

---

## 📷 Demo (Example)

**Input (Urdu):**  
> "پاکستان کی معیشت حالیہ برسوں میں چیلنجز کا شکار رہی ہے۔ مہنگائی بڑھی اور روپے کی قدر کم ہوئی۔ حکومت نے آئی ایم ایف سے قرض لیا۔"

**Output (Summary):**  
> "حکومت نے معیشت کے چیلنجز کے پیشِ نظر آئی ایم ایف سے قرض لیا۔"

---

## 🧰 Features

- ✅ Fine-tunes **Mistral 7B** for **Urdu summarization**
- 🔧 Uses **LoRA (Low-Rank Adaptation)** for parameter-efficient tuning
- ⚡ Powered by **Unsloth** – 2x faster 4-bit training
- 📊 Supports **CSV datasets** (easy data loading)
- 💬 Instruction-tuned prompts in **Urdu**
- 📦 Full training & inference scripts 
- 🚀 Ready to deploy with Gradio (optional)

---

