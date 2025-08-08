# 📘 MLE_HW_KevinLi ✨✨

Hi, I'm Kevin!  
Welcome to my journey through the **Machine Learning Engineering in GenAI** course. I'm a beginner in Python and have some basic HTML/CSS knowledge - afterall, I come from a graphic design background. This repo is where I’ll document my weekly homework, projects, and notes as I learn and grow through this 10-week program.

---

## 🧠 Repo Overview

This repo contains my homework and notes for each week of the MLE course. Each week is organized with space for notes, learnings, and code snippets. I’ll update this regularly as I complete each assignment.

---

## 🗓️ Weekly Breakdown

### Week 1: 🧱 Introduction to Large Language Models (LLMs) and Prompt Engineering
- 🔬 **Main Learnings** 
  - **MCP Frameworks (Model Context Protocol):** A framework that allows language models to interact seamlessly with external tools and data sources, enabling more AI capabilities.
  - **Locally Run Models:** Ran models locally, like Llama.
  - **LangChain:** Simple framework to build LLM-powered applications by chaining prompts and tools.
  - **Gradio:** Python library to quickly create and host web UIs for AI models.

- 📝 **Project 1.3:** Testing Puppeteer in Claude
  - 📌 *Note:* Puppeteer only takes screenshots within its environment inside Claude.
- 📝 **Project 3.1:** Prompt Engineering with Langchain
  - 📌 *Note:* It is difficult to prompt engineer the model to only provide 1 word/ city name answers without additional text. This may be a model issue.

---

### Week 2: ⚙️ LLM Training Phases and Data Requirements
- 🔬 **Main Learnings** 
  - **Self-Attention:** A mechanism where each token in a sequence considers and weighs every other token to capture their relationships and context within the same input.
  - **Multi-Head Attention:** A process where multiple self-attention mechanisms run in parallel, allowing the model to capture diverse types of relationships by focusing on different aspects of the data simultaneously.
  - **Parameters:** Model components such as weights and biases (e.g., Q, K, V matrices) that are learned and updated during training to optimize performance.
  - **Hyperparameters:** Settings like learning rate, batch size, and number of layers that are chosen before training and control how the model is built and learns, rather than being learned from data.
  - **Context Window:** The maximum number of tokens the model can process at once, which limits how much information it can consider at a time during training or inference.
  - **Quantization:** A method of converting model weights and activations from high-precision (e.g., float32) to lower-precision (e.g., int8) to reduce memory use and speed up inference, usually with a minor loss in accuracy.
  - **Mixed Precision:** A training and inference technique where both float16 and float32 are used together to save memory and increase computational speed while maintaining model stability.

- 📝 **Project 1.1:** Tesseract OCR
  - 📌 *Note:* I tried best practices by cleaning the input image to black and white, and cleaned up grains. However, the output was actually worse than the uncleaned version. This may be because I'm cleaning it wrong, or because of limitations of tesseract itself.
- 📝 **Project Bonus.1:** HTML scraping with trafilatura and extracting with tesseract
  - 📌 *Note:* Trafilatura is already capable of converting html and optimizing its content to JSON files. Not sure where tesseract comes into play.
- 📝 **Project Bonus.3:** Whisper auto language detection failure
  - 📌 *Note:* The "base" weight model for whisper detected "welsh" instead of english, when auto language detection was turned on, which caused errors. This was manually fixed by determining the language=en
- 📝 **Project Bonus.4:** Stats
  - 📌 *Note:* Not sure the types of stats we should display
---

### Week 3: 🧠 Pretraining Data Collection and Extraction
- 📝 *[Record notes and code snippets]*

---

### Week 4: 🛠️ RAG
- 📝 *[Record notes and code snippets]*

---

### Week 5: 🤖 Supervised Fine Tuning - Part 1
- 📝 *[Record notes and code snippets]*

---

### Week 6: 🤖 Supervised Fine Tuning - Part 2
- 📝 *[Record notes and code snippets]*

---

### Week 7: 🧪 Alignment Techniques for LLMs
- 📝 *[Record notes and code snippets]*

---

### Week 8: 🌐 Hallucination, Jailbreak, a n d Ethical Considerations
- 📝 *[Record notes and code snippets]*

---

### Week 9: 📉 Voice Agents & Multimodal Interfaces
- 📝 *[Record notes and code snippets]*

---

### Week 10: 🚀  AI Agent: Agentic Systems & Workflow Automation
- 📝 *[Describe your final project idea, results, and reflections]*

---

## 🛠️ Tools & Tech Stack

- Python (core)
- MCP (standardized tool calling) https://github.com/punkpeye/awesome-mcp-servers
- Gradio (for app UIs) https://www.gradio.app/
- LangChain (for chaining LLMs) https://www.langchain.com/
- Ollama (for running local LLMs) https://ollama.com/
- Claude (LLM with native MCP support - Anthropic)

---

## 📌 Goal

- To be able to build fully functional AI-Agents with enjoyable to use user interfaces supported by robust backend architecture.

---

## 🙌 A Note to Future Me

Every expert once started where I am. Keep going. The only way to get better is to keep building. 🚀


