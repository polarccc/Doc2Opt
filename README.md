# 🧠 Doc2Opt: An Effective and Efficient System for Document-Grounded Automatic Optimization Modeling

---

## ✨ Overview
The **Doc2Opt Framework** , which highlights three key designs discussed in the following.

### 1. **Cross-modal retrieval-augmented modeling**
The cross-modal retrieval module serves as a semantic filter toretain only relevant document pages for different modeling tasks. The retrieved images of pages and the task description are then fed into a VLM to generate the optimization model as discussed below.
### 2. **Five-element representation**

Instead of directly generating the
mathematical model, we adopt a more concise and structured five-element formulation as:
- 🎯 **Objective**
- 🔢 **Decision Variables**
- 📦 **Sets**
- 📊 **Parameters**
- 📏 **Constraints**

### 3. **Autonomous Refinement**

Inspired bythe LLM-as-a-judge approach, we use another VLM as an evaluator to assess the quality of the model produced by the generator.

---
## 🔧 Usage
To use Doc2Opt, follow these steps:

First, install the required dependencies:
```bash
pip install -r requirements.txt
```
Then, run the main script to show the GUI:
```bash
python app.py
```


Alternatively, run the following command to execute the pipeline:
```bash
python Doc2Opt.py  
    --model "Vision-Language Model" \
    --files "Path/to/your/files" 
    --question "Natural Language Description" \
    --api-key "Your API Key" \
    --base-url "Your API URL"  \
```
