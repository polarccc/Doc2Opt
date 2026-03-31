<h1 align="center">
💡Doc2Opt
</h1>
<h2 align="center">
An Effective and Efficient System for Document-Grounded Automatic Optimization Modeling
</h2>

## ✨ Overview
<p align="center">
<img src="image\README\framework.png" width="800"/>
</p>

The **Doc2Opt Framework** , which highlights three key designs discussed in the following.

### 🗂️ 1. Cross-modal retrieval-augmented modeling
The *cross-modal retrieval* module serves as a semantic filter toretain only relevant document pages for different modeling tasks. The retrieved images of pages and the task description are then fed into a VLM to generate the optimization model as discussed below.
### 🚀2. Five-element representation

Instead of directly generating the
mathematical model, we adopt a more concise and structured *five-element formulation* as an intermediate representation to improve the reliability and interpretability of the modeling.
- 🎯 **Objective**
- 🔢 **Decision Variables**
- 📦 **Sets**
- 📊 **Parameters**
- 📏 **Constraints**

Moreover, we instruct the generator to explain the *references* for the construction of each item in *five-element formulation*, from which content
of the NL description or document pages a specific attribute or
constraint is derived, which makes the modeling process tracable
and interpretable. Also, it allows an easy refinement of the model by
correcting the inappropriate terms based on the explanation.
### 🔄3. Autonomous Refinement

To achieve more robust and adaptive modeling to improve accuracy, we design a simple yet effective mechanism to automatically *evaluate and refine* the constructed model.

Inspired bythe *LLM-as-a-judge* approach, we use another VLM as an evaluator to assess the quality of the model produced by the generator. The evaluator is instructed to analyze the five-element
representation and explanation to determine whether the model is required to be refined.


---
## 🔧 Usage
To use Doc2Opt, follow these steps:

First, clone the repository:
```bash
git clone https://github.com/polarccc/Doc2Opt
```
Then, install the required dependencies:
```bash
pip install -r requirements.txt
```
We provide two ways to run the project depending on your needs:
### 🖥️ Option 1: Launch the Interactive App 
```bash
streamlit run app.py
```


### ⚡ Option 2: Run Core Pipeline Directly
```bash
python Doc2Opt.py  
    --model "Vision-Language Model" \
    --files "Path/to/your/files" 
    --question "Natural Language Description" \
    --api-key "Your API Key" \
    --base-url "Your API URL"  \
```
---
## 🖥️ GUI Demo
### (a) Document upload for cross-modal retrieval
<p align="center">
<img src="image\README\demo_1.png" width="600"/>
</p>

### (b) Task description using natural language
<p align="center">
<img src="image\README\demo_2.png" width="600"/>
</p>

### (c) Five-element with explanation
<p align="center">
<img src="image\README\demo_3.png" width="600"/>
</p>

### (d) Retrieved pages
<p align="center">
<img src="image\README\demo_4.png" width="300"/>
</p>

### (e) Complete model
<p align="center">
<img src="image\README\demo_5.png" width="300"/>
</p>

### (f) Manual refinement
<p align="center">
<img src="image\README\demo_6.png" width="300"/>
</p>