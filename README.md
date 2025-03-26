# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](https://github.com/ewfx/gaied-quantum-coders/tree/main/artifacts/demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction

Solution Approaches
We propose two approaches to solve the problem:
Google Vertex AI Large Language Model
Custom ML based classification.

# Approach 1: Google Vertex AI (LLM-based Solution) 
Using Google Vertex AI   gemini-1.5-flash-001 model to classify incoming emails with attachments, generate structured output, and perform duplicates detection using the open-source BERT model. To extract content from PDF and images we leverage the PyTesseract  and PyPDF2 framework.

# Approach 2: Custom ML Model Solution
Develop a Machine Learning-based system trained on historical data to classify service requests, extract key attributes, and detect duplicates.
The approach differs only for Email Classification stage , where we would be using a Machine Learning model to classify an incoming email instead of google vertex AI.

## 🎥 Demo
🔗 [Live Demo](https://github.com/ewfx/gaied-quantum-coders/blob/main/artifacts/demo/demo_1ftu_gaied.mp4) (if applicable)  
📹 [Video Demo](https://github.com/ewfx/gaied-quantum-coders/blob/main/artifacts/demo/demo_1ftu_gaied.mp4) (if applicable)  
🖼️ Screenshots:

![Screenshot 1]([link-to-image](https://github.com/ewfx/gaied-quantum-coders/blob/main/artifacts/demo))

## 💡 Inspiration
The project was inspired by the need to automate email classification and data extraction in Commercial Bank Lending Services. Traditionally, loan servicing teams receive high volumes of service requests via email, requiring manual effort to:

Read and interpret email content and attachments
Identify the request type and sub-type
Extract key attributes for processing
Assign the request to the correct team
This manual triage process is time-consuming, error-prone, and inefficient, especially during peak workloads. Misclassifications can lead to delays, compliance risks, and operational inefficiencies.

## ⚙️ What It Does
Problem We're Solving
We aim to automate this workflow using Machine Learning (ML) and Generative AI to:
✅ Accurately classify emails into categories & subcategories
✅ Extract key details (e.g., amounts, dates, names)
✅ Detect duplicate requests using AI-based similarity checks
✅ Reduce manual effort and improve processing speed & accuracy

By leveraging our own ML model (RandomForest, SVM, BERT) or Google Vertex AI, we ensure:
✔ Faster turnaround time for loan servicing requests
✔ Improved accuracy in classification & extraction
✔ Scalability for large volumes of emails

## 🛠️ How We Built It
We built this AI-powered email classification and extraction system using two approaches:

Approach 1: Custom Machine Learning Model
1️⃣ Email Ingress

Connected a dedicated email inbox
A scheduler runs every minute to fetch unread emails
Extracted email content & attachments, then stored them in an input folder
Marked emails as read after processing
2️⃣ Email Translation

Parsed .eml files and extracted email body & attachments
Converted the extracted data into structured JSON
Created a processing pipeline for classification & duplicate detection
3️⃣ Classifier

Defined categories & subcategories
Trained an ML model using historical email data
Used:
✅ TF-IDF Vectorizer for text transformation
✅ Cosine Similarity for similarity measurement
✅ RandomForestClassifier & SVM for classification
Predicted category & subcategory based on email priority
Generated structured output for further processing
4️⃣ Duplicate Detection

Used BERT embeddings for semantic similarity analysis
Applied cosine similarity to detect duplicate emails
Configured a threshold (0.9) for duplicate detection
5️⃣ Output Processing

Added a duplicate flag to the classifier output
Generated a final structured JSON output for loan servicing
Tech Stack: Python 🐍 | FastAPI ⚡ | Scikit-learn 🤖 | BERT 🧠


Approach 2: Google Vertex AI (LLM)
Instead of training a custom ML model, we leveraged Google Vertex AI for:
✅ Text classification (LLM-based intent detection)
✅ Email & attachment processing
✅ Duplicate detection (embedding-based similarity analysis)

Tech Stack: Python 🐍 | Google Vertex AI 🤖 | Spring AI 🌱

## 🚧 Challenges We Faced
1️⃣ Handling Unstructured Email Data 📧

Emails contained unstructured text, tables, and attachments (PDFs, DOCX, TXT).
Parsing and extracting meaningful data required advanced NLP techniques.
2️⃣ Accurate Classification of Requests 🎯

Some emails had ambiguous intent, making classification difficult.
Similar emails had different meanings depending on context.
Solution: Used TF-IDF, cosine similarity, and BERT embeddings to improve accuracy.
3️⃣ Duplicate Detection with High Precision 🔄

Some emails were slightly modified duplicates (e.g., different wording, same intent).
Solution: Used BERT embeddings + cosine similarity (threshold: 0.9) to detect duplicates.
4️⃣ Efficient Email Processing at Scale ⚡

High volume of incoming emails required real-time processing.
Solution: Implemented a queue-based pipeline with asynchronous processing.
5️⃣ Attachment Processing & OCR 📄🔍

Extracting relevant text from PDFs, scanned documents, and images was a challenge.
Solution: Used Tesseract OCR for text extraction from non-searchable PDFs/images.
6️⃣ Integrating with Loan Servicing Workflow 🔗

The system needed to seamlessly integrate with existing banking workflows.
Solution: Generated a structured JSON output that could be directly ingested into the system.
7️⃣ Fine-Tuning Google Vertex AI (LLM) 🧠

While Vertex AI provided out-of-the-box NLP, fine-tuning for banking-specific terms was needed.
Solution: Used custom embeddings and few-shot learning for better classification.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone [https://github.com/your-repo.git](https://github.com/ewfx/gaied-quantum-coders/)]
   ```
2. Install dependencies  
   ```sh
   navigate to code/src folder and find the modules below there
      1. duplicateDetection
      2. emailClassifier
      3. emailExtraction
      4. emailIngress
      5. vertexAIClassifer
   
   Got to each folder and run the each command for installing the dependencies
      1. python -m venv ven
      2. source venv/bin/activate
      3. pip install -r requirement.txt
      4. For only emailClassifier folder run the command "python -m spacy download en_core_web_sm"
      5. vertexAIClassifier build artificat and run as spring boot project
   ```
3. Run the project  
   ```sh
   navigate to code/src folder and find the modules below there
      1. duplicateDetection
            uvicorn similarityDetection:app --port=8002 --reload
      2. emailClassifier
            uvicorn emailMLClassifier:app --port=8001 --reload
      3. emailExtraction
            uvicorn emailTranslation:app --port=8000 --reload
      4. emailIngress
            check the .env files and provide your system path to workflow execute
            python email_scheduler.py
      5. vertexAIClassifier
            java -jar vertexAIClassifier.jar
   ```

![SWAGGER_UI](https://github.com/ewfx/gaied-quantum-coders/blob/e5122686c4146d2f0d1948fe012ba783d7fc7135/artifacts/demo/Screenshot%202025-03-26%20at%202.10.32%E2%80%AFPM.png)
![SWAGGER_UI](https://github.com/ewfx/gaied-quantum-coders/blob/e5122686c4146d2f0d1948fe012ba783d7fc7135/artifacts/demo/Screenshot%202025-03-26%20at%202.10.55%E2%80%AFPM.png)
![SWAGGER_UI](https://github.com/ewfx/gaied-quantum-coders/blob/e5122686c4146d2f0d1948fe012ba783d7fc7135/artifacts/demo/Screenshot%202025-03-26%20at%202.11.05%E2%80%AFPM.png)
## 🏗️ Tech Stack
- 🔹 Frontend: 
- 🔹 Backend: Python / FastAPI / Django
- 🔹 Database: 
- 🔹 Other: OpenAI API 

## 👥 Team
- **Quantum Coder** - [GitHub](#) | [LinkedIn](#)
- **Karthikeyan Sethurama** - [GitHub](#) | [LinkedIn](#)
- **Selvi Senthamarai** - [GitHub](#) | [LinkedIn](#)
- **Bhavna** - [GitHub](#) | [LinkedIn](#)
- **Vivek** - [GitHub](#) | [LinkedIn](#)
- **Azeez Pallor** - [GitHub](#) | [LinkedIn](#)


