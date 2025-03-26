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

![Screenshot 1]([link-to-image](https://github.com/ewfx/gaied-quantum-coders/blob/main/artifacts/demo)

## 💡 Inspiration
What inspired you to create this project? Describe the problem you're solving.

## ⚙️ What It Does
Explain the key features and functionalities of your project.

## 🛠️ How We Built It
Briefly outline the technologies, frameworks, and tools used in development.

## 🚧 Challenges We Faced
Describe the major technical or non-technical challenges your team encountered.

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
   Got to each folder and run the each command for installing the dependencies
      1. python -m venv ven
      2. source venv/bin/activate
      3. pip install -r requirement.txt
      4. For only emailClassifier folder run the command "python -m spacy download en_core_web_sm"
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
- **Your Name** - [GitHub](#) | [LinkedIn](#)
- **Teammate 2** - [GitHub](#) | [LinkedIn](#)
