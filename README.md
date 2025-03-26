# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
A brief overview of your project and its purpose. Mention which problem statement are your attempting to solve. Keep it concise and engaging.

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

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

## 🏗️ Tech Stack
- 🔹 Frontend: 
- 🔹 Backend: Python / FastAPI / Django
- 🔹 Database: 
- 🔹 Other: OpenAI API 

## 👥 Team
- **Your Name** - [GitHub](#) | [LinkedIn](#)
- **Teammate 2** - [GitHub](#) | [LinkedIn](#)
