# Resume Category Predictor

A simple and interactive web app built with [Streamlit](https://streamlit.io/) that predicts the job category of uploaded PDF resumes. The app uses a Logistic Regression model trained on TF-IDF features extracted from resume text.

---

## Features

- Upload one or more PDF resumes.
- Automatically extract text from PDFs using `pdfplumber`.
- Preprocess and vectorize the resume text with TF-IDF.
- Predict the job category using a trained Logistic Regression model.
- Display predicted categories in a neat table.

---

## Demo

![demo-screenshot](path_to_screenshot.png)  
*(Replace with your screenshot or GIF)*

---

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/resume-category-predictor.git
   cd resume-category-predictor
