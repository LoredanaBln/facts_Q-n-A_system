# Facts Q&A System

## Overview

The Facts Q&A System is a desktop application that allows users to ask questions and receive short answers based on a custom-trained language model. The system is a fine-tuned GPT-2 model and provides an user-friendly interface built with PyQt6.

## Features

- **Modern GUI:** User-friendly interface for a Q&A session.
- **Custom Knowledge Base:** Answers are generated from a curated dataset of facts, organized by topic.
- **Topic Extraction:** The system extracts the topic from questions to provide a relevant and based on real information answer.
- **Model Training:** Includes scripts to train and fine-tune a GPT-2 model on a dataset.

## Project Structure

- `ui/main_window.py` — Main PyQt6 application window and UI logic.
- `fact_Q_n_A_system/train/ask.py` — Functions for loading the model, extracting topics, and generating answers.
- `fact_Q_n_A_system/train/trainer.py` — Script for training or fine-tuning the GPT-2 model on the dataset.
- `dataset/` — Directory for the training data.
- `run.py` — Entry point to launch the application.

## How It Works

1. **Training:**  
   Use `trainer.py` to fine-tune a GPT-2 model on facts from a custom dataset.
2. **Model loading:**  
   The application loads the trained model and available topics at startup.
3. **Question answering:**  
   Users enter a question; the app extracts the topic and returns the relevant fact if available.
4. **Available topics:**  
   The UI displays all topics the system can answer about.

## Getting Started

1. **Install requirements:**  
   ```
   pip install -r requirements.txt
   ```

2. **Prepare dataset:**  
   - Place your facts in `dataset/facts_dataset.json` (for answering).
   - Convert JSON file to text file
   ```
   python fact_Q_n_A_system/train/convertor.py
   ```

3. **Train the Model:**  
   ```
   python fact_Q_n_A_system/train/trainer.py
   ```

4. **Run the Application:**  
   ```
   python run.py
   ```

## Requirements

- Python 3.8+
- PyQt6
- transformers
- datasets

(See `requirements.txt` for full dependencies.)
