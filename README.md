# Voice-Enabled Student Admission Chatbot

## Implementation of Signal-Aware Voice Enabled Hybrid Admission Chatbot Using Retrieval Augmented Generation and Rule-Based Reasoning

### Project Overview

This project is a Voice-Enabled Student Admission Chatbot developed to assist students with admission-related queries. The chatbot combines Retrieval-Augmented Generation (RAG) and Rule-Based Reasoning to provide accurate, reliable, and context-aware responses.

The system supports both text and voice interactions, allowing prospective students to obtain information about admissions, cutoffs, fees, hostel facilities, seat availability, placements, and institutional details.

---

## Features

* Voice-enabled interaction
* Admission query handling
* Retrieval-Augmented Generation (RAG)
* Rule-based reasoning for eligibility checks
* Hybrid response generation
* Knowledge base using JSON datasets
* Evaluation framework for performance analysis
* Student-friendly conversational interface

---

## Technologies Used

* Python
* JSON
* Retrieval-Augmented Generation (RAG)
* Rule-Based Reasoning
* Speech Recognition
* Natural Language Processing (NLP)

---

## Project Structure

```text
Voice-Enabled-Student-Admission-Chatbot/
│
├── base_loader.py
├── chat_with_bot.py
├── voice_input.py
├── vanilla.py
├── rag.py
├── hybrid.py
├── evaluator.py
├── run_evaluation.py
├── evaluation.py
├── requirements.txt
│
├── cutoffs.json
├── fees.json
├── hostel.json
├── programs.json
├── seats.json
├── evaluation_dataset.json
│
└── MAINREPORT.pdf
```

## Knowledge Base

The chatbot uses structured institutional data stored in JSON files:

* Cutoff ranks
* Fee details
* Hostel information
* Program information
* Seat availability
* Admission guidelines

---

## Evaluation

The project includes an evaluation framework for comparing:

1. Vanilla Response Generation
2. Retrieval-Augmented Generation (RAG)
3. Hybrid Approach (RAG + Rule-Based Reasoning)

Evaluation metrics include:

* Response Accuracy
* Hallucination Detection
* ROUGE-L Score
* Response Latency

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Chatbot

```bash
python chat_with_bot.py
```

### Run Voice Interaction

```bash
python voice_input.py
```

### Run Evaluation

```bash
python run_evaluation.py
```

---

## Applications

* Student Admission Assistance
* Academic Information Support
* Institutional Help Desk Automation
* Voice-Based Information Retrieval

---

## Future Enhancements

* Multilingual Support
* Mobile Application Integration
* Advanced Speech Processing
* Real-Time Database Connectivity
* Personalized Student Guidance

---

## Authors

Final Year B.Tech Project Team

Department of Computer Science and Engineering

Vignan's Lara Institute of Technology & Science (VLITS)

---

## Project Report

Detailed implementation and evaluation can be found in:

`MAINREPORT.pdf`
