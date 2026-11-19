# 🎓 End-to-End Student Exam Performance Indicator

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/<YOUR_HF_USERNAME>/student-performance-predictor)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)

> **🚀 Live Interactive App:** [Try the Student Performance Predictor on Hugging Face Spaces](https://huggingface.co/spaces/<YOUR_HF_USERNAME>/student-performance-predictor)  
> **📊 Live Inference Audit Log:** View real-time logged predictions at `/predictions` on the deployed app.

---

## 📌 Project Overview

An end-to-end Machine Learning web application that predicts a student's **Mathematics Score** based on demographic attributes, parental education level, lunch type, test preparation status, and reading/writing scores.

Rather than stopping at model training in a notebook, this project implements a modular, production-grade ML pipeline covering automated data ingestion, feature transformation, model selection with hyperparameter tuning, real-time inference logging, and containerized cloud deployment.

---

## 🏗️ System Architecture & MLOps Workflow

```text
GitHub Repository (Modular ML Pipeline + Flask App)
       │
       ├──► Containerized via Docker (Python 3.11 + Gunicorn)
       │         └──► Deployed on Hugging Face Spaces (Live Public Demo)
       │
       ├──► CI/CD Tested with AWS CodePipeline & Elastic Beanstalk
       │
       └──► Live Inference Monitoring (/predictdata ──► artifacts/prediction_summary.csv ──► /predictions)
       