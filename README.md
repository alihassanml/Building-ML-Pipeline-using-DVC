# Human Emotion Detection from Text using DVC

This project builds a machine learning pipeline to detect human emotions (e.g., joy, sadness, anger) from text input. It uses Python, scikit-learn, and DVC (Data Version Control) to manage data, model training, and experiment tracking.

## 📌 Project Overview

- **Objective**: Classify human emotions based on text data.
- **Labels**: joy, sadness, anger, love, fear, surprise.
- **Pipeline**: Built using [DVC](https://dvc.org/) to automate and reproduce stages: data ingestion, preprocessing, training, and evaluation.

## 🛠️ Tech Stack

- Python, scikit-learn, pandas, matplotlib, seaborn
- NLP: NLTK, TfidfVectorizer
- ML Models: Logistic Regression, SVM, Random Forest, MultinomialNB
- DVC for pipeline and metrics tracking
- Git & GitHub for version control

## 📂 Directory Structure

```

.
├── data/
│   ├── raw/                  # Raw dataset
│   └── preprocess/           # Cleaned dataset
├── results/
│   └── training/             # Metrics from model training
├── src/
│   ├── data\_ingestion.py     # Load & clean raw data
│   ├── model\_training.py     # Train and evaluate ML models
│   ├── model\_evaluation.py   # (Optional future) Model evaluation
│   └── pickle/               # Serialized models and encoders
├── dvc.yaml                  # DVC pipeline definition
└── README.md

````

## 🔁 DVC Pipeline

The pipeline includes the following stages:

1. **data_ingestion**: Loads and saves cleaned text dataset.
2. **model_training**: Trains 4 classifiers and evaluates their accuracy.
3. **model_evaluation** (optional): For detailed evaluation (coming soon).

### Run the full pipeline:
```bash
dvc repro
````

## 📊 Metrics

Training metrics are stored in:

```
./results/training/metrics.json
```

Sample:

```json
[
  {"Model": "RandomForestClassifier", "Accuracy": 0.85},
  {"Model": "MultinomialNB", "Accuracy": 0.80},
  {"Model": "SVC", "Accuracy": 0.82},
  {"Model": "LogisticRegression", "Accuracy": 0.86}
]
```

## 💾 Model Files

The trained models and encoders are saved in `src/pickle/`:

* `logistic_regression.pkl`
* `label_encoder.pkl`
* `vector.pkl`

## 📦 Installation

```bash
git clone https://github.com/alihassanml/Building-ML-Pipeline-using-DVC.git
cd Building-ML-Pipeline-using-DVC

# Install dependencies
pip install -r requirements.txt

# (Optional) Initialize DVC
dvc init
```

## 📈 Example Input

```
Text: "I am feeling very sad today."
Predicted Emotion: Sadness
```

## 🧪 Future Improvements

* Add a REST API using FastAPI or Flask
* Deploy as a web app
* Use deep learning models (e.g., LSTM, BERT)
* Expand dataset and support multilingual text

## 📄 License

This project is open-source under the MIT License.

---

## 👤 Author

**Ali Hassan**
GitHub: [@alihassanml](https://github.com/alihassanml)

```