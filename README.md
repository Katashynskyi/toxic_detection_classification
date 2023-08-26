# Toxic Detection and Classification

![GitHub](https://img.shields.io/github/license/Katashynskyi/toxic_detection_classification)
![GitHub last commit](https://img.shields.io/github/last-commit/Katashynskyi/toxic_detection_classification)

## Business Idea

This project aims to create an efficient classifier for identifying toxic comments on forums, Telegram channels, and similar platforms. Driven by the client's needs, the model is designed to operate in real-time and can also be triggered at intervals or on demand. Upon detecting objectively toxic comments, the algorithm will activate. As per the algorithm's architecture, the corresponding tools will either automatically hide the comment, temporarily block the commenter, or even apply a ban.

The severity of actions can be configured based on the predict_proba output of the model and the platform's moderation policy. In an ideal scenario, when a certain toxicity threshold is reached, the comment is hidden, and the algorithm sends it for review by administrators. Essentially, the model requires a Human-in-the-loop process for these cases.## Approaches Tried

- **Text Embeddings**: Experimented with various embeddings including TF-IDF, FastText, and Spacy (-sm and -md models).
- **Binary Classification Algorithms**: Explored Support Vector Classifier (SVC), XGBoost, and LightGBM for binary toxicity classification.
- **Multiclass Classification**: Utilized the DistilBERT_uncased transformer for multiclass classification.
- **Data Preprocessing**: Applied diverse preprocessing techniques to optimize model performance.

## Steps

- Data Preprocessing: Implementation of text preprocessing techniques to clean and tokenize input text.
- Model Training: Construction and training of a machine learning model using a labeled dataset.
- Model Evaluation: Assessment of the model's performance through metrics such as accuracy, precision, recall, and F1-score.
## Steps TODO
- Real-time Classification: Swift detection and management of toxic comments in real-time.(I'm here)
- Trigger Options: Possibility to invoke the model periodically or manually, based on specific requirements.
- Adaptive Actions: Implementation of diverse actions based on model predictions, such as comment hiding, user blocking, or banning.
- Human-in-the-Loop: Incorporation of a review process for comments exceeding a predefined toxicity threshold.
## Getting Started
### Prerequisites

- Python 3.9+
- Virtual environment (Conda 3.9)

### Installation

- Clone the repository: git clone git@github.com:Katashynskyi/toxic_detection_classification.git
- Create conda venv (Conda 3.9)
- Install the required packages: pip install -r requirements.txt

## Usage

After installing the required packages, you can preprocess, train and ~~deploy~~. 

- To preprocess your data and train the models: Choose method 1 (boosting models) or method 2 (bert transformer) and then run_train_pipeline.py 

## Model Training
The model is trained on [Toxic Comment Classification Challenge by Jigsaw on kaggle.com](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview)
. Where are multiple binary targets for 6 types of toxicity.
## Deployment
- in progress

