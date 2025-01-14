# Project Title
A Machine Learning Pipeline for Text Classification and Spam Detection

## Overview
This project implements a spam detection system for YouTube comments using machine learning techniques. The primary goal is to classify comments as either spam or not spam based on their content. The project utilizes Python and various libraries such as Scikit-learn, Pandas, NLTK, and Matplotlib for data preprocessing, feature extraction, model training, and evaluation.

This project consists of two Jupyter Notebooks:
1. **`geng.ipynb`**: Handles data preprocessing, model training, and evaluation.
2. **`pers.ipynb`**: Focuses on the persistence of the trained model and vectorizer, as well as predictions on new comments.

---

## Features
### 1. `geng.ipynb`
- Preprocesses text data (tokenization, stopword removal, stemming/lemmatization).
- Extracts features using TF-IDF vectorization.
- Splits data into training and test sets.
- Trains a logistic regression classifier.
- Evaluates the model with accuracy, confusion matrix, and classification report.
- Visualizes results with Matplotlib.

### 2. `pers.ipynb`
- Loads a pre-trained spam detection model and vectorizer using Joblib.
- Provides a function (`pred_class`) to classify text as spam or not spam.
- Demonstrates the model's use with example inputs.

---

## Setup and Usage

### Prerequisites
Ensure the following are installed:
- Python 3.12+
- Required libraries: Pandas, Scikit-learn, NLTK, Matplotlib, Joblib

Install dependencies using pip:
```bash
pip install pandas scikit-learn nltk matplotlib joblib
```

### Running the Notebooks
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Open the notebooks using Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```

3. Run the cells sequentially to execute the workflows in each notebook.

---

## File Descriptions
- **`geng.ipynb`**: End-to-end pipeline for text classification.
- **`pers.ipynb`**: Script for using a pre-trained spam classification model.
- **Pre-trained model files** (e.g., `youtubespam_model.joblib`, `youtubespam_vectorizer.joblib`): Required for inference in `pers.ipynb`.

---

## Example Usage (`pers.ipynb`)

### Loading Model and Vectorizer
```python
import joblib

MODEL_PATH = "youtubespam_model.joblib"
VECTORIZER_PATH = "youtubespam_vectorizer.joblib"

loaded_model = joblib.load(MODEL_PATH)
loaded_vectorizer = joblib.load(VECTORIZER_PATH)
```

### Predicting Class
```python
def pred_class(comment):
    comment_vectorized = loaded_vectorizer.transform([comment])
    predicted_class = loaded_model.predict(comment_vectorized)[0]
    if predicted_class == 0:
        return f'"{comment}" is NOT A SPAM MESSAGE'
    else:
        return f'"{comment}" IS A SPAM MESSAGE'

# Example usage
comment = "Click my link to win a prize"
print(pred_class(comment))
```

---

## Example Usage (`geng.ipynb`)
### Preprocessing and Training
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('youtube_comments.csv')

# Preprocess comments (tokenization, stopwords removal etc.)
# ...

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['CONTENT'])
y = data['CLASS']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions)}')
```

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- Libraries: NLTK, Scikit-learn, Pandas, Matplotlib
- Pre-trained model for spam detection

