# import joblib
# from flask import Flask, render_template, request, jsonify
# import numpy as np

# app = Flask(__name__)

# # Load the saved model and vectorizer
# MODEL_PATH = "model/youtubespam_model.joblib"
# VECTORIZER_PATH = "model/youtubespam_vectorizer.joblib"

# loaded_model = joblib.load(MODEL_PATH)
# loaded_vectorizer = joblib.load(VECTORIZER_PATH)

# def predict_spam(comment):
#     """
#     Predict if a comment is spam or not
    
#     Args:
#         comment (str): Input comment text
    
#     Returns:
#         dict: Prediction result and probability
#     """
#     # Transform the input comment using the loaded vectorizer
#     comment_vectorized = loaded_vectorizer.transform([comment])

#     # Predict using the loaded model
#     predicted_class = loaded_model.predict(comment_vectorized)[0]
    
#     # Get prediction probabilities
#     prediction_proba = loaded_model.predict_proba(comment_vectorized)[0]
    
#     return {
#         'is_spam': bool(predicted_class),
#         'spam_probability': float(prediction_proba[1]),
#         'message': "SPAM MESSAGE" if predicted_class == 1 else "NOT A SPAM MESSAGE"
#     }

# @app.route('/')
# def index():
#     """Render the main page"""
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Endpoint for spam prediction"""
#     comment = request.form.get('comment', '')
    
#     if not comment:
#         return jsonify({'error': 'No comment provided'}), 400
    
#     result = predict_spam(comment)
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)



# import joblib
# from flask import Flask, render_template, request, jsonify
# import numpy as np

# app = Flask(__name__)

# # Load the saved model and vectorizer
# MODEL_PATH = "model/youtubespam_model.joblib"
# VECTORIZER_PATH = "model/youtubespam_vectorizer.joblib"

# loaded_model = joblib.load(MODEL_PATH)
# loaded_vectorizer = joblib.load(VECTORIZER_PATH)

# def predict_spam(comment):
#     """
#     Predict if a comment is spam or not
    
#     Args:
#         comment (str): Input comment text
    
#     Returns:
#         dict: Prediction result and probability
#     """
#     try:
#         # Transform the input comment using the loaded vectorizer
#         comment_vectorized = loaded_vectorizer.transform([comment])

#         # Predict using the loaded model
#         predicted_class = loaded_model.predict(comment_vectorized)[0]
        
#         # Get prediction probabilities
#         prediction_proba = loaded_model.predict_proba(comment_vectorized)[0]
        
#         return {
#             'is_spam': bool(predicted_class),
#             'spam_probability': float(prediction_proba[1]),
#             'message': "IS A SPAM MESSAGE" if predicted_class == 1 else "IS NOT A SPAM MESSAGE"
#         }
#     except Exception as e:
#         return {
#             'error': str(e),
#             'is_spam': False,
#             'spam_probability': 0.0,
#             'message': "COULD NOT PROCESS MESSAGE"
#         }

# @app.route('/')
# def index():
#     """Render the main page"""
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Endpoint for spam prediction"""
#     comment = request.form.get('comment', '')
    
#     if not comment:
#         return jsonify({
#             'error': 'No comment provided',
#             'is_spam': False,
#             'spam_probability': 0.0,
#             'message': "NO COMMENT ENTERED"
#         }), 400
    
#     result = predict_spam(comment)
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)


import joblib
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the saved model and vectorizer
MODEL_PATH = "model/youtubespam_model.joblib"
VECTORIZER_PATH = "model/youtubespam_vectorizer.joblib"

loaded_model = joblib.load(MODEL_PATH)
loaded_vectorizer = joblib.load(VECTORIZER_PATH)

def predict_spam(comment):
    """
    Predict if a comment is spam or not
    
    Args:
        comment (str): Input comment text
    
    Returns:
        dict: Prediction result and probability
    """
    try:
        # Transform the input comment using the loaded vectorizer
        comment_vectorized = loaded_vectorizer.transform([comment])

        # Predict using the loaded model
        predicted_class = loaded_model.predict(comment_vectorized)[0]
        
        # Get prediction probabilities
        prediction_proba = loaded_model.predict_proba(comment_vectorized)[0]
        
        return {
            'is_spam': bool(predicted_class),
            'spam_probability': float(prediction_proba[1]),
            'message': "IS A SPAM MESSAGE" if predicted_class == 1 else "is NOT A SPAM MESSAGE"
        }
    except Exception as e:
        return {
            'error': str(e),
            'is_spam': False,
            'spam_probability': 0.0,
            'message': "COULD NOT PROCESS MESSAGE"
        }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for spam prediction"""
    # Change from form.get to request.form.get
    comment = request.form.get('comment', '')
    
    if not comment:
        return jsonify({
            'error': 'No comment provided',
            'is_spam': False,
            'spam_probability': 0.0,
            'message': "NO COMMENT ENTERED"
        }), 400
    
    result = predict_spam(comment)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)