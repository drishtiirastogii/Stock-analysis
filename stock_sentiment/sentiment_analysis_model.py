# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pickle
# import pandas as pd
#
# def load_and_preprocess_data(file_path):
#
#     # Load the dataset
#     df = pd.read_csv(file_path, encoding="ISO-8859-1")
#     df.fillna("", inplace=True)
#
#     # Combine all news headlines into a single string per row
#     df['Combined_News'] = df.iloc[:, 2:27].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
#
#     # Split data into features and labels
#     X = df['Combined_News']
#     y = df['Label']
#     return X, y
#
# def train_lstm_model(X, y):
#
#     # Tokenize and pad sequences
#     tokenizer = Tokenizer(num_words=5000)
#     tokenizer.fit_on_texts(X)
#     X_tokenized = tokenizer.texts_to_sequences(X)
#     X_padded = pad_sequences(X_tokenized, maxlen=200)
#
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
#
#     # Build LSTM model
#     model = Sequential([
#         Embedding(input_dim=5000, output_dim=128, input_length=200),
#         LSTM(128, return_sequences=False),
#         Dropout(0.5),
#         Dense(64, activation='relu'),
#         Dropout(0.5),
#         Dense(1, activation='sigmoid')
#     ])
#
#     # Compile the model
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     # Train the model
#     model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
#
#     # Evaluate the model
#     loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
#     print(f"Test Loss: {loss}")
#     print(f"Test Accuracy: {accuracy}")
#
#     # print(f"Model Accuracy: {accuracy * 100:.2f}%")
#     # Save the tokenizer for later use
#     with open("tokenizer.pkl", "wb") as handle:
#         pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     return model, tokenizer, accuracy

# MLP- 50%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import pickle
import numpy as np

# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df.fillna("", inplace=True)

    # Combine all news headlines into a single string per row
    df['Combined_News'] = df.iloc[:, 2:27].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Features and labels
    X = df['Combined_News']
    y = df['Label']
    return X, y

# Feature engineering and data preprocessing
def preprocess_features(X):
    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X).toarray()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tfidf)

    return X_scaled, vectorizer, scaler

# Train MLP model
def train_mlp_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define MLP model
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )

    # Train the model
    mlp.fit(X_train_resampled, y_train_resampled)

    # Evaluate the model
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return mlp, accuracy

# Predict sentiment percentage
def predict_sentiment_percentage(model, vectorizer, scaler, news):
    # Preprocess the input news
    news_tfidf = vectorizer.transform([news]).toarray()
    news_scaled = scaler.transform(news_tfidf)

    # Predict probabilities
    probabilities = model.predict_proba(news_scaled)
    positive_percentage = probabilities[0][1] * 100  # Probability of the positive class
    negative_percentage = probabilities[0][0] * 100  # Probability of the negative class

    return positive_percentage, negative_percentage

# Main function
if __name__ == "__main__":
    try:
        file_path = "stock_sentiment/stock_senti_analysis.csv"
        X, y = load_and_preprocess_data(file_path)
        X_processed, vectorizer, scaler = preprocess_features(X)
        model, accuracy = train_mlp_model(X_processed, y)

        # Save the model, vectorizer, and scaler
        with open("mlp_sentiment_model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        with open("vectorizer.pkl", "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        with open("scaler.pkl", "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)

        print("Model, vectorizer, and scaler saved successfully.")

        # Example news input
        news = input("Enter news headline(s) to analyze sentiment: ")
        # news = "bad news"
        positive, negative = predict_sentiment_percentage(model, vectorizer, scaler, news)
        print(f"Positive Sentiment: {positive:.2f}%")
        print(f"Negative Sentiment: {negative:.2f}%")
    except Exception as e:
        print(f"Error: {e}")

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pickle
# import pandas as pd
#
# # Load and preprocess data
# def load_and_preprocess_data(file_path):
#     df = pd.read_csv(file_path, encoding="ISO-8859-1")
#     df.fillna("", inplace=True)
#     df['Combined_News'] = df.iloc[:, 2:27].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
#     X = df['Combined_News']
#     y = df['Label']
#     return X, y
#
# def preprocess_features(X):
#     vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
#     X_tfidf = vectorizer.fit_transform(X)
#
#     # Dimensionality reduction
#     svd = TruncatedSVD(n_components=300, random_state=42)
#     X_reduced = svd.fit_transform(X_tfidf)
#
#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_reduced)
#
#     return X_scaled, vectorizer, scaler, svd
#
# def train_mlp_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Handle class imbalance using SMOTE
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#
#     # Define MLP model
#     mlp = MLPClassifier(max_iter=300, random_state=42, early_stopping=True)
#     param_dist = {
#         'hidden_layer_sizes': [(128, 64), (256, 128), (128, 128, 64)],
#         'activation': ['relu', 'tanh'],
#         'alpha': [0.0001, 0.001, 0.01],
#         'learning_rate_init': [0.001, 0.01]
#     }
#
#     search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
#     search.fit(X_train_resampled, y_train_resampled)
#
#     best_model = search.best_estimator_
#
#     # Evaluate the model
#     y_pred = best_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Best Parameters: {search.best_params_}")
#     print(f"Test Accuracy: {accuracy * 100:.2f}%")
#
#     return best_model, accuracy
#
# # Main function
# if __name__ == "__main__":
#     file_path = "stock_sentiment/stock_senti_analysis.csv"
#     X, y = load_and_preprocess_data(file_path)
#     X_processed, vectorizer, scaler, svd = preprocess_features(X)
#     model, accuracy = train_mlp_model(X_processed, y)
#
#     # Save the model, vectorizer, scaler, and SVD
#     with open("mlp_sentiment_model.pkl", "wb") as model_file:
#         pickle.dump(model, model_file)
#     with open("vectorizer.pkl", "wb") as vectorizer_file:
#         pickle.dump(vectorizer, vectorizer_file)
#     with open("scaler.pkl", "wb") as scaler_file:
#         pickle.dump(scaler, scaler_file)
#     with open("svd.pkl", "wb") as svd_file:
#         pickle.dump(svd, svd_file)
#
#     print("Model, vectorizer, scaler, and SVD saved successfully.")

# Logistic regression: 48.96% accuracy
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
#
# # Load and preprocess data
# def load_and_preprocess_data(file_path):
#     # Load dataset
#     df = pd.read_csv(file_path, encoding="ISO-8859-1")
#     df.fillna("", inplace=True)
#
#     # Combine all news headlines into a single string per row
#     df['Combined_News'] = df.iloc[:, 2:27].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
#
#     # Features and labels
#     X = df['Combined_News']
#     y = df['Label']
#     return X, y
#
# # Feature engineering and data preprocessing
# def preprocess_features(X):
#     # Convert text to numerical features using TF-IDF
#     vectorizer = TfidfVectorizer(max_features=5000)
#     X_tfidf = vectorizer.fit_transform(X).toarray()
#     return X_tfidf, vectorizer
#
# # Train logistic regression model
# def train_logistic_regression_model(X, y):
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Define logistic regression model
#     logistic_model = LogisticRegression(max_iter=200, random_state=42)
#
#     # Train the model
#     logistic_model.fit(X_train, y_train)
#
#     # Evaluate the model
#     y_pred = logistic_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Test Accuracy: {accuracy * 100:.2f}%")
#
#     return logistic_model, accuracy
#
# # Predict sentiment percentage
# def predict_sentiment_percentage(model, vectorizer, news):
#     # Preprocess the input news
#     news_tfidf = vectorizer.transform([news]).toarray()
#
#     # Predict probabilities
#     probabilities = model.predict_proba(news_tfidf)
#     positive_percentage = probabilities[0][1] * 100  # Probability of the positive class
#     negative_percentage = probabilities[0][0] * 100  # Probability of the negative class
#
#     return positive_percentage, negative_percentage
#
# # Main function
# if __name__ == "__main__":
#     file_path = "stock_sentiment/stock_senti_analysis.csv"
#     X, y = load_and_preprocess_data(file_path)
#     X_processed, vectorizer = preprocess_features(X)
#     model, accuracy = train_logistic_regression_model(X_processed, y)
#
#     # Example news input
#     news = "The stock market is performing exceptionally well today."
#     positive, negative = predict_sentiment_percentage(model, vectorizer, news)
#     print(f"Positive Sentiment: {positive:.2f}%")
#     print(f"Negative Sentiment: {negative:.2f}%")