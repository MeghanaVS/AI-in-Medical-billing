
/content/drive/MyDrive/Colab Notebooks/icd-9dataset-final.txt

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import csv

input_txt_file = '/content/drive/MyDrive/Colab Notebooks/icd-9dataset-final.txt'
output_csv_file = '/content/drive/MyDrive/Colab Notebooks/icd-9dataset-final.csv'

with open(input_txt_file, 'r') as txt_file:
    lines = txt_file.readlines()

data = [line.strip().split('\t') for line in lines]

with open(output_csv_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the data to the CSV file
    csv_writer.writerows(data)

print(f'Conversion from {input_txt_file} to {output_csv_file} completed.')

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/icd-9dataset-final.csv', sep=',')

import pandas as pd

# Read the CSV file with one column
input_file = '/content/drive/MyDrive/Colab Notebooks/icd-9dataset-final.csv'
df = pd.read_csv(input_file, header=None, names=['icd code,Description,DiagnosisNotes'])

# Split the single column into three columns by ","
df[['icd code', 'Description', 'DiagnosisNotes']] = df['icd code,Description,DiagnosisNotes'].str.split(',', n=2, expand=True)

# Drop the original 'Data' column
df = df.drop(columns=['icd code,Description,DiagnosisNotes'])

# Save the DataFrame to a new CSV file with three columns
output_file = 'output.csv'
df.to_csv(output_file, index=False)

print(f'CSV file with three columns saved as {output_file}')

df= pd.read_csv("output.csv")
df = df.iloc[1:]

# Reset the index
df.reset_index(drop=True, inplace=True)
df= df.dropna(how='all')

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords and non-alphanumeric tokens, and lemmatizing the tokens.

    Args:
        text: The text to preprocess.

    Returns:
        A preprocessed string.
    """
    if pd.isna(text):
        return ''

    # Tokenization
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Convert tokens back to a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Check if the 'DiagnosisNotes' column exists
if 'DiagnosisNotes' in df.columns:
    # Apply preprocessing to the 'DiagnosisNotes' column
    df['DiagnosisNotes'] = df['DiagnosisNotes'].apply(preprocess_text)

# Check if the 'DiagnosisNotes' column exists
if 'DiagnosisNotes' in data.columns:

    # Transform the data into TF-IDF vectors
    X_tfidf = tfidf_vectorizer.fit_transform(data['DiagnosisNotes'])

    # Create the DiagnosisNotes column if it does not exist
elif 'DiagnosisNotes' not in data.columns:
    data['DiagnosisNotes'] = ''

else:

    # Handle the case where the 'DiagnosisNotes' column does not exist
    print('The DiagnosisNotes column does not exist in the data DataFrame.')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed

# Transform the data into TF-IDF vectors
X_tfidf = tfidf_vectorizer.fit_transform(df['DiagnosisNotes'])

# Text Preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Convert tokens back to a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Apply preprocessing to the 'DiagnosisNotes' column
df['DiagnosisNotes'] = df['DiagnosisNotes'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(df['DiagnosisNotes'])

# Convert TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Add the 'icd code' and 'Description' columns back to the DataFrame
tfidf_df[['icd code', 'Description']] = df[['icd code', 'Description']]

# Display the preprocessed and vectorized DataFrame
print(tfidf_df.head())

# Define the lemmatize_and_remove_stopwords function
def lemmatize_and_remove_stopwords(text):
    # Create a lemmatizer within the function
    lemmatizer = WordNetLemmatizer()

    if isinstance(text, str):
        # Tokenize the sentence
        tokens = nltk.word_tokenize(text.lower())

        # Remove stopwords and non-alphanumeric tokens
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

        # Lemmatization using the local lemmatizer
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Convert tokens back to a single string
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    else:
        # Handle non-string or missing values (e.g., NaN)
        return text

# Load the ICD code and description data
icd_code_data = pd.read_csv('output.csv')

# Preprocess the ICD code and description data
icd_code_data['icd code'] = icd_code_data['icd code'].apply(lemmatize_and_remove_stopwords)
icd_code_data['Description'] = icd_code_data['Description'].apply(lemmatize_and_remove_stopwords)

# Fill missing values in the 'icd code' column with an empty string
icd_code_data['icd code'].fillna('', inplace=True)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the ICD code and description data into TF-IDF vectors
X = vectorizer.fit_transform(icd_code_data['icd code'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, icd_code_data['Description'], test_size=0.25, random_state=42)

# Drop rows with missing values from X_train and y_train
X_train = X_train[~pd.isna(y_train)]  # Use pd.isna to check for missing values
y_train = y_train[~pd.isna(y_train)]  # Use pd.isna to check for missing values

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print('Accuracy:', accuracy)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print('Accuracy:', accuracy)

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
y_train = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
y_test = encoder.transform(np.array(y_test).reshape(-1, 1))

# Create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define the number of classes in your classification problem
num_classes = len(np.unique(y_train))

# Define XGBoost parameters
params = {
    "objective": "multi:softmax",  # for multiclass classification
    "num_class": num_classes,  # number of classes
    "max_depth": 3,
    "eta": 0.1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "eval_metric": "mlogloss"
}

# Train the XGBoost model
num_round = 100
model = xgb.train(params, dtrain, num_round)

# Make predictions
y_pred = model.predict(dtest)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

models = {
    "Support Vector Machine": SVC(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Model: {model_name}')
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

