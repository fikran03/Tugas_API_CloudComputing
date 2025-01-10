from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import streamlit as st
import pickle

# Memuat dataset
dataset = load_dataset("mail4dy/phishing-detection-dataset")
data = dataset['train']

# Membagi data menjadi fitur (X) dan label (y)
X = data['text']
y = data['label']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat pipeline model
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))),
    ('clf', MultinomialNB())
])

# Melatih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Menyimpan model ke file
with open('phishing_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Streamlit Web App
st.title("Phishing Detection Web App")

# Muat model
with open('phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Input teks dari pengguna
text = st.text_area("Enter the text or email content to classify:")

if st.button("Predict"):
    # Prediksi dengan model
    prediction = model.predict([text])[0]
    result = "Phishing" if prediction == 1 else "Not Phishing"
    st.write(f"The message is classified as: {result}")
