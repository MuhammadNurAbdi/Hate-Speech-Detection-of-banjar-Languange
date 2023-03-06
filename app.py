import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('banjar.csv')
df.columns = ['Text', 'Label']

# remove missing values
df = df.dropna()

# encode target label
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# establish input and output
X = list(df['Text'])
y = list(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
tfidf_train = tfidf.fit_transform(X_train)

# Fitting and transforming input data
tfidf_train = tfidf.transform(X_train)
tfidf_test = tfidf.transform(X_test)


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def main():
    st.title("TF-IDF dan SVM Klasifikasi Ujaran Kebencian Bahasa Banjar")
    news_text = st.text_area("Enter News Here", "Type Here")
    prediction_labels = {'Ujaran Kebencian': 1, 'Bukan Ujaran Kebencian': 0}
    if st.button("Classify"):
        st.text("Original Text::\n{}".format(news_text))
        vect_text = tfidf.transform([news_text])
        with open('model.pkl', 'rb') as f:
            predictor = pickle.load(f)
        prediction = predictor.predict(vect_text)
        final_result = get_key(prediction, prediction_labels)
        st.success("News Categorized as:: {}".format(final_result))


if __name__ == '__main__':
    main()
