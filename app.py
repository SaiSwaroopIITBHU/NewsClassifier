import pickle
import streamlit as st
import gzip

with gzip.open("classifier_model_compressed.pkl.gz", "rb") as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


category_map = {
    1: 'World 🌏',
    2: 'Sports 🏏',
    3: 'Business 💼',
    4: 'Science/Technology 🔬'
}

st.set_page_config(page_title="News Category Classifier", layout="centered")

st.title("📰 News Category Classifier")

st.markdown("Paste a news headline or article text (not a link):")

user_input = st.text_area(
    label="",
    value="",
    height=100
)

st.caption("Example: 'The stock market saw a major rally today as tech shares soared.'")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    
    else:
        with st.spinner("Classifying..."):

            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]
            st.success(f"**Predicted Category:** {category_map[prediction]}")
            


st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")