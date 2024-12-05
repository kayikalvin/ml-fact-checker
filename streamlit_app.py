import streamlit as st

st.title('🎈 App Name')

import streamlit as st
from transformers import pipeline

# Load pre-trained fact-checking model
fact_checker = pipeline("zero-shot-classification", model="distilbert-base-uncased")

# Define the labels for classification (can be modified)
labels = ["True", "False"]

# Function to check fact validity
def check_fact(statement):
    result = fact_checker(statement, candidate_labels=labels)
    return result['labels'][0], result['scores'][0]

# Streamlit Interface
st.title("Fact Checker")

# Instructions
st.write("Enter a statement below to check its validity.")

# User input
statement = st.text_area("Enter the statement:", height=150)

if st.button("Check"):
    if statement:
        label, score = check_fact(statement)
        st.write(f"The statement is: **{label}** with a confidence score of {score:.2f}")
    else:
        st.write("Please enter a statement to check.")

