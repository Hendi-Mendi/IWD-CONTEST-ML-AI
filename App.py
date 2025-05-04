import streamlit as st
import pickle
import numpy as np
import os


def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

model = load_model('BP_model.pkl')


st.title("BP Abnormality Notifier")
st.image("hrt.jpg", use_container_width=True)
st.sidebar.header("Personal Information")
st.sidebar.write("Fill in the assessment form and click on predict to assess your blood pressure")
hemo_level = st.sidebar.slider(
    "what is your current Level of Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.5, step=0.1)
pedigree = st.sidebar.slider(
    "What is your Genetic Pedigree Coefficient", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
phys_activity = st.sidebar.selectbox(
    "Do you engage in any physical activity?", options=["No", "Yes"])
chronic_kidney_disease = st.sidebar.selectbox(
    "Do you have chronic kidney disease  or a history of it in your family?", options=["No", "Yes"])
adrenal_disorder = st.sidebar.selectbox(
    "Do you have an adrenal / thyroid disorder or a history of it in your family?", options=["No", "Yes"])
bmi = st.sidebar.number_input(
    "What is your Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
sex = st.sidebar.selectbox(
    "Sex", options=["Female", "Male"])




if sex == "Male":
    sex = 1 
else:
    sex=0
if chronic_kidney_disease == "Yes":
    ckd = 1 
else:
     ckd = 0
if adrenal_disorder == "Yes":
   adrenal = 1 
else:
     adrenal = 0
if phys_activity == "Yes":
    phys=1
else:
    phys =9

features = np.array([[hemo_level, pedigree, ckd, adrenal, sex, bmi, phys -1]])

if st.button("Predict"):
    probability = model.predict_proba(features)[0,1]
    threshold = 0.5
    st.write(f"**Predicted probability of abnormal blood pressure:** {probability:.2f}")
    if probability >= threshold:
        st.error(" High risk of blood pressure abnormality detected. Please consult a doctor!")
    else:
        st.success("Your blood pressure appears normal. Stay healthy!")

