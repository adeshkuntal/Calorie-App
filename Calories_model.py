import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf

#Load the Model
model = joblib.load("Calories.pkl")

st.title("Calories Prediction Model")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    print("--------------------------------->",img)
    st.image(img, caption="Uploaded Image", use_column_width=True)


dic = {"Migraine" : "Aim for increasing fruit and vegetable intake. Half your plate should be fruits and vegetables, every time!",
       "Diabetes" : "The carbohydrates in apples don’t raise your blood sugar like processed sugar because they contain fiber. That said, it is best to eat them moderately and whole. Avoid apple juice, as it is higher in sugar and does not contain fiber",
       "Cancer" : "Eating a variety of fruits during and after cancer treatment can be beneficial. Bananas, strawberries, and apples are among some of the fruits that can help relieve symptoms",
       "COVID-19" : "Having a healthy immune system is important for COVID-19 recovery. You may consider complimenting COVID-19 treatments with foods that include vitamins A, C, and D, as well as carotenoids, zinc, and omega-3 fatty acids.",
       "Heart disease" : "Diet is an important part of managing coronary heart disease. Balancing your eating with fruits, vegetables, whole grains, lean protein, and healthy fats is key",
       "none":"Eat everything"}

#Input Fields
dis = st.selectbox('Select a Disease :',["Migraine","Cancer","Diabetes","COVID-19","Heart disease","none"])



l = ["apple","banana","kiwi","mango","orange","pineapple","pomegranate"]
#Prediction
if st.button("Predict"):
    fruits = tf.keras.utils.load_img(uploaded_file,target_size=(180,180))
    print(fruits)
    arr = tf.keras.utils.img_to_array(fruits)
    arr = tf.expand_dims(arr, 0)
    prediction = model.predict(arr)
    s = tf.nn.softmax(prediction[0])
    
    b= l[np.argmax(s)]
    print(b)
    st.write(b)
    
    d = {"apple":95, "banana":105, "kiwi":44, "mango":201, "orange":45, "pineapple":450, "pomeganate":233}
    for i in d.keys():
        if(i==b):
            c = d[i]
            st.write("Calorie : ", d[i])
    for i in dic:
        if(dis==i):
            st.write(dic[i])
    
    
             
             
