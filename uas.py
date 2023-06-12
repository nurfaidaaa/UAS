import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# import warnings
# warnings.filterwarnings("ignore")


st.title("UAS PENDATA")
st.write("##### Nama  : Nurfaida Oktafiani")
st.write("##### Nim   : 210411100078")
st.write("##### Kelas : Penambangan Data B ")

description, upload_data, preprocessing, modeling, implementation = st.tabs(["Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

with description:
    st.write("###### Aplikasi ini digunakan untuk memprediksi seseorang terkena penyakit batu ginjal atau tidak dengan menggunakan beberapa metode seperti Naive Bayes,K-NN, Decision Tree & MLP ")
    st.write("""# Dekripsi Dataset """)
    st.write("###### Data Set Ini Adalah : Kidney Diagnosis (Prediksi Penyakit batu Ginjal) ")
    st.write("###### Sumber Dataset ini diambil dari Kaggle :https://www.kaggle.com/datasets/mansoordaku/ckdisease")
    st.write(" Dataset ini dapat digunakan untuk memprediksi seseorang terkena batu ginjal atau tidak. ")
    st.write("Terdapat Sebanyak 400 data tipe numerik yang digunakan untuk memprediksi seseorang terkena penyakit batu ginjal atau tidak")
    st.write("""# Deskripsi Data""")
    st.write("Total datanya adalah 400 data, dan terdapat 23 atribut")
    st.write("##### Output:  ")
    st.write ("""0 = tidak terkena batu ginjal""")
    st.write ("""1 = terkena batu ginjal""")

with upload_data:
    st.write("""# Dataset Asli """)
    df = pd.read_csv('https://raw.githubusercontent.com/nurfaidaaa/pendata/master/kidney_d.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
   
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['classification'])
    y = df['classification'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('output Label')
    dumies = pd.get_dummies(df.classification).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]]
    })

    st.write(labels)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.20, random_state=42)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.20, random_state=42)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        neural = st.checkbox('MLP')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=7
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        #ANNNBP
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=1)
        mlp.fit(training, training_label)
        mlp_pred = mlp.predict(test)
        mlp_akurasi = round(100 * accuracy_score(test_label, mlp_pred))


        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            if neural:
                st.write("Model ANNBP accuracy score : {0:0.2f}".format(mlp_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi, mlp_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree','MLP'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  ##Implementation
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        bp  = st.number_input('Masukkan nilai bp') # sesuai data set
        sg  = st.number_input('Masukkan nilai sg')
        al  = st.number_input('Masukkan nlai al')
        su  = st.number_input('Masukkan nilai su')
        rbc  = st.number_input('Masukkan nilai rbc')
        pc  = st.number_input('Masukkan nilai pc')
        pcc  = st.number_input('Masukkan nilai pcc')
        ba  = st.number_input('Masukkan nilai ba')
        bgr  = st.number_input('Masukkan nilai bgr')
        bu = st.number_input('Masukkan nilai bu')
        sc  = st.number_input('Masukkan nilai sc')
        sod  = st.number_input('Masukkan nilai sod')
        pot  = st.number_input('Masukkan nilai pot')
        hemo  = st.number_input('Masukkan nilai hemo')
        pcv  = st.number_input('Masukkan nilai pcv')
        wc  = st.number_input('Masukkan nilai wc')
        rc  = st.number_input('Masukkan nilai rc')
        htn  = st.number_input('Masukkan nilai htn')
        dm  = st.number_input('Masukkan nilai dm')
        cad  = st.number_input('Masukkan nilai cad')
        appet = st.number_input('Masukkan nilai appet')
        pe = st.number_input('Masukkan nilai pe')
        ane = st.number_input('Masukkan nilai ane')
        
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree','MLP'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt
            if model == 'MLP':
                mod = mlp

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
