import streamlit as st
import pandas as pd
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import base64

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# Function to create a download link for a DataFrame as a CSV file
def get_binary_file_downloader_html(df):
    csv=df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict','Bulk Predict','Model Information'])

with tab1:
    # --- User Inputs ---
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male","Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl"," > 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal","ST-T wave abnormality","Left ventricular hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No","Yes"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping","Flat","Downsloping"])

    # --- Convert categorical inputs to numeric ---
    sex = 0 if sex=="Male" else 1
    chest_pain = ["Atypical Angina","Non-anginal Pain","Asymptomatic","Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs=="> 120 mg/dl" else 0
    resting_ecg = ["Normal","ST-T wave abnormality","Left ventricular hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina=="Yes" else 0
    st_slope = ["Upsloping","Flat","Downsloping"].index(st_slope)

    # --- Create DataFrame with user input ---
    input_data = pd.DataFrame({
        'age':[age],
        'sex':[sex],
        'chest_pain_type':[chest_pain],
        'resting_bp':[resting_bp],
        'cholesterol':[cholesterol],
        'fasting_bs':[fasting_bs],
        'resting_ecg':[resting_ecg],
        'max_hr':[max_hr],
        'exercise_angina':[exercise_angina],
        'oldpeak':[oldpeak],
        'st_slope':[st_slope]
    })

    # --- Model names and display names ---
    algonames = ['Decision Trees','Logistic Regression','Random Forest','Support Vector Machine','GridRandom']
    modelnames = ['tree.pkl','LogisticRegression.pkl','RandomForest.pkl','SVM.pkl','gridrf.pkl']

    # --- Prediction function ---
    def predict_heart_disease(data):
        # Rename columns to match training
        data = data.rename(columns={
            'age': 'Age',
            'sex': 'Sex',
            'chest_pain_type': 'ChestPainType',
            'resting_bp': 'RestingBP',
            'cholesterol': 'Cholesterol',
            'fasting_bs': 'FastingBS',
            'resting_ecg': 'RestingECG',
            'max_hr': 'MaxHR',
            'exercise_angina': 'ExerciseAngina',
            'oldpeak': 'Oldpeak',
            'st_slope': 'ST_Slope'
        })
        # Ensure correct column order
        data = data[['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                     'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']]
        
        predictions = []
        for modelname in modelnames:
            model = pickle.load(open(modelname,'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    # --- Submit button ---
    if st.button("Submit"):
        st.subheader('Prediction Results')
        st.markdown('-------------------------')

        results = predict_heart_disease(input_data)

        for i in range(len(results)):
            st.subheader(algonames[i])
            if results[i][0] == 0:
                st.write("No Heart Disease Detected")
            else:
                st.write("Heart Disease Detected")
            st.markdown('-------------------------')
with tab2:
    st.title("Upload CSV File")
    st.subheader('Instructions to note before uploading the file:')
    st.info("""
            1. No NaN values allowed.
            2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
            3. Check the spellings of the feature names.
            4. Feature values conventions: \n
               - Age: age of the patient [years]\n
               - Sex: sex of the patient [0: Male, 1: Female]\n
               - ChestPainType: chest pain type [3: Typical Angina, 2: Atypical Angina, 1: Non-anginal Pain, 0: Asymptomatic]\n
               - RestingBP: resting blood pressure [mm Hg]\n
               - Cholesterol: serum cholesterol [mg/dl]\n
               - FastingBS: fasting blood sugar [0: <= 120 mg/dl, 1: if FastingBS > 120 mg/dl]\n
               - RestingECG: resting electrocardiographic results [0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy]\n
               - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]\n
               - ExerciseAngina: exercise induced angina [0: No, 1: Yes]\n
               - Oldpeak: oldpeak = ST [Numeric value measured in depression]\n
               - ST_Slope: slope of the peak exercise ST segment [0: Upsloping, 1: Flat, 2: Downsloping]
            """)
    #Create a file uploader in the sidebar
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)
        model=pickle.load(open('LogisticRegression.pkl','rb'))

        # Ensure that the input DataFrame matches the expected columns and format
        expected_columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
        'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

        input_data['Sex'] = input_data['Sex'].map({'M':0,'F':1,'Male':0,'Female':1})
        input_data['ExerciseAngina'] = input_data['ExerciseAngina'].map({'Y':1,'N':0,'Yes':1,'No':0})
        input_data['ChestPainType'] = input_data['ChestPainType'].map({
            'TA':3,'ATA':2,'NAP':1,'ASY':0,
            'Typical Angina':3,'Atypical Angina':2,
            'Non-anginal Pain':1,'Asymptomatic':0
        })
        input_data['ST_Slope'] = input_data['ST_Slope'].map({'Up':0,'Flat':1,'Down':2,'Upsloping':0})
        input_data['RestingECG'] = input_data['RestingECG'].map({
            'Normal':0,'ST':1,'ST-T wave abnormality':1,
            'LVH':2,'Left ventricular hypertrophy':2
        })

        if set(expected_columns).issubset(input_data.columns):
            X = input_data[expected_columns]
            input_data['Prediction LR'] = model.predict(X)

            st.subheader('Predictions:')
            st.write(input_data)
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("Please make sure the uploaded CSV file contains the required columns.")
    else:
        st.info("upload a CSV file to get predictions.")
with tab3:
    import plotly.express as px

    data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 84.22,
        'GridRandom': 89.75
    }

    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(
        list(zip(Models, Accuracies)),
        columns=['Models', 'Accuracies']
    )

    fig = px.bar(
        df,
        x='Models',
        y='Accuracies',
        title='Model Accuracies Comparison',
        text=df['Accuracies'].astype(str) + '%',
        labels={'Accuracies': 'Accuracy (%)'}
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis_range=[0, 100])

    st.plotly_chart(fig, width="stretch")