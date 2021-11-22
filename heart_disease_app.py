

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg
import pickle

image1 = mpimg.imread('h1.png')     
image2 = mpimg.imread('h2.png')     
image3 = mpimg.imread('h3.png')     
image4 = mpimg.imread('h4.png')     
image5 = mpimg.imread('h5.png')     
image6 = mpimg.imread('h6.png')     
image7 = mpimg.imread('h7.png')     
image8 = mpimg.imread('h8.png')     
image9 = mpimg.imread('h9.png')     
image10 = mpimg.imread('h10.png')     
image11 = mpimg.imread('h11.png')     
image12 = mpimg.imread('h12.png')     
image13 = mpimg.imread('h13.png')     
image14 = mpimg.imread('h14.png')     
image15 = mpimg.imread('h15.png')     
image16 = mpimg.imread('h16.png')     
image17 = mpimg.imread('h17.png')     
image18 = mpimg.imread('h18.png')     
image19 = mpimg.imread('h19.png')     

st.write('''
# Heart Disease Detection using Streamlit
 ''')
#st.set_page_config(page_title='Heart Disease Detection Machine Learning App',layout='wide')
imageha = mpimg.imread('ha.jpg')     
st.image(imageha)
st.write("""
# Heart Disease Detection Machine Learning App

In this implementation, various **Machine Learning** algorithms are used in this app for building a **Classification Model** to **Detect Heart Disease**.
""")

data = pd.read_csv('heart.csv')
X = data[['age', 'sex', 'cp', 'trestbps', 'chol',
       'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']]
y = data['target']


Selection = st.sidebar.selectbox("Select Option", ("Heart Disease Detection App","Exploratory Data Analysis","Source Code"))

if Selection == "Heart Disease Detection App":
    
    #st.markdown('The Diabetes dataset used for training the model is:')
    #st.write(data.head(5))
    
    st.write("'cp' - chest pain")
    st.write("'testbps' - resting blood pressure (in mm Hg on admission in hospital)")
    st.write("'chol' - serum cholestrol in mg/dl")
    st.write("'fbs' - (fasting blood sugar > 120mg/dl) 1 = true, 0 = false")
    st.write("'restecg'- Rest ECG")
    st.write("'exang' - exercise induced angina")
    st.write("'oldpeak' - ST depression induced by exercise related to rest")
    st.write("'slope' - the slope of the peak exercise ST segment")
    st.write("'ca' - number of major vessels(0-3) colored by flourosopy")
    st.write("'thal' - (0-3) 3 = normal; 6 = fixed defect; 7 = reversible defect")
    st.write("'target' - 1 or 0")
    
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(data.head(5))
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
     #st.sidebar.header('2. Set Parameters'):
    age = st.sidebar.slider('age',29,77,40,1)
    cp = st.sidebar.slider('cp', 0, 3, 1, 1)
    sex = st.sidebar.slider('sex',0,1,0,1)
    trestbps = st.sidebar.slider('trestbps', 94, 200, 80, 1)
    chol = st.sidebar.slider('chol', 126, 564, 246, 2)
    fbs = st.sidebar.slider('fbs', 0, 1, 0, 1)
    restecg = st.sidebar.slider('restecg', 0, 2, 1, 1)
    exang = st.sidebar.slider('exang', 0, 1, 0, 1)
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 6.2, 3.2, 0.2)
    slope= st.sidebar.slider('slope', 0, 2, 1, 1)
    ca= st.sidebar.slider('ca', 0, 4, 2, 1)
    thal= st.sidebar.slider('thal', 0, 3, 1, 1)
    thalach = st.sidebar.slider('thalach',71,202,150,1)
    
    X_test_sc = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

    #logregs = LogisticRegression()
    #logregs.fit(X_train, y_train)
    #y_pred_st = logregs.predict(X_test_sc)

    #accuracy_score(y_train, y_pred_st)
    
    load_clf = pickle.load(open('heart_clf.pkl', 'rb'))

# Apply model to make predictions
    prediction = load_clf.predict(X_test_sc)
    prediction_proba = load_clf.predict_proba(X_test_sc)
    
    answer = prediction[0]
        
    if answer == 0:

        st.title("**The prediction is that the Heart Disease was not Detected**")
        #st.title(acc)    
    else:   
        st.title("**The prediction is that the Heart Disease was Detected**")
        
    st.write('Note: This prediction is based on the Machine Learning Algorithm, Logistic Regression.')
    
elif Selection == "Exploratory Data Analysis":
    
    st.write("'Age and Target' Countplot")
    st.image(image19)
    
    st.write("'Chest Pain' Countplot")
    st.image(image3)

    st.write("'Sex' Countplot")
    st.image(image2)
    
    st.write("'trestbps' Distplot")
    st.image(image4)
    
    st.write("'trestbps' Histogram")
    st.image(image5)
    
    st.write("'chol' Distplot")
    st.image(image6)
    
    st.write("'chol' Histogram")
    st.image(image7)
    
    st.write("'fbs' Countplot")
    st.image(image8)
    
    st.write("'restecg' Countplot")
    st.image(image9)
    
    st.write("'thalach' Distplot")
    st.image(image10)
    
    st.write("'thalach' Histogram")
    st.image(image11)
    
    st.write("'exang' Countplot")
    st.image(image12)
    
    st.write("'oldpeak' Distplot")
    st.image(image13)
    
    st.write("'oldpeak' Histogram")
    st.image(image14)
    
    st.write("'oldpeak' Histogram")
    st.image(image15)
    
    st.write("'slope' Countplot")
    st.image(image16)
    
    st.write("'ca' Countplot")
    st.image(image17)
    
    st.write("'thal' Countplot")
    st.image(image18)
    


else:
    
    st.subheader("Source Code")
    
    code = """
    
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import cv2

image1 = cv2.imread('h1.png')     
image2 = cv2.imread('h2.png')     
image3 = cv2.imread('h3.png')     
image4 = cv2.imread('h4.png')     
image5 = cv2.imread('h5.png')     
image6 = cv2.imread('h6.png')     
image7 = cv2.imread('h7.png')     
image8 = cv2.imread('h8.png')     
image9 = cv2.imread('h9.png')     
image10 = cv2.imread('h10.png')     
image11 = cv2.imread('h11.png')     
image12 = cv2.imread('h12.png')     
image13 = cv2.imread('h13.png')     
image14 = cv2.imread('h14.png')     
image15 = cv2.imread('h15.png')     
image16 = cv2.imread('h16.png')     
image17 = cv2.imread('h17.png')     
image18 = cv2.imread('h18.png')     
image19 = cv2.imread('h19.png')     


st.set_page_config(page_title='Heart Disease Detection Machine Learning App',layout='wide')
st.write(

In this implementation, various **Machine Learning** algorithms are used in this app for building a **Classification Model** to **Detect Heart Disease**.
)

data = pd.read_csv('heart.csv')
X = data[['age', 'sex', 'cp', 'trestbps', 'chol',
       'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']]
y = data['target']


Selection = st.sidebar.selectbox("Select Option", ("Heart Disease Detection App","Exploratory Data Analysis","Source Code"))

if Selection == "Heart Disease Detection App":
    
    #st.markdown('The Diabetes dataset used for training the model is:')
    #st.write(data.head(5))
    
    st.write("'cp' - chest pain")
    st.write("'testbps' - resting blood pressure (in mm Hg on admission in hospital)")
    st.write("'chol' - serum cholestrol in mg/dl")
    st.write("'fbs' - (fasting blood sugar > 120mg/dl) 1 = true, 0 = false")
    st.write("'restecg'- Rest ECG")
    st.write("'exang' - exercise induced angina")
    st.write("'oldpeak' - ST depression induced by exercise related to rest")
    st.write("'slope' - the slope of the peak exercise ST segment")
    st.write("'ca' - number of major vessels(0-3) colored by flourosopy")
    st.write("'thal' - (0-3) 3 = normal; 6 = fixed defect; 7 = reversable defect")
    st.write("'target' - 1 or 0")
    
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(data.head(5))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
     #st.sidebar.header('2. Set Parameters'):
    age = st.sidebar.slider('age',29,77,40,1)
    cp = st.sidebar.slider('cp', 0, 3, 1, 1)
    sex = st.sidebar.slider('sex',0,1,0,1)
    trestbps = st.sidebar.slider('trestbps', 94, 200, 80, 1)
    chol = st.sidebar.slider('chol', 126, 564, 246, 2)
    fbs = st.sidebar.slider('fbs', 0, 1, 0, 1)
    restecg = st.sidebar.slider('restecg', 0, 2, 1, 1)
    exang = st.sidebar.slider('exang', 0, 1, 0, 1)
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 6.2, 3.2, 0.2)
    slope= st.sidebar.slider('slope', 0, 2, 1, 1)
    ca= st.sidebar.slider('ca', 0, 4, 2, 1)
    thal= st.sidebar.slider('thal', 0, 3, 1, 1)
    thalach = st.sidebar.slider('thalach',71,202,150,1)
    
    X_test_sc = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

    logregs = LogisticRegression()
    logregs.fit(X_train, y_train)
    y_pred_st = logregs.predict(X_test_sc)
    
    answer = y_pred_st[0]
        
    if answer == 0:

        st.title("**The prediction is that the Heart Disease was not Detected**")
   
    else:   
        st.title("**The prediction is that the Heart Disease was Detected**")
        
    st.write('Note: This prediction is based on the Machine Learning Algorithm, Logistic Regression.')
    
elif Selection == "Exploratory Data Analysis":
    
    st.write("'Age and Target' Countplot")
    st.image(image19)
    
    st.write("'Chest Pain' Countplot")
    st.image(image3)

    st.write("'Sex' Countplot")
    st.image(image2)
    
    st.write("'trestbps' Distplot")
    st.image(image4)
    
    st.write("'trestbps' Histogram")
    st.image(image5)
    
    st.write("'chol' Distplot")
    st.image(image6)
    
    st.write("'chol' Histogram")
    st.image(image7)
    
    st.write("'fbs' Countplot")
    st.image(image8)
    
    st.write("'restecg' Countplot")
    st.image(image9)
    
    st.write("'thalach' Distplot")
    st.image(image10)
    
    st.write("'thalach' Histogram")
    st.image(image11)
    
    st.write("'exang' Countplot")
    st.image(image12)
    
    st.write("'oldpeak' Distplot")
    st.image(image13)
    
    st.write("'oldpeak' Histogram")
    st.image(image14)
    
    st.write("'oldpeak' Histogram")
    st.image(image15)
    
    st.write("'slope' Countplot")
    st.image(image16)
    
    st.write("'ca' Countplot")
    st.image(image17)
    
    st.write("'thal' Countplot")
    st.image(image18)
    


    """
    st.code(code, language='python')


#1 is male, 0 is female
