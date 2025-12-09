import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, RocCurveDisplay

# Modern Custom CSS (prompted GPT to help me with the CSS styling, prompt was "please help me find a nice modern looking font and style the page to use LinkedIn Blue" )
## using what they wrote, used the shell to tune what exactly want the page to look like. 
st.markdown("""
            
    <style>
    /* Import modern font (This will use the same font that LinkedIn Uses) 
     Will get 4 different weights to determine the size of the font) */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
        
    /* Global styling (this will make sure that all the text will be using this font)
            
    Adding the check to in case inter fails , san-serif will be used*/

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main content area
    want to add some animations
    */
    .main {
        padding: 2rem 3rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        animation : gradientShift 5s ease infinite;
    }
    @keyframes graidentShift { 
        0% {
            background-position: 0% 50%;}  
        50% {
            background-position: 100% 50%;} 
        100% {
            background-position: 0% 50%;}   
            }
    /* want to add smooth scrolling 
    This will make it so you can scroll smoothly without the page jumping or tearing
    */
    *{ scroll-behavior: smooth;}
    
    /* Headers */
    h1 {
        color: #0A66C2;
        padding-bottom: 1rem;
        border-bottom: 3px solid #0A66C2;
    }
    
    h2 {
        color: #0A66C2;
        margin-top: 2rem;
    }
    
    h3 {
        color: #0A66C2;
    }
    
    /* Sidebar styling -- Make It a LinkedIn Blue Background */
            
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A66C2 0%, #004182 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #0A66C2;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #5e6c84;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0A66C2 0%, #0A66C2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-weight: 500;
        color: #0A66C2 !important;
    }
    
    /* Slider */
    .stSlider > label {
        font-weight: 500;
        color: #0A66C2 !important;
    }
    /* Still need to change the slider color to blue */
            
    .stSlider [data-baseweb = 'slider'] > div > div {
            background-color: #e3f2fd !important; }
    .stSlider [data-baseweb = 'slider'] > div > div > div { 
            background-color: #0A66C2 !important; }
    .stSlider > div > div > div > div[data-baseweb="slider"] > div:last-child > div {
            background-color = #0A66C2 !important; border: 2px solid white !important; }
    
    /* Info boxes */
    .element-container div[data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError {
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #0A66C2, transparent);
    }
    
    </style>
""", unsafe_allow_html=True)

## Create the Page, set up the sidebar
st.set_page_config(
    page_title = 'LinkedIn Usage Prediction Project' , 
    layout= 'wide' ,
    initial_sidebar_state= 'expanded'
)

## Read in the data then clean the data 
s = pd.read_csv('social_media_usage.csv')

def clean_sm(x):
    x = np.where(x == 1 , 1, 0)
    return x

ss = pd.DataFrame(
    {
        'sm_li' : clean_sm(s['web1h']),
        'income' : np.where(s['income'] <= 9 , s['income'] , np.nan),
        'education': np.where(s['educ2'] <= 8 , s['educ2'], np.nan),
        'parent' : clean_sm(s['par']),
        'married' : clean_sm(s['marital']),
        'female' : np.where(s['gender'] == 2, 1, 0),
        'age' : np.where(s['age'] < 98 , s['age'] , np.nan)
    } 
)

# want to create a model that uses drop na (ss)
## want to create a model that uses KNN imputation and a pipeline to
ss1 = ss.copy()
ss= ss.dropna()


## visualize EDA
## create a value to see the LinkedIn user distribution
sm_chart = alt.Chart(ss).mark_bar().encode(
    x=alt.X('sm_li:N', title='LinkedIn User (0 = Not a User, 1 = LinkedIn User)'),
    y=alt.Y('count()', title='Value Counts'),
    tooltip=['sm_li:N', 'count()']
).properties(
    width=300,
    height=300,
    title='LinkedIn User Distribution'
)

sm_chart_income = alt.Chart(ss).mark_bar().encode(
    x=alt.X('income:O', title='Income Level (1 = Low , 9 = High)'), 
    y=alt.Y('count()', title='Value Count per Income Class'),
    color=alt.Color('sm_li:N', legend=alt.Legend(title='LinkedIn User')),
    tooltip=['income:O', 'sm_li:N', 'count()']
).properties(
    width=300,
    height=300,
    title='LinkedIn Usage by Income Level'
)

sm_chart_edu = alt.Chart(ss).mark_bar().encode(
    x=alt.X('education:O', title='Education Level (1 = Low , 8 = High)'), 
    y=alt.Y('count()', title='Value Count per Education Level'),
    color=alt.Color('sm_li:N', legend=alt.Legend(title='LinkedIn User')),
    tooltip=['education:O', 'sm_li:N', 'count()']
).properties(
    width=300,
    height=300,
    title='LinkedIn Usage by Education Level'
)

sm_chart_parent = alt.Chart(ss).mark_bar().encode(
    x=alt.X('parent:O', title='Parent (0 = No , 1 = Yes)'), 
    y=alt.Y('count()', title='Value Count by Parent Status'),
    color=alt.Color('sm_li:N', legend=alt.Legend(title='LinkedIn User')),
    tooltip=['parent:O', 'sm_li:N', 'count()']
).properties(
    width=300,
    height=300,
    title='LinkedIn Usage by Parent Status'
)

sm_chart_married = alt.Chart(ss).mark_bar().encode(
    x=alt.X('married:O', title='Marital Status (0 = No , 1 = Yes)'), 
    y=alt.Y('count()', title='Value Count by Marital Status'),
    color=alt.Color('sm_li:N', legend=alt.Legend(title='LinkedIn User')),
    tooltip=['married:O', 'sm_li:N', 'count()']
).properties(
    width=150,
    height=300,
    title='LinkedIn Usage by Marital Status'
)

sm_chart_gender = alt.Chart(ss).mark_bar().encode(
    x=alt.X('female:O', title='Gender (0 = Male , 1 = Female)'), 
    y=alt.Y('count()', title='Value Count by Gender'),
    color=alt.Color('sm_li:N', legend=alt.Legend(title='LinkedIn User')),
    tooltip=['female:O', 'sm_li:N', 'count()']
).properties(
    width=150,
    height=300,
    title='LinkedIn Usage by Gender'
)

sm_chart_age = alt.Chart(ss).mark_area().encode(
    x=alt.X('age:Q', title='Age'),
    y=alt.Y('count()', stack=None, title='Count'),
    color=alt.Color('sm_li:N', legend=alt.Legend(title='LinkedIn User')),
    xOffset='sm_li:N',
    tooltip=['age:Q', 'sm_li:N', 'count()']
).properties(
    width=600,
    height=300,
    title='LinkedIn Usage by Age'
)

# Combine all charts into one dashboard
row1 = alt.hconcat(sm_chart_income, sm_chart_edu, sm_chart_parent)
row2 = alt.hconcat(sm_chart_married, sm_chart_gender, sm_chart_age)

# Final Dashboard
dashboard = alt.vconcat(
    row1,
    row2
).properties(
    title=alt.TitleParams(
        text='LinkedIn Usage - Exploratory Data Analysis Dashboard',
        fontSize=20,
        fontWeight='bold',
        anchor='middle'
    )
).configure_axis(
    labelFontSize=11,
    titleFontSize=12
).configure_title(
    fontSize=14,
    fontWeight='bold'
)

# the target variable is y 

target_variables = ss['sm_li']
target_variables_advanced = ss1['sm_li']
# define the features 

features = ss.drop('sm_li' , axis =1)
features_advanced = ss1.drop('sm_li' , axis =1)

X_train_simple , X_test_simple , y_train_simple , y_test_simple = train_test_split(
    features, 
    target_variables, 
    test_size= 0.20, 
    random_state=42, 
    stratify=target_variables
)



X_train_advanced , X_test_advanced , y_train_advanced , y_test_advanced = train_test_split(
    features_advanced, 
    target_variables_advanced, 
    test_size= 0.20, 
    random_state=42, 
    stratify=target_variables_advanced
)

## instatiate the simple logisitic regression model 
simple_lg = LogisticRegression(class_weight= 'balanced' , random_state= 42 , max_iter = 1000)
simple_lg.fit(X_train_simple, y_train_simple)
## make predictions on train and test set
y_pred_lg_train = simple_lg.predict(X_train_simple)
y_pred_lg_test = simple_lg.predict(X_test_simple)
## Get probabilities for the train and test set 
y_pred_proba_train = simple_lg.predict_proba(X_train_simple)[:,1]
y_pred_proba_test = simple_lg.predict_proba(X_test_simple)[:,1]
## calculate metrics and correlation table 
cm_train_simple = confusion_matrix(y_train_simple, y_pred_lg_train)
cm_test_simple = confusion_matrix(y_test_simple, y_pred_lg_test)
accuracy_train_simple = accuracy_score(y_train_simple, y_pred_lg_train)
accuracy_test_simple = accuracy_score(y_test_simple, y_pred_lg_test)
precision_test_simple = precision_score(y_test_simple, y_pred_lg_test)
recall_test_simple = recall_score(y_test_simple, y_pred_lg_test)
f1_test_simple = f1_score(y_test_simple, y_pred_lg_test)

## for the advanced model , want to use cross validation and pipeline with KNN imputation

cv = StratifiedKFold(n_splits = 5 , shuffle = True , random_state=42)

## set up the pipeline for the advanced dataset

knn_pipeline = Pipeline(steps = [ 
    ('imputer' , KNNImputer(n_neighbors=5)),
    ('scaler' , StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced' , max_iter=2500 , random_state=42))
])

scores_knn = cross_val_score(
    knn_pipeline,
    X_train_advanced,
    y_train_advanced,
    cv=cv,
    scoring='accuracy'
)

# calculate precision and recall
knn_pipeline.fit(X_train_advanced, y_train_advanced)
y_pred_train = knn_pipeline.predict(X_train_advanced)
y_pred_test = knn_pipeline.predict(X_test_advanced)

cm_train = confusion_matrix(y_train_advanced, y_pred_train)
cm_test = confusion_matrix(y_test_advanced, y_pred_test)
accuracy_train = accuracy_score(y_train_advanced, y_pred_train)
accuracy_test = accuracy_score(y_test_advanced, y_pred_test)
precision_train = precision_score(y_train_advanced, y_pred_train)
precision_test = precision_score(y_test_advanced, y_pred_test)
recall_train = recall_score(y_train_advanced, y_pred_train)
recall_test = recall_score(y_test_advanced, y_pred_test)
f1_test_simple = f1_score(y_test_advanced, y_pred_test)


st.sidebar.markdown('## Navigation')
page = st.sidebar.radio(
    'Select Page:',
    ['Data Exploration & Model Creation', 'Make Predictions'],
    index=0
)

st.sidebar.markdown('---')
st.sidebar.markdown(f'''
### Model Performance

Simple Model (No NAs):
- Accuracy on Test set: {accuracy_test_simple * 100:.1f}%
- Precision on Test set: {precision_test_simple * 100:.1f}%
- Recall on Test set: {recall_test_simple * 100:.1f}%

Advanced Model (KNN Imputation):
- Accuracy on Test set: {accuracy_test * 100:.1f}%
- Precision on Test set: {precision_test * 100:.1f}%
- Recall on Test set: {recall_test * 100:.1f}%
- CV Score: {scores_knn.mean() * 100:.1f}% ± {scores_knn.std() * 100:.1f}%
'''
)

if page == 'Data Exploration & Model Creation':
    st.markdown('## Exploratory Data Analysis')
    st.markdown('---')
    st.markdown('### Dataset Overview')
    col1_simple, col2_simple, col3_simple = st.columns(3)
    with col1_simple:
        st.metric('Total Samples in Simple Model', f'{len(ss):,}')
    with col2_simple:
        linkedin_rate_simple = ss['sm_li'].mean() * 100
        st.metric('LinkedIn Users', f'{linkedin_rate_simple:.1f}%')
    with col3_simple:
        st.metric('Features', '6')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Samples in KNN imputation model', f'{len(ss1):,}')
    with col2:
        linkedin_rate = ss1['sm_li'].mean() * 100
        st.metric('LinkedIn Users', f'{linkedin_rate:.1f}%')
    with col3:
        st.metric('Features', '6')
    st.markdown('---')
    st.markdown('## Key Insights')
    st.markdown('- Education is the Strongest Predictor: A one-unit increase in education level increases LinkedIn usage odds by 40%, confirming that more educated individuals actively seek professional opportunities on the platform.')
    st.markdown('- Income is Highly Influential: A one-unit increase in income increases LinkedIn usage odds by 35%, validating that higher-income professionals prioritize networking through LinkedIn.')
    st.markdown('- Strong Model Performance: Low multicollinearity between predictors (confirmed via correlation matrix) enabled effective modeling, producing a logistic regression model achieving 0.76 ROC-AUC')
    st.markdown('---')
    st.altair_chart(dashboard , height = 900 , width = 900)

    st.markdown('---')

    st.markdown('LinkedIn Usage Correlation Matrix')
    
    corr_matrix = ss.corr()

    # Create correlation matrix
    corr_matrix = ss.corr()

    # Convert to long format for Altair
    corr_data = corr_matrix.stack().reset_index()
    corr_data.columns = ['Feature1', 'Feature2', 'Correlation']

    # Create heatmap
    fig_corr = alt.Chart(corr_data).mark_rect().encode(
        x=alt.X('Feature1:N', title='Features'),
        y=alt.Y('Feature2:N', title='Features'),
        color=alt.Color('Correlation:Q'),
        tooltip=['Feature1', 'Feature2', alt.Tooltip('Correlation:Q', format='.2f')]
    ).properties(
        width=900,
        height=900,
        title='LinkedIn Usage Correlation Matrix'
    )
    st.altair_chart(fig_corr, use_container_width=True)

    st.markdown('---')

    ## EDA complete , now will show model confusion matrix for both models

    st.markdown('### Confusion Matrix for Simple Model')
    col_train , col_test = st.columns(2)
    with col_train:
        st.markdown('#### Training Set')
        df_simple_train = pd.DataFrame(
            cm_train_simple, 
            index = ['Actual: Not LinkedIn User' , 'Actual: LinkedIn User'],
            columns=  ['Prediction: Not LinkedIn User' , 'Prediction: LinkedIn User']
        )
        
        df_simple_train_pivot = df_simple_train.reset_index()
        df_simple_train_pivot = df_simple_train_pivot.melt(id_vars='index', var_name='Predicted', value_name='Count')
        df_simple_train_pivot.columns = ['Actual', 'Predicted', 'Count']

        cm_train_plot = alt.Chart(df_simple_train_pivot).mark_rect().encode(
            x = alt.X('Predicted' , title = 'Predicted'),
            y = alt.Y('Actual' , title = 'Actual'),
            color = alt.Color('Count'),
            tooltip = ['Actual' , 'Predicted' , 'Count']
        ).properties(
            width = 450,
            height = 450, 
            title = 'Confusion Matrix for the Training Set '
        )
        st.altair_chart(cm_train_plot)
    with col_test:
        st.markdown('#### Testing Set')
        df_simple_test = pd.DataFrame(
            cm_test_simple, 
            index = ['Actual: Not LinkedIn User' , 'Actual: LinkedIn User'],
            columns=  ['Prediction: Not LinkedIn User' , 'Prediction: LinkedIn User']
        )
        
        df_simple_test_pivot = df_simple_test.reset_index()
        df_simple_test_pivot = df_simple_test_pivot.melt(id_vars='index', var_name='Predicted', value_name='Count')
        df_simple_test_pivot.columns = ['Actual', 'Predicted', 'Count']

        cm_test_plot = alt.Chart(df_simple_test_pivot).mark_rect().encode(
            x = alt.X('Predicted' , title = 'Predicted'),
            y = alt.Y('Actual' , title = 'Actual'),
            color = alt.Color('Count'),
            tooltip = ['Actual' , 'Predicted' , 'Count']
        ).properties(
            width = 450,
            height = 450, 
            title = 'Confusion Matrix for the Testing Set '
        )
        st.altair_chart(cm_test_plot)

    st.markdown('---')
    st.markdown('### Confusion Matrix for KNN Imputation Model')
    col_train2 , col_test2 = st.columns(2)
    with col_train2:
        st.markdown('#### Training Set')
        df_train = pd.DataFrame(
            cm_train, 
            index = ['Actual: Not LinkedIn User' , 'Actual: LinkedIn User'],
            columns=  ['Prediction: Not LinkedIn User' , 'Prediction: LinkedIn User']
        )
        
        df_train_pivot = df_train.reset_index()
        df_train_pivot = df_train_pivot.melt(id_vars='index', var_name='Predicted', value_name='Count')
        df_train_pivot.columns = ['Actual', 'Predicted', 'Count']

        cm_train_fig = alt.Chart(df_train_pivot).mark_rect().encode(
            x = alt.X('Predicted' , title = 'Predicted'),
            y = alt.Y('Actual' , title = 'Actual'),
            color = alt.Color('Count'),
            tooltip = ['Actual' , 'Predicted' , 'Count']
        ).properties(
            width = 450,
            height = 450, 
            title = 'Confusion Matrix for the Training Set '
        )
        st.altair_chart(cm_train_fig)
    with col_test2:
        st.markdown('#### Testing Set')
        df_test = pd.DataFrame(
            cm_test, 
            index = ['Actual: Not LinkedIn User' , 'Actual: LinkedIn User'],
            columns=  ['Prediction: Not LinkedIn User' , 'Prediction: LinkedIn User']
        )
        
        df_test_pivot = df_test.reset_index()
        df_test_pivot = df_test_pivot.melt(id_vars='index', var_name='Predicted', value_name='Count')
        df_test_pivot.columns = ['Actual', 'Predicted', 'Count']

        cm_test_fig = alt.Chart(df_test_pivot).mark_rect().encode(
            x = alt.X('Predicted' , title = 'Predicted'),
            y = alt.Y('Actual' , title = 'Actual'),
            color = alt.Color('Count'),
            tooltip = ['Actual' , 'Predicted' , 'Count']
        ).properties(
            width = 450,
            height = 450, 
            title = 'Confusion Matrix for the Training Set '
        )
        st.altair_chart(cm_test_fig)
    st.markdown('---')
    col1 , col2 = st.columns(2)
    with col1:
        st.markdown('Advanced Model (KNN Imputation)')
        # Fit the pipeline
        knn_pipeline.fit(X_train_advanced, y_train_advanced)

        # Get predicted probabilities (need probabilities for ROC, not just predictions)
        y_pred_proba_adv = knn_pipeline.predict_proba(X_test_advanced)[:, 1]  # probability of class 1

        # Calculate ROC curve
        fpr_adv, tpr_adv, thresholds = roc_curve(y_test_advanced, y_pred_proba_adv)

        # Calculate AUC score
        auc_score_adv = roc_auc_score(y_test_advanced, y_pred_proba_adv)

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))  # ← Use fig, ax for Streamlit
        ax.plot(fpr_adv, tpr_adv, color='#0A66C2', lw=2, label=f'ROC curve (AUC = {auc_score_adv:.3f})') 
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random guess')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
        st.pyplot(fig)  # ← Use st.pyplot() instead of plt.show()
    
        # Display AUC score
        st.metric("AUC Score", f"{auc_score_adv:.4f}")

        print(f"AUC Score: {auc_score_adv:.4f}")
    with col2:
        st.markdown('Simple Model')
        # Fit the pipeline
        simple_lg.fit(X_train_simple, y_train_simple)

        # Get predicted probabilities (need probabilities for ROC, not just predictions)
        y_pred_proba = simple_lg.predict_proba(X_test_simple)[:, 1]  # probability of class 1

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test_simple, y_pred_proba)

        # Calculate AUC score
        auc_score = roc_auc_score(y_test_simple, y_pred_proba)

        # Plot ROC curve
        fig1, ax1 = plt.subplots(figsize=(8, 6))  # ← Use fig, ax for Streamlit
        ax1.plot(fpr, tpr, color='#0A66C2', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')  # ← FIXED variable name
        ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random guess')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
    
        st.pyplot(fig1)  # ← Use st.pyplot() instead of plt.show()
    
        # Display AUC score
        st.metric("AUC Score", f"{auc_score:.4f}")

        print(f"AUC Score: {auc_score:.4f}")
    st.markdown('---')
elif page == 'Make Predictions':
    

## rename the labels so the sliders will make sense

    income_labels = {
    1: 'Less than $10,000',
    2: '$10,000 to under $20,000',
    3: '$20,000 to under $30,000',
    4: '$30,000 to under $40,000',
    5: '$40,000 to under $50,000',
    6: '$50,000 to under $75,000',
    7: '$75,000 to under $100,000',
    8: '$100,000 to under $150,000',
    9: '$150,000 or more'
    }

    education_labels = {
    1: 'Less than high school',
    2: 'High school incomplete',
    3: 'High school graduate',
    4: 'Some college, no degree',
    5: 'Associate degree',
    6: "Bachelor's degree",
    7: 'Some graduate school',
    8: 'Graduate degree'
    }
    parent_labels = {
    0: 'Not A Parent',
    1: 'Parent'
    }
    gender_labels = {
    0: 'Male',
    1: 'Female'
    }
    married_labels = {
    0: 'No',
    1: 'Yes'
    }
    income_keys = list(income_labels.keys())
    income_display = [income_labels[i] for i in income_keys]

    education_keys = list(education_labels.keys())
    education_display = [education_labels[i] for i in education_keys]

    parent_keys = list(parent_labels.keys())
    parent_display = [parent_labels[i] for i in parent_keys]

    gender_keys = list(gender_labels.keys())
    gender_display = [gender_labels[i] for i in gender_keys]

    married_keys = list(married_labels.keys())
    married_display = [married_labels[i] for i in married_keys]

    st.markdown('### Choose your Model')
    which_model = st.radio(
        '## Which Model Do You Want to Use?',
        ['Simple Model (Dropped NaNs)' , 'Advanced Model (KNN Impuation and Cross Validation)'], 
        horizontal = True
    )
    st.markdown('---')
    st.markdown('### Please Enter Your Profile Information')
    col1_design , col2_design = st.columns(2)
    with col1_design:
        get_income = st.select_slider(
            '### Income Level',
            options = income_keys,
            value = 5, 
            format_func=dict(zip(income_keys, income_display)).get
        )
        st.markdown('---')
        get_education = st.select_slider(
            '### Education Level',
            options = education_keys,
            value = 5, 
            format_func = dict(zip(education_keys, education_display)).get
        )

        st.markdown('---')
        get_age = st.slider(
            '### Age',
            min_value = 18, 
            max_value = 97,
            value = 50, 
            step = 1
        )

        st.markdown('---')
    
    with col2_design:
        get_parent = st.radio(
            '### Are you a Parent to a Child 18 or Younger?',
            options = [0,1],
            format_func= dict(zip(parent_keys , parent_display)).get,
            horizontal = True

        )
        st.markdown('---')

        get_married = st.radio(
            '### Are You Married?',
            options = [0,1],
            format_func=dict(zip(married_keys, married_display)).get,
            horizontal = True

        )

        st.markdown('---')

        get_gender = st.radio(
            '### What is Your Gender?',
            options = [0,1],
            format_func= dict(zip(gender_keys, gender_display)).get,
            horizontal = True
        )

        st.markdown('---')
    ## create the predict buttom 
    if st.button('## Predict LinkedIn Usage' , type = 'primary'):
        person_df = pd.DataFrame({
            'income' : [get_income],
            'education': [get_education],
            'parent' : [get_parent],
            'married': [get_married],
            'female' : [get_gender],
            'age' : [get_age]
        })
        
    ## now we need to define a make preditions button
        if 'Simple Model (Dropped NaNs)' in which_model:
            predictions = simple_lg.predict(person_df)[0]
            probability = simple_lg.predict_proba(person_df)[0]
            model_used = 'Simple Model (Dropped NaNs)'

        else: 
            predictions = knn_pipeline.predict(person_df)[0]
            probability = knn_pipeline.predict_proba(person_df)[0]
            model_used = 'Advanced Model (KNN Impuation and Cross Validation)'
        st.markdown('---')

        st.markdown(f'### Prediction Results ({model_used})')
        
        if predictions == 1:
            st.success('### You Are A LinkedIN User')
        else:
            st.error('### You Are Not A LinkedIn User')
        ## provide metrics
        col_1 , col_2 = st.columns(2)
        with col_1:
            st.metric(
                '### LinkedIn User Probability',
                f'{probability[1]*100:.2f}%'
            )
        with col_2:
            st.metric(
                'Not a LinkedIn User Probability',
                f'{probability[0]*100:.2f}%'
            )

        st.markdown('---')

        ## marginal effect analysis 

        st.markdown( '### Marginal Effect Analysis')
        st.markdown('### Check how changing each feature affects your LinkedIn Usage Probabilities')

        # need to get the model coefficients 
        if 'Simple Model (Dropped NaNs)' in which_model:
            coefficients = simple_lg.coef_[0]

        else: 
            coefficients = knn_pipeline.named_steps['model'].coef_[0]
        feature_names = ['Income' , 'Education' , 'Parent' , 'Married' , 'Gender' , 'Age']

        ## calculate the odds ratios 
        odds_ratio = np.exp(coefficients)

        ## can start to create and visualize this dataframe 
        bar_me_df = pd.DataFrame({
            'Feature' : feature_names,
            'Odds Ratio' : odds_ratio,
            'Impact' : ['Increase' if i > 1 else 'Decrease' for i in odds_ratio]
        })
        ### sort the dataframe
        bar_me_df = bar_me_df.sort_values('Odds Ratio' , ascending=False)

        ## create a visualization, bar chart 
        fig_me = alt.Chart(bar_me_df).mark_bar().encode(
            x = alt.X('Odds Ratio:Q', title = 'Odds Ratio'),
            y = alt.Y('Feature:N', sort = '-x', title = 'Features'),
            color = alt.Color('Impact:N'),
            tooltip = ['Feature' , 'Odds Ratio' , 'Impact']
        ).properties(
            
            title = 'Marginal Effects of Each Feature Used'
    
        )
        st.altair_chart(fig_me)
