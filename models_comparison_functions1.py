# Basic data management
import pandas as pd
import numpy as np
# Data management and manipulation
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Hiperparameters optimization
from sklearn.model_selection import RandomizedSearchCV
# Processing time and emissions trackers
from codecarbon import EmissionsTracker
from time import time
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import lightgbm as lgb
import torch
import transformers
from transformers import pipeline
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# UTIL FUNCTIONS:

# Define a function to categorize polarity scores
def categorize_polarity(polarity):
    if polarity >= 0.05:
        return 'POSITIVE'
    elif polarity <= -0.05:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'
    

###################################################################

### CUSTOM METRIC

# Define custom metric function with dynamic weights
def custom_metric(
        results_df,
        good_weights=None,
        bad_weights=None
    ):
    """
    Function to calculate a custom metric for model evaluation by scaling time and emissions columns
    and combining them with weighted accuracy, precision, recall, and F1-Score.
    Parameters:
    - results_df: DataFrame containing results and metrics for each model.
    - good_weights: Dictionary with weights for 'Accuracy', 'Recall', 'Precision', and 'F1-Score' (default: None)
    - bad_weights: Dictionary with weights for 'Adjusted Prediction Time (s)' and 'Adjusted Prediction Emissions (kg CO2e)' (default: None)

    Returns:
    - results_df: DataFrame with a new 'Custom Metric' column.
    """

    # Set default weights if not provided
    if good_weights is None:
        good_weights = {
            'Accuracy': 0.65,
            'Recall': 0.15,
            'Precision': 0.15,
            'F1-Score': 0.15
        }

    if bad_weights is None:
        bad_weights = {
            'Adjusted Prediction Time (s)': 0.50,
            'Adjusted Prediction Emissions (kg CO2e)': 0.50
        }

    # Columns to be normalized
    columns_to_normalize = [
        'Training Time (s)', 'Adjusted Prediction Time (s)',
        'Training Emissions (kg CO2e)', 'Adjusted Prediction Emissions (kg CO2e)'
    ]
    normalized_columns = [
        'n_Training Time (s)', 'n_Adjusted Prediction Time (s)',
        'n_Training Emissions (kg CO2e)', 'n_Adjusted Prediction Emissions (kg CO2e)'
    ]

    # Apply the scaler only to the specified columns
    scaler = MinMaxScaler()
    results_df[normalized_columns] = scaler.fit_transform(results_df[columns_to_normalize])

    # Calculate the 'good' part of the custom metric (accuracy, recall, precision, f1-score)
    results_df['Good Metric'] = (
        results_df['Accuracy'] * good_weights['Accuracy'] +
        results_df['Recall'] * good_weights['Recall'] +
        results_df['Precision'] * good_weights['Precision'] +
        results_df['F1-Score'] * good_weights['F1-Score']
    )

    # Calculate the 'bad' part of the custom metric (time and emissions)
    results_df['Bad Metric'] = (
        results_df['n_Adjusted Prediction Time (s)'] * bad_weights['Adjusted Prediction Time (s)'] +
        results_df['n_Adjusted Prediction Emissions (kg CO2e)'] * bad_weights['Adjusted Prediction Emissions (kg CO2e)']
    )

    # Calculate the final custom metric (higher is better)
    results_df['Custom Metric'] = results_df['Good Metric'] / results_df['Bad Metric']

    return results_df




################################################################################################


## FUNCTIONS FOR DIFERENT MODELS ANALISYS

#### VADERSENTIMENT



# Define the generalized function
def vader_sentiment_analysis(df, text_column, target_column, results_df, dataset_name='dataset'):
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Columns in DataFrame: {df.columns.tolist()}")
    
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    # For tracking time and carbon emissions
    tracker = EmissionsTracker()

    ######################################## PREDICT ##############################
    # Start emissions and time tracking
    tracker.start()
    start_time = time()

    # Apply VADER on the text in the specified column
    df['vader_sentiment'] = df[text_column].apply(lambda text: categorize_polarity(analyzer.polarity_scores(text)['compound']))

    # Stop tracking
    end_time = time()
    emissions = tracker.stop()

    # Processing time for VADER
    vader_processing_time = end_time - start_time

    print(f"VADER sentiment distribution: {df['vader_sentiment'].value_counts()}")
    print(f"Target column distribution: {df[target_column].value_counts()}")

    ####################################### METRICS ##############################
    # Calculate metrics using scikit-learn
    accuracy_vader = accuracy_score(df[target_column], df['vader_sentiment'])
    precision_vader = precision_score(df[target_column], df['vader_sentiment'], average='weighted')
    recall_vader = recall_score(df[target_column], df['vader_sentiment'], average='weighted')
    f1_vader = f1_score(df[target_column], df['vader_sentiment'], average='weighted')

    print(f"Accuracy: {accuracy_vader}")
    print(f"Precision: {precision_vader}")
    print(f"Recall: {recall_vader}")
    print(f"F1-Score: {f1_vader}")

    # Add VADER results to the results DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'VADER',
        'Dataset': dataset_name,
        'Training Time (s)': 0,  # VADER has no training time
        'Original Prediction Time (s)': vader_processing_time,
        'Adjusted Prediction Time (s)': vader_processing_time,  # Use the same value because it operates at 100%
        'Training Emissions (kg CO2e)': 0,  # VADER has no training emissions
        'Original Prediction Emissions (kg CO2e)': emissions,
        'Adjusted Prediction Emissions (kg CO2e)': emissions,  # Use the same value because it operates at 100%
        'Accuracy': accuracy_vader,
        'Precision': precision_vader,
        'Recall': recall_vader,
        'F1-Score': f1_vader
    }])], ignore_index=True)

    return results_df






### SUPORT VECTOR MACHINE


# Define the SVM sentiment function
def SVM_sentiment(X_train, X_test, y_train, y_test, results_df=None, dataset_name='dataset'):
    
    #Function to train an SVM model, track time, emissions, and evaluate performance metrics.
    

    # Instance of SVM model
    svm_model = SVC(kernel='linear', probability=True)

    # Instance of EmissionsTracker to track carbon emissions
    tracker = EmissionsTracker()

    ####################################### TRAINING ##############################
    # Start emissions and time tracking for training
    tracker.start()
    start_time = time()

    # SVM training
    svm_model.fit(X_train, y_train)

    # Stop tracking time and emissions for training
    end_time = time()
    training_time = end_time - start_time
    training_emissions = tracker.stop()

    ######################################## PREDICTION ###########################
    # Start emissions and time tracking for prediction
    tracker.start()
    start_time = time()

    # Predict on the test data
    y_pred = svm_model.predict(X_test)

    # Stop tracking time and emissions for prediction
    end_time = time()
    predict_emissions = tracker.stop()

    # Calculate the time spent in prediction
    svm_prediction_time = end_time - start_time

    ######################################## METRICS ##############################
    # Adjust prediction time and emissions to reflect their value if the model had processed the entire dataset
    adjusted_prediction_time = svm_prediction_time * (100 / 20)  # 20% of the dataset was used for test
    adjusted_prediction_emissions = predict_emissions * (100 / 20)

    # Calculate performance metrics using scikit-learn
    accuracy_svm = accuracy_score(y_test, y_pred)
    precision_svm = precision_score(y_test, y_pred, average='weighted')
    recall_svm = recall_score(y_test, y_pred, average='weighted')
    f1_svm = f1_score(y_test, y_pred, average='weighted')

    ####################################### RESULTS ###############################

    # Adding SVM results to the results dataframe
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'SVM',
        'Dataset': dataset_name,  # Change to the appropriate dataset name
        'Training Time (s)': training_time,
        'Original Prediction Time (s)': svm_prediction_time,
        'Adjusted Prediction Time (s)': adjusted_prediction_time,
        'Training Emissions (kg CO2e)': training_emissions,
        'Original Prediction Emissions (kg CO2e)': predict_emissions,
        'Adjusted Prediction Emissions (kg CO2e)': adjusted_prediction_emissions,
        'Accuracy': accuracy_svm,
        'Precision': precision_svm,
        'Recall': recall_svm,
        'F1-Score': f1_svm
    }])], ignore_index=True)

    # Return the results dataframe
    return results_df



### NAIVE BAYES CLASSIFIER


def naive_sentiment(X_train, X_test, y_train, y_test, results_df=None, dataset_name='dataset'):
    # Inicialize NB model
    nb_model = MultinomialNB()

    # Inicialize emission tracker
    tracker = EmissionsTracker()

    ####################################### TRAINING ##############################

    # Start tracking emissions and time for training
    tracker.start()
    start_time = time()

    # Naive Bayes training
    nb_model.fit(X_train, y_train)

    # Stop tracking and get results
    end_time = time()
    training_emissions = tracker.stop()
    training_time = end_time - start_time

    ######################################## PREDICT ##############################

    # Start tracking emissions and time for prediction (20% of total data)
    tracker.start()
    start_time = time()

    # To predict on the test dataset
    y_pred = nb_model.predict(X_test)

    # Stop tracking and get results
    end_time = time()
    prediction_emissions = tracker.stop()
    prediction_time = end_time - start_time

    ######################################## METRICS ##############################

    # To get metrics using scikit-learn
    accuracy_nb = accuracy_score(y_test, y_pred)
    precision_nb = precision_score(y_test, y_pred, average='weighted')
    recall_nb = recall_score(y_test, y_pred, average='weighted')
    f1_nb = f1_score(y_test, y_pred, average='weighted')

    # To adjust the prediction time and emissions to reflect their value if the model had processed the whole dataset
    adjusted_prediction_time = prediction_time * (100 / 20)
    adjusted_prediction_emissions = prediction_emissions * (100 / 20)

    # Adding MultinomialNB results to dataframe
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'Naive Bayes',
        'Dataset': dataset_name,
        'Training Time (s)': training_time,
        'Original Prediction Time (s)': prediction_time,
        'Adjusted Prediction Time (s)': adjusted_prediction_time,
        'Training Emissions (kg CO2e)': training_emissions,
        'Original Prediction Emissions (kg CO2e)': prediction_emissions,
        'Adjusted Prediction Emissions (kg CO2e)': adjusted_prediction_emissions,
        'Accuracy': accuracy_nb,
        'Precision': precision_nb,
        'Recall': recall_nb,
        'F1-Score': f1_nb
    }])], ignore_index=True)

    return results_df


# Define the xgboost_sentiment function
# Definir la función xgboost_sentiment con el uso del LabelEncoder como parámetro
def xgboost_sentiment(X_train, X_test, y_train, y_test, results_df=None, dataset_name='dataset'):
    """
    Function to train an XGBoost model, track emissions, and evaluate its performance.
    
    Parameters:
    - X_train: Training features (array-like)
    - X_test: Test features (array-like)
    - y_train: Training labels (array-like)
    - y_test: Test labels (array-like)
    - label_encoder: Predefined LabelEncoder instance for label transformation
    - results_df: DataFrame to store results (optional)
    - dataset_name: Name of the dataset used (string, optional)

    Returns:
    - results_df: DataFrame containing the results of the training and evaluation
    """

    # Encode labels using the provided LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Crear una instancia del modelo XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Instanciar el tracker de emisiones
    tracker = EmissionsTracker()

    ####################################### ENTRENAMIENTO ##############################
    # Iniciar el seguimiento de tiempo y emisiones para el entrenamiento
    tracker.start()
    start_time = time()

    # Entrenamiento del modelo
    xgb_model.fit(X_train, y_train_encoded)

    # Detener el seguimiento de tiempo y emisiones después del entrenamiento
    training_time = time() - start_time
    training_emissions = tracker.stop()

    ######################################## PREDICCIÓN ###########################
    # Iniciar el seguimiento de tiempo y emisiones para la predicción
    tracker.start()
    start_time = time()

    # Realizar predicciones con el modelo entrenado
    y_pred = xgb_model.predict(X_test)

    # Detener el seguimiento de tiempo y emisiones después de la predicción
    prediction_time = time() - start_time
    prediction_emissions = tracker.stop()

    ######################################## MÉTRICAS ##############################
    # Calcular métricas de rendimiento usando scikit-learn
    accuracy_xgb = accuracy_score(y_test_encoded, y_pred)
    precision_xgb = precision_score(y_test_encoded, y_pred, average='weighted')
    recall_xgb = recall_score(y_test_encoded, y_pred, average='weighted')
    f1_xgb = f1_score(y_test_encoded, y_pred, average='weighted')

    # Ajustar el tiempo y las emisiones de predicción para reflejar su valor si se hubiera procesado todo el dataset
    adjusted_prediction_time = prediction_time * (100 / 20)  # Asumiendo que el 20% de los datos se usó
    adjusted_prediction_emissions = prediction_emissions * (100 / 20)

    ####################################### RESULTADOS ###############################
    # Preparar el DataFrame de resultados si no se proporcionó uno
    if results_df is None:
        results_df = pd.DataFrame(columns=[
            'Model', 'Dataset', 'Training Time (s)', 'Original Prediction Time (s)',
            'Adjusted Prediction Time (s)', 'Training Emissions (kg CO2e)',
            'Original Prediction Emissions (kg CO2e)', 'Adjusted Prediction Emissions (kg CO2e)',
            'Accuracy', 'Precision', 'Recall', 'F1-Score'
        ])

    # Agregar los resultados de XGBoost al DataFrame de resultados
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'XGBoost',
        'Dataset': dataset_name,
        'Training Time (s)': training_time,
        'Original Prediction Time (s)': prediction_time,
        'Adjusted Prediction Time (s)': adjusted_prediction_time,
        'Training Emissions (kg CO2e)': training_emissions,
        'Original Prediction Emissions (kg CO2e)': prediction_emissions,
        'Adjusted Prediction Emissions (kg CO2e)': adjusted_prediction_emissions,
        'Accuracy': accuracy_xgb,
        'Precision': precision_xgb,
        'Recall': recall_xgb,
        'F1-Score': f1_xgb
    }])], ignore_index=True)

    # Devolver el DataFrame de resultados con todas las métricas
    return results_df



### LOGISTIC REGRESSION


# Define the logistic regression sentiment function
def logistic_regression_sentiment(X_train, X_test, y_train, y_test, results_df=None, dataset_name='dataset'):
    
    '''Function to train a Logistic Regression model, track emissions, and evaluate its performance.
    Parameters:
    - X_train: Training features
    - X_test: Test features
    - y_train: Training labels
    - y_test: Test labels
    - results_df: DataFrame to store results 
    - dataset_name: Name of the dataset used

    Returns:
    - results_df: DataFrame containing the results of the training and evaluation
    '''
    

    # Create an instance of the Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)

    # Create an instance of EmissionsTracker to track carbon emissions
    tracker = EmissionsTracker()

    ####################################### TRAINING ##############################
    # Measure time and emissions during training
    tracker.start()
    start_time = time()

    # Train the Logistic Regression model
    lr_model.fit(X_train, y_train)

    # Stop tracking time and emissions after training
    training_time = time() - start_time
    training_emissions = tracker.stop()

    ######################################## PREDICT ##############################
    # Measure time and emissions during prediction
    tracker.start()
    start_time = time()

    # Make predictions on the test set
    y_pred = lr_model.predict(X_test)

    # Stop tracking after prediction
    prediction_time = time() - start_time
    prediction_emissions = tracker.stop()

    ######################################## METRICS ##############################
    # Calculate performance metrics using scikit-learn
    accuracy_lr = accuracy_score(y_test, y_pred)
    precision_lr = precision_score(y_test, y_pred, average='weighted')
    recall_lr = recall_score(y_test, y_pred, average='weighted')
    f1_lr = f1_score(y_test, y_pred, average='weighted')

    ####################################### RESULTS ###############################
    # Adjust prediction time and emissions to reflect their value for 100% of the dataset (if only 20% used)
    adjusted_prediction_time = prediction_time * (100 / 20)
    adjusted_prediction_emissions = prediction_emissions * (100 / 20)

    # Prepare the results dataframe if not provided
    if results_df is None:
        results_df = pd.DataFrame(columns=[
            'Model', 'Dataset', 'Training Time (s)', 'Original Prediction Time (s)',
            'Adjusted Prediction Time (s)', 'Training Emissions (kg CO2e)',
            'Original Prediction Emissions (kg CO2e)', 'Adjusted Prediction Emissions (kg CO2e)',
            'Accuracy', 'Precision', 'Recall', 'F1-Score'
        ])

    # Add Logistic Regression results to the results dataframe
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'Logistic Regression',
        'Dataset': dataset_name,
        'Training Time (s)': training_time,
        'Original Prediction Time (s)': prediction_time,
        'Adjusted Prediction Time (s)': adjusted_prediction_time,
        'Training Emissions (kg CO2e)': training_emissions,
        'Original Prediction Emissions (kg CO2e)': prediction_emissions,
        'Adjusted Prediction Emissions (kg CO2e)': adjusted_prediction_emissions,
        'Accuracy': accuracy_lr,
        'Precision': precision_lr,
        'Recall': recall_lr,
        'F1-Score': f1_lr
    }])], ignore_index=True)

    # Return the results dataframe with all metrics
    return results_df



### RANDOM FOREST

# Define the Random Forest sentiment analysis function
def rforest_sentiment(X_train, X_test, y_train, y_test, results_df=None, dataset_name='dataset', n_estimators=100):
    """
    Function to train a Random Forest model, track emissions, and evaluate its performance.
    Parameters:
    - X_train: Training features
    - X_test: Test features
    - y_train: Training labels
    - y_test: Test labels
    - results_df: DataFrame to store results (optional)
    - dataset_name: Name of the dataset used (string, optional)
    - n_estimators: Number of trees in the Random Forest (default: 100)

    Returns:
    - results_df: DataFrame containing the results of the training and evaluation
    """

    # Create an instance of the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Create an instance of EmissionsTracker to track carbon emissions
    tracker = EmissionsTracker()

    ####################################### TRAINING ##############################
    # Start emissions and time tracking for training
    tracker.start()
    start_time = time()

    # Train the Random Forest model
    rf_model.fit(X_train, y_train)

    # Stop tracking for training
    training_time = time() - start_time
    training_emissions = tracker.stop()

    ######################################## PREDICTION ###########################
    # Start emissions and time tracking for prediction
    tracker.start()
    start_time = time()

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Stop tracking for prediction
    prediction_time = time() - start_time
    prediction_emissions = tracker.stop()

    ######################################## METRICS ##############################
    # Calculate performance metrics using scikit-learn
    accuracy_rf = accuracy_score(y_test, y_pred)
    precision_rf = precision_score(y_test, y_pred, average='weighted')
    recall_rf = recall_score(y_test, y_pred, average='weighted')
    f1_rf = f1_score(y_test, y_pred, average='weighted')

    # Adjust prediction time and emissions to reflect their value if the model had processed the entire dataset
    adjusted_prediction_time = prediction_time * (100 / 20)  # Assuming 20% of the dataset was used
    adjusted_prediction_emissions = prediction_emissions * (100 / 20)

    ####################################### RESULTS ###############################
    # Prepare the results dataframe if not provided
    if results_df is None:
        results_df = pd.DataFrame(columns=[
            'Model', 'Dataset', 'Training Time (s)', 'Original Prediction Time (s)',
            'Adjusted Prediction Time (s)', 'Training Emissions (kg CO2e)',
            'Original Prediction Emissions (kg CO2e)', 'Adjusted Prediction Emissions (kg CO2e)',
            'Accuracy', 'Precision', 'Recall', 'F1-Score'
        ])

    # Add Random Forest results to the results dataframe
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'Random Forest',
        'Dataset': dataset_name,
        'Training Time (s)': training_time,
        'Original Prediction Time (s)': prediction_time,
        'Adjusted Prediction Time (s)': adjusted_prediction_time,
        'Training Emissions (kg CO2e)': training_emissions,
        'Original Prediction Emissions (kg CO2e)': prediction_emissions,
        'Adjusted Prediction Emissions (kg CO2e)': adjusted_prediction_emissions,
        'Accuracy': accuracy_rf,
        'Precision': precision_rf,
        'Recall': recall_rf,
        'F1-Score': f1_rf
    }])], ignore_index=True)

    # Return the results dataframe with all metrics
    return results_df


### LIGHTGBM


# Define the LightGBM sentiment analysis function
# Ajuste en la Definición de la Función lightgbm_sentiment para Incluir label_encoder
def lightgbm_sentiment(X_train, X_test, y_train, y_test, results_df=None, dataset_name='dataset'):
    """
    Function to train a LightGBM model, track emissions, and evaluate its performance.
    Parameters:
    - X_train: Training features (DataFrame or array-like)
    - X_test: Test features (DataFrame or array-like)
    - y_train: Training labels (array-like)
    - y_test: Test labels (array-like)
    - label_encoder: Predefined LabelEncoder instance for label transformation
    - results_df: DataFrame to store results (optional)
    - dataset_name: Name of the dataset used (string, optional)

    Returns:
    - results_df: DataFrame containing the results of the training and evaluation
    """
    # Encode labels to numeric using the provided LabelEncoder
    # Ajustar el `LabelEncoder` para cada dataset
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train_encoded)
    test_data = lgb.Dataset(X_test, label=y_test_encoded, reference=train_data)

    # Set parameters for LightGBM
    params = {
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),  # Number of classes
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 100,
        'learning_rate': 0.03,
        'feature_fraction': 0.9
    }

    # Create an instance of EmissionsTracker to track carbon emissions
    tracker = EmissionsTracker()

    ####################################### TRAINING ##############################
    # Start emissions and time tracking for training
    tracker.start()
    start_time = time()

    # Train the LightGBM model
    gbm_model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])

    # Stop tracking after training
    training_time = time() - start_time
    training_emissions = tracker.stop()

    ######################################## PREDICTION ###########################
    # Start emissions and time tracking for prediction
    tracker.start()
    start_time = time()

    # Make predictions on the test set
    y_pred = gbm_model.predict(X_test, num_iteration=gbm_model.best_iteration)

    # Stop tracking after prediction
    prediction_time = time() - start_time
    prediction_emissions = tracker.stop()

    ######################################## METRICS ##############################
    # Convert predictions to class labels using the passed label_encoder
    y_pred_labels = [label_encoder.inverse_transform([np.argmax(line)])[0] for line in y_pred]

    # Calculate performance metrics using scikit-learn
    accuracy_gbm = accuracy_score(y_test, y_pred_labels)
    precision_gbm = precision_score(y_test, y_pred_labels, average='weighted')
    recall_gbm = recall_score(y_test, y_pred_labels, average='weighted')
    f1_gbm = f1_score(y_test, y_pred_labels, average='weighted')

    # Adjust prediction time and emissions to reflect their value if the model had processed the entire dataset
    adjusted_prediction_time = prediction_time * (100 / 20)  # Assuming 20% of the dataset was used
    adjusted_prediction_emissions = prediction_emissions * (100 / 20)

    ####################################### RESULTS ###############################
    # Prepare the results dataframe if not provided
    if results_df is None:
        results_df = pd.DataFrame(columns=[
            'Model', 'Dataset', 'Training Time (s)', 'Original Prediction Time (s)',
            'Adjusted Prediction Time (s)', 'Training Emissions (kg CO2e)',
            'Original Prediction Emissions (kg CO2e)', 'Adjusted Prediction Emissions (kg CO2e)',
            'Accuracy', 'Precision', 'Recall', 'F1-Score'
        ])

    # Add LightGBM results to the results dataframe
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'LightGBM',
        'Dataset': dataset_name,
        'Training Time (s)': training_time,
        'Original Prediction Time (s)': prediction_time,
        'Adjusted Prediction Time (s)': adjusted_prediction_time,
        'Training Emissions (kg CO2e)': training_emissions,
        'Original Prediction Emissions (kg CO2e)': prediction_emissions,
        'Adjusted Prediction Emissions (kg CO2e)': adjusted_prediction_emissions,
        'Accuracy': accuracy_gbm,
        'Precision': precision_gbm,
        'Recall': recall_gbm,
        'F1-Score': f1_gbm
    }])], ignore_index=True)

    # Return the results dataframe with all metrics
    return results_df




### BERT SENTIMENT ANALYSIS PIPELINE.


#- Just for comparison because it is the model we used to autolabel the real dataset


# to empty GPU cache if needed
# torch.cuda.empty_cache()



# Define the RoBERTa sentiment analysis function with configurable dataset and column names
def roberta_sentiment(
        dataframe,
        text_column='user_clean_text',
        label_column='user_sentiment',
        results_df=None,
        dataset_name='dataset'
    ):
    """
    Function to perform sentiment analysis using RoBERTa, track emissions, and evaluate its performance.
    Parameters:
    - dataframe: The dataset to be used for analysis (DataFrame)
    - text_column: The column containing the text data (string)
    - label_column: The column containing the true labels for evaluation (string)
    - results_df: DataFrame to store results
    - dataset_name: Name of the dataset being processed

    Returns:
    - results_df: DataFrame containing the results of the evaluation
    """

    # Create sentiment analysis pipeline using RoBERTa model
    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=0,  # Use GPU if available
        truncation=True,
        max_length=512,
        batch_size=12800
    )

    # Create an instance of EmissionsTracker
    tracker = EmissionsTracker()

    ######################################## PREDICT ##############################
    # Start emissions and time tracking for prediction
    tracker.start()
    start_time = time()

    # Apply the RoBERTa pipeline to the specified text column
    dataframe['roberta_sentiment'] = dataframe[text_column].apply(
        lambda x: sentiment_analysis(x)[0]['label']
    )

    # Stop tracking and calculate the prediction time
    prediction_time_roberta = time() - start_time
    emissions_roberta = tracker.stop()

    ######################################## METRICS ##############################
    # Map RoBERTa's original labels to match the desired labels
    dataframe['roberta_sentiment'] = dataframe['roberta_sentiment'].map(
        {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'}
    )

    # Define true and predicted labels using the specified label column
    y_true = dataframe[label_column]
    y_pred = dataframe['roberta_sentiment']

    # Calculate performance metrics using scikit-learn
    accuracy_roberta = accuracy_score(y_true, y_pred)
    precision_roberta = precision_score(y_true, y_pred, average='weighted')
    recall_roberta = recall_score(y_true, y_pred, average='weighted')
    f1_roberta = f1_score(y_true, y_pred, average='weighted')

    ####################################### RESULTS ###############################
    # Prepare the results dataframe
    if results_df is None:
        results_df = pd.DataFrame(columns=[
            'Model', 'Dataset', 'Training Time (s)', 'Original Prediction Time (s)',
            'Adjusted Prediction Time (s)', 'Training Emissions (kg CO2e)',
            'Original Prediction Emissions (kg CO2e)', 'Adjusted Prediction Emissions (kg CO2e)',
            'Accuracy', 'Precision', 'Recall', 'F1-Score'
        ])

    # Add RoBERTa results to the results dataframe
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'RoBERTa',
        'Dataset': dataset_name,
        'Training Time (s)': 0,  # RoBERTa is pre-trained, so training time is set to 0
        'Original Prediction Time (s)': prediction_time_roberta,
        'Adjusted Prediction Time (s)': prediction_time_roberta,  # No need to adjust if the full dataset is used
        'Training Emissions (kg CO2e)': 0,  # Emissions for training are set to 0 since it's pre-trained
        'Original Prediction Emissions (kg CO2e)': emissions_roberta,
        'Adjusted Prediction Emissions (kg CO2e)': emissions_roberta,
        'Accuracy': accuracy_roberta,
        'Precision': precision_roberta,
        'Recall': recall_roberta,
        'F1-Score': f1_roberta
    }])], ignore_index=True)

    # Return the results dataframe with all metrics
    return results_df



# SPACY SENTIMENT ANALISYS

# - Install needed modules as specified in the example shown in https://spacy.io/universe/project/spacy-textblob

# example of use

# import spacy
# from spacytextblob.spacytextblob import SpacyTextBlob

# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe('spacytextblob')
# text = 'I had a really horrible day. It was the worst day ever! But every now and then I have a really good day that makes me happy.'
# doc = nlp(text)
# doc._.blob.polarity                            # Polarity: -0.125
# doc._.blob.subjectivity                        # Subjectivity: 0.9
# doc._.blob.sentiment_assessments.assessments   # Assessments: [(['really', 'horrible'], -1.0, 1.0, None), (['worst', '!'], -1.0, 1.0, None), (['really', 'good'], 0.7, 0.6000000000000001, None), (['happy'], 0.8, 1.0, None)]
# doc._.blob.ngrams()



'''# Define the Spacy sentiment analysis function
def spacy_sentiment(dataframe, text_column='user_clean_text', label_column='user_sentiment', results_df=None, dataset_name='dataset'):
    """
    Function to perform sentiment analysis using Spacy with different models, track emissions, and evaluate performance.
    Parameters:
    - dataframe: The dataset to be used for analysis (DataFrame)
    - text_column: The column containing the text data (string)
    - label_column: The column containing the true labels for evaluation (string)
    - results_df: DataFrame to store results (optional)
    - dataset_name: Name of the dataset being processed (string, optional)

    Returns:
    - results_df: DataFrame containing the results of the evaluation
    """

    # Load Spacy models 
    nlp_sm = spacy.load('en_core_web_sm')
    nlp_lg = spacy.load('en_core_web_lg')
    nlp_trf = spacy.load('en_core_web_trf')

    # Add SpacyTextBlob to the pipelines
    nlp_sm.add_pipe('spacytextblob')
    nlp_lg.add_pipe('spacytextblob')
    nlp_trf.add_pipe('spacytextblob')

    # Define models dic
    models = {'small': nlp_sm, 'large': nlp_lg, 'transformer': nlp_trf}

    # Prepare the results dataframe if not provided
    if results_df is None:
        results_df = pd.DataFrame(columns=[
            'Model', 'Dataset', 'Training Time (s)', 'Original Prediction Time (s)',
            'Adjusted Prediction Time (s)', 'Training Emissions (kg CO2e)',
            'Original Prediction Emissions (kg CO2e)', 'Adjusted Prediction Emissions (kg CO2e)',
            'Accuracy', 'Precision', 'Recall', 'F1-Score'
        ])

    # Iterate over the three models (small, large, transformer)
    for model_name, model in models.items():
        # Start tracking emissions and processing time for each model
        tracker = EmissionsTracker()
        tracker.start()
        start_time = time()

        # Create the predicted sentiment column for each model dynamically
        sentiment_column = f'{model_name}_sentiment'
        dataframe[sentiment_column] = None

        # Apply the Spacy pipeline to each text entry in the specified column
        for index, row in dataframe.iterrows():
            doc = model(row[text_column])
            polarity = categorize_polarity(doc._.blob.polarity)
            dataframe.at[index, sentiment_column] = polarity  # Update the specific row with the predicted sentiment

        # Stop emissions tracking
        emissions = tracker.stop()
        processing_time = time() - start_time

        # Define true and predicted labels for each model
        y_true = dataframe[label_column]
        y_pred = dataframe[sentiment_column]

        # Calculate performance metrics using scikit-learn
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Add metrics to the results DataFrame for each model
        results_df = pd.concat([results_df, pd.DataFrame([{
            'Model': f'Spacy_{model_name}',
            'Dataset': dataset_name,
            'Training Time (s)': 0,  # No training time for Spacy pipelines in this context
            'Original Prediction Time (s)': processing_time,
            'Adjusted Prediction Time (s)': processing_time,  # No need to adjust if the full dataset is used
            'Training Emissions (kg CO2e)': 0,  # Emissions for training are set to 0 as the models are pre-built
            'Original Prediction Emissions (kg CO2e)': emissions,
            'Adjusted Prediction Emissions (kg CO2e)': emissions,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }])], ignore_index=True)

    # Return the results dataframe with all metrics
    return results_df'''


    # Define the Spacy sentiment analysis function with transformer core



def spacy_sentiment(dataframe, text_column='user_clean_text', label_column='user_sentiment', results_df=None, dataset_name='dataset'):
    """
    Perform sentiment analysis using Spacy with different models, track emissions, and evaluate performance.
    Parameters:
    - dataframe: The dataset to be used for analysis (DataFrame)
    - text_column: The column containing the text data (string)
    - label_column: The column containing the true labels for evaluation (string)
    - results_df: DataFrame to store results (optional)
    - dataset_name: Name of the dataset being processed (string, optional)

    Returns:
    - results_df: DataFrame containing the results of the evaluation
    """

    # 1. Cargar los modelos básicos (small y large) en CPU
    nlp_sm = spacy.load('en_core_web_sm')
    nlp_lg = spacy.load('en_core_web_lg')

    # 2. Añadir SpacyTextBlob a los pipelines de los modelos de CPU
    nlp_sm.add_pipe('spacytextblob')
    nlp_lg.add_pipe('spacytextblob')

    # 3. Cargar el modelo transformer en GPU solo si se requiere
    nlp_trf = None

    # Definir el diccionario de modelos con el modelo transformer inicializado como None
    models = {'small': nlp_sm, 'large': nlp_lg, 'transformer': nlp_trf}

    # 4. Preparar el DataFrame de resultados si no se proporciona uno
    if results_df is None:
        results_df = pd.DataFrame(columns=[
            'Model', 'Dataset', 'Training Time (s)', 'Original Prediction Time (s)',
            'Adjusted Prediction Time (s)', 'Training Emissions (kg CO2e)',
            'Original Prediction Emissions (kg CO2e)', 'Adjusted Prediction Emissions (kg CO2e)',
            'Accuracy', 'Precision', 'Recall', 'F1-Score'
        ])

    # 5. Iterar sobre los modelos y asegurarse de que la GPU solo se use cuando se necesite
    for model_name, model in models.items():
        if model_name == 'transformer':
            # Activar GPU para el modelo transformer
            spacy.require_gpu()
            nlp_trf = spacy.load('en_core_web_trf')  # Cargar el modelo transformer en la GPU
            nlp_trf.add_pipe('spacytextblob')
            model = nlp_trf
        else:
            # Para `small` y `large`, no necesitamos hacer nada especial (se cargan en CPU por defecto)
            pass

        # 6. Empezar el rastreo de emisiones y medir tiempo de procesamiento
        tracker = EmissionsTracker()
        tracker.start()
        start_time = time()

        # Crear la columna de sentimiento para cada modelo
        sentiment_column = f'{model_name}_sentiment'
        dataframe[sentiment_column] = None

        # Aplicar el pipeline de Spacy a cada entrada de texto en la columna especificada
        for index, row in dataframe.iterrows():
            doc = model(row[text_column])
            polarity = categorize_polarity(doc._.blob.polarity)
            dataframe.at[index, sentiment_column] = polarity

        # 7. Detener el rastreo de emisiones y calcular tiempo de procesamiento
        emissions = tracker.stop()
        processing_time = time() - start_time

        # Definir etiquetas verdaderas y predichas para cada modelo
        y_true = dataframe[label_column]
        y_pred = dataframe[sentiment_column]

        # Calcular métricas de rendimiento usando scikit-learn
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Agregar las métricas al DataFrame de resultados para cada modelo
        results_df = pd.concat([results_df, pd.DataFrame([{
            'Model': f'Spacy_{model_name}',
            'Dataset': dataset_name,
            'Training Time (s)': 0,
            'Original Prediction Time (s)': processing_time,
            'Adjusted Prediction Time (s)': processing_time,
            'Training Emissions (kg CO2e)': 0,
            'Original Prediction Emissions (kg CO2e)': emissions,
            'Adjusted Prediction Emissions (kg CO2e)': emissions,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }])], ignore_index=True)

    # 8. Devolver el DataFrame con los resultados de todas las métricas
    return results_df





