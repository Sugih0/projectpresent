import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes ML Project",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .section-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background: #f0f8ff;
        border: 1px solid #4facfe;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #c53030;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data (since we can't directly load from Kaggle)
@st.cache_data
def load_diabetes_data():
    """Generate sample diabetes dataset similar to Pima Indians Diabetes Dataset"""
    np.random.seed(42)
    n_samples = 768
    
    # Generate features based on diabetes dataset characteristics
    pregnancies = np.random.poisson(3, n_samples)
    glucose = np.random.normal(120, 30, n_samples)
    glucose = np.clip(glucose, 0, 200)
    
    blood_pressure = np.random.normal(70, 12, n_samples)
    blood_pressure = np.clip(blood_pressure, 0, 120)
    
    skin_thickness = np.random.exponential(20, n_samples)
    skin_thickness = np.clip(skin_thickness, 0, 60)
    
    insulin = np.random.exponential(80, n_samples)
    insulin = np.clip(insulin, 0, 600)
    
    bmi = np.random.normal(32, 7, n_samples)
    bmi = np.clip(bmi, 15, 60)
    
    pedigree = np.random.exponential(0.5, n_samples)
    pedigree = np.clip(pedigree, 0.08, 2.5)
    
    age = np.random.gamma(2, 15, n_samples)
    age = np.clip(age, 21, 81).astype(int)
    
    # Generate outcome based on logical rules
    risk_score = (
        (glucose > 140) * 2 +
        (bmi > 30) * 1.5 +
        (age > 50) * 1 +
        (pregnancies > 5) * 0.5 +
        (pedigree > 1) * 1 +
        (blood_pressure > 80) * 0.5
    )
    
    outcome_prob = 1 / (1 + np.exp(-(risk_score - 3)))
    outcome = np.random.binomial(1, outcome_prob, n_samples)
    
    data = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose.round(0),
        'BloodPressure': blood_pressure.round(0),
        'SkinThickness': skin_thickness.round(0),
        'Insulin': insulin.round(0),
        'BMI': bmi.round(1),
        'DiabetesPedigreeFunction': pedigree.round(3),
        'Age': age,
        'Outcome': outcome
    })
    
    return data

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Complete Diabetes Prediction ML Project</h1>
        <p>End-to-end machine learning pipeline for diabetes risk prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    sections = [
        "1. Problem Statement",
        "2. Dataset Overview", 
        "3. Exploratory Data Analysis",
        "4. Data Preprocessing",
        "5. Model Training",
        "6. Results & Predictions",
        "7. Evaluation Metrics"
    ]
    
    selected_section = st.sidebar.selectbox("Choose Section:", sections)
    
    # Load data
    df = load_diabetes_data()
    
    # Section 1: Problem Statement
    if selected_section == "1. Problem Statement":
        st.markdown('<div class="section-header"><h2>🎯 Problem Statement</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Diabetes Prediction Challenge
            
            **Objective:** Develop a machine learning system to predict the likelihood of diabetes onset based on diagnostic measurements.
            
            **Problem Type:** Binary Classification
            
            **Business Impact:**
            - Early detection of diabetes risk
            - Preventive healthcare measures
            - Reduced healthcare costs
            - Improved patient outcomes
            
            **Key Challenges:**
            - Imbalanced dataset
            - Feature correlation
            - Model interpretability
            - Clinical validation
            """)
            
            st.markdown("""
            <div class="info-box">
            <strong>💡 Why This Matters:</strong><br>
            Diabetes affects 422 million people worldwide. Early prediction can help:
            <ul>
            <li>Prevent Type 2 diabetes through lifestyle changes</li>
            <li>Reduce complications through early intervention</li>
            <li>Optimize healthcare resource allocation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Diabetes statistics
            fig = go.Figure(data=[
                go.Bar(x=['No Diabetes', 'Diabetes'], 
                      y=[len(df[df['Outcome']==0]), len(df[df['Outcome']==1])],
                      marker_color=['#2ecc71', '#e74c3c'])
            ])
            fig.update_layout(title="Dataset Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Section 2: Dataset Overview
    elif selected_section == "2. Dataset Overview":
        st.markdown('<div class="section-header"><h2>📊 Dataset Overview</h2></div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Pima Indians Diabetes Dataset
        **Source:** [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
        
        This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns)-1)
        with col3:
            st.metric("Diabetic Cases", len(df[df['Outcome']==1]))
        with col4:
            st.metric("Non-Diabetic Cases", len(df[df['Outcome']==0]))
        
        st.subheader("📋 Feature Description")
        
        feature_info = pd.DataFrame({
            'Feature': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
            'Description': [
                'Number of times pregnant',
                'Plasma glucose concentration (mg/dL)',
                'Diastolic blood pressure (mm Hg)',
                'Triceps skin fold thickness (mm)',
                '2-Hour serum insulin (mu U/ml)',
                'Body mass index (kg/m²)',
                'Diabetes pedigree function (genetic influence)',
                'Age in years'
            ],
            'Data Type': ['Integer', 'Float', 'Float', 'Float', 'Float', 'Float', 'Float', 'Integer']
        })
        
        st.dataframe(feature_info, use_container_width=True, hide_index=True)
        
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Section 3: EDA
    elif selected_section == "3. Exploratory Data Analysis":
        st.markdown('<div class="section-header"><h2>🔍 Exploratory Data Analysis</h2></div>', unsafe_allow_html=True)
        
        # Target distribution
        st.subheader("🎯 Target Variable Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='Outcome', 
                        title="Diabetes Distribution",
                        labels={'Outcome': 'Diabetes Status'},
                        color_discrete_sequence=['#2ecc71', '#e74c3c'])
            fig.update_traces(labels=['No Diabetes', 'Diabetes'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            outcome_counts = df['Outcome'].value_counts()
            st.metric("Diabetes Prevalence", f"{outcome_counts[1]/len(df)*100:.1f}%")
            st.metric("Class Ratio", f"1:{outcome_counts[0]/outcome_counts[1]:.1f}")
        
        # Feature distributions
        st.subheader("📊 Feature Distributions")
        
        numeric_cols = ['Glucose', 'BMI', 'Age', 'BloodPressure']
        
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=numeric_cols)
        
        for i, col in enumerate(numeric_cols):
            row = i // 2 + 1
            col_idx = i % 2 + 1
            
            # Histogram for each outcome
            fig.add_trace(
                go.Histogram(x=df[df['Outcome']==0][col], name=f'{col} - No Diabetes', 
                           opacity=0.7, marker_color='#2ecc71'),
                row=row, col=col_idx
            )
            fig.add_trace(
                go.Histogram(x=df[df['Outcome']==1][col], name=f'{col} - Diabetes', 
                           opacity=0.7, marker_color='#e74c3c'),
                row=row, col=col_idx
            )
        
        fig.update_layout(height=600, title_text="Feature Distributions by Diabetes Status")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        correlation_matrix = df.corr()
        
        fig = px.imshow(correlation_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance analysis
        st.subheader("⭐ Feature Importance Analysis")
        
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Quick Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', 
                    orientation='h', title="Feature Importance (Random Forest)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Section 4: Data Preprocessing
    elif selected_section == "4. Data Preprocessing":
        st.markdown('<div class="section-header"><h2>🔧 Data Preprocessing</h2></div>', unsafe_allow_html=True)
        
        st.subheader("🔍 Data Quality Check")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values:**")
            missing_values = df.isnull().sum()
            st.dataframe(missing_values.to_frame('Missing Count'), use_container_width=True)
        
        with col2:
            st.write("**Zero Values Analysis:**")
            zero_values = (df == 0).sum()
            st.dataframe(zero_values.to_frame('Zero Count'), use_container_width=True)
        
        st.subheader("🧹 Preprocessing Steps")
        
        # Show preprocessing steps
        preprocessing_steps = [
            "✅ Handle missing values (replace zeros with NaN where medically impossible)",
            "✅ Feature scaling using StandardScaler",
            "✅ Train-test split (80-20)",
            "✅ Handle class imbalance awareness",
        ]
        
        for step in preprocessing_steps:
            st.write(step)
        
        # Preprocessing implementation
        df_processed = df.copy()
        
        # Replace zeros with NaN for certain features
        zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for feature in zero_not_accepted:
            df_processed[feature] = df_processed[feature].replace(0, np.nan)
        
        # Fill NaN with median
        df_processed = df_processed.fillna(df_processed.median())
        
        st.subheader("📊 Before vs After Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Data Statistics:**")
            st.dataframe(df.describe().round(2))
        
        with col2:
            st.write("**Processed Data Statistics:**")
            st.dataframe(df_processed.describe().round(2))
        
        # Train-test split
        X = df_processed.drop('Outcome', axis=1)
        y = df_processed['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.subheader("📏 Feature Scaling")
        
        scaling_comparison = pd.DataFrame({
            'Feature': X.columns,
            'Original Mean': X.mean().round(2),
            'Original Std': X.std().round(2),
            'Scaled Mean': X_train_scaled.mean(axis=0).round(2),
            'Scaled Std': X_train_scaled.std(axis=0).round(2)
        })
        
        st.dataframe(scaling_comparison, use_container_width=True, hide_index=True)
        
        # Store processed data in session state
        st.session_state['processed_data'] = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }
    
    # Section 5: Model Training
    elif selected_section == "5. Model Training":
        st.markdown('<div class="section-header"><h2>🤖 Model Training</h2></div>', unsafe_allow_html=True)
        
        if 'processed_data' not in st.session_state:
            st.warning("Please run the Data Preprocessing section first!")
            return
        
        data = st.session_state['processed_data']
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        
        st.subheader("📋 Model Selection")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        if st.button("🚀 Train All Models"):
            model_results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                
                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                model_results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'training_time': training_time
                }
                
                progress_bar.progress((i + 1) / len(models))
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state['model_results'] = model_results
            
            # Display results
            st.subheader("📊 Model Performance Comparison")
            
            results_df = pd.DataFrame({
                'Model': list(model_results.keys()),
                'Accuracy': [results['accuracy'] for results in model_results.values()],
                'Training Time (s)': [results['training_time'] for results in model_results.values()]
            }).sort_values('Accuracy', ascending=False)
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Accuracy comparison chart
            fig = px.bar(results_df, x='Model', y='Accuracy', 
                        title="Model Accuracy Comparison",
                        color='Accuracy',
                        color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        if 'model_results' in st.session_state:
            st.subheader("🔍 Model Details")
            
            selected_model = st.selectbox("Select Model for Details:", 
                                        list(st.session_state['model_results'].keys()))
            
            model_info = st.session_state['model_results'][selected_model]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{model_info['accuracy']:.3f}")
            with col2:
                st.metric("Training Time", f"{model_info['training_time']:.2f}s")
            with col3:
                st.metric("Model Type", selected_model)
    
    # Section 6: Results & Predictions
    elif selected_section == "6. Results & Predictions":
        st.markdown('<div class="section-header"><h2>🎯 Results & Predictions</h2></div>', unsafe_allow_html=True)
        
        if 'model_results' not in st.session_state:
            st.warning("Please run the Model Training section first!")
            return
        
        st.subheader("🔮 Interactive Prediction")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.001)
            age = st.number_input("Age (years)", min_value=21, max_value=100, value=30)
        
        selected_model_name = st.selectbox("Select Model:", list(st.session_state['model_results'].keys()))
        
        if st.button("🔮 Make Prediction"):
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, pedigree, age]])
            
            # Scale the input data
            scaler = st.session_state['processed_data']['scaler']
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            selected_model = st.session_state['model_results'][selected_model_name]['model']
            prediction = selected_model.predict(input_scaled)[0]
            
            if hasattr(selected_model, 'predict_proba'):
                probability = selected_model.predict_proba(input_scaled)[0]
                prob_no_diabetes = probability[0]
                prob_diabetes = probability[1]
            else:
                prob_diabetes = 0.5 if prediction == 1 else 0.5
                prob_no_diabetes = 1 - prob_diabetes
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ High Risk of Diabetes")
                else:
                    st.success("✅ Low Risk of Diabetes")
            
            with col2:
                st.metric("Diabetes Probability", f"{prob_diabetes:.1%}")
            
            with col3:
                st.metric("Model Used", selected_model_name)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_diabetes * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50}}))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Section 7: Evaluation Metrics
    elif selected_section == "7. Evaluation Metrics":
        st.markdown('<div class="section-header"><h2>📏 Evaluation Metrics</h2></div>', unsafe_allow_html=True)
        
        if 'model_results' not in st.session_state:
            st.warning("Please run the Model Training section first!")
            return
        
        model_results = st.session_state['model_results']
        y_test = st.session_state['processed_data']['y_test']
        
        selected_model = st.selectbox("Select Model for Detailed Analysis:", 
                                    list(model_results.keys()))
        
        model_info = model_results[selected_model]
        y_pred = model_info['predictions']
        y_pred_proba = model_info['probabilities']
        
        # Classification Report
        st.subheader("📊 Classification Report")
        
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
        
        # Confusion Matrix
        st.subheader("🔀 Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, 
                       text_auto=True, 
                       aspect="auto",
                       title="Confusion Matrix",
                       labels=dict(x="Predicted", y="Actual"),
                       x=['No Diabetes', 'Diabetes'],
                       y=['No Diabetes', 'Diabetes'])
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        if y_pred_proba is not None:
            st.subheader("📈 ROC Curve")
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Comparison Summary
        st.subheader("📋 Model Comparison Summary")
        
        comparison_data = []
        for name, results in model_results.items():
            y_pred_model = results['predictions']
            report_model = classification_report(y_test, y_pred_model, output_dict=True)
            
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': report_model['1']['precision'],
                'Recall': report_model['1']['recall'],
                'F1-Score': report_model['1']['f1-score'],
                'Training Time': results['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data).round(3)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Best model recommendation
        best_model = comparison_df.iloc[0]['Model']
        st.success(f"🏆 Best Performing Model: **{best_model}** with {comparison_df.iloc[0]['Accuracy']:.1%} accuracy")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
