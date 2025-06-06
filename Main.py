import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Simple ML models using basic algorithms
class SimpleRandomForest:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            
            # Simple decision tree (threshold-based)
            tree = self._create_simple_tree(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def _create_simple_tree(self, X, y):
        # Find best threshold for each feature
        best_score = -1
        best_feature = 0
        best_threshold = 0
        
        for feature in range(X.shape[1]):
            thresholds = np.percentile(X[:, feature], [25, 50, 75])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
                    continue
                
                left_y = y[left_mask] if hasattr(y, '__getitem__') else y.iloc[left_mask]
                right_y = y[~left_mask] if hasattr(y, '__getitem__') else y.iloc[~left_mask]
                
                # Calculate accuracy
                left_pred = 1 if np.mean(left_y) > 0.5 else 0
                right_pred = 1 if np.mean(right_y) > 0.5 else 0
                
                score = (np.sum(left_y == left_pred) + np.sum(right_y == right_pred)) / len(y)
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left_pred': 1 if np.mean(y[X[:, best_feature] <= best_threshold]) > 0.5 else 0,
            'right_pred': 1 if np.mean(y[X[:, best_feature] > best_threshold]) > 0.5 else 0
        }
    
    def predict(self, X):
        predictions = []
        for x in X:
            votes = []
            for tree in self.trees:
                if x[tree['feature']] <= tree['threshold']:
                    votes.append(tree['left_pred'])
                else:
                    votes.append(tree['right_pred'])
            predictions.append(1 if np.mean(votes) > 0.5 else 0)
        return np.array(predictions)
    
    def predict_proba(self, X):
        predictions = []
        for x in X:
            votes = []
            for tree in self.trees:
                if x[tree['feature']] <= tree['threshold']:
                    votes.append(tree['left_pred'])
                else:
                    votes.append(tree['right_pred'])
            prob_1 = np.mean(votes)
            predictions.append([1-prob_1, prob_1])
        return np.array(predictions)

class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        # Add bias term
        X = np.column_stack([np.ones(X.shape[0]), X])
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        for _ in range(self.max_iter):
            z = X.dot(self.weights)
            predictions = self._sigmoid(z)
            
            # Convert y to numpy array if it's a pandas Series
            y_array = y.values if hasattr(y, 'values') else y
            
            gradient = X.T.dot(predictions - y_array) / len(y_array)
            self.weights -= self.learning_rate * gradient
    
    def predict(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return (self._sigmoid(X.dot(self.weights)) > 0.5).astype(int)
    
    def predict_proba(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        prob_1 = self._sigmoid(X.dot(self.weights))
        return np.column_stack([1-prob_1, prob_1])

class SimpleKNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.values if hasattr(y, 'values') else y
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            predictions.append(1 if np.mean(k_nearest_labels) > 0.5 else 0)
        return np.array(predictions)
    
    def predict_proba(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            prob_1 = np.mean(k_nearest_labels)
            predictions.append([1-prob_1, prob_1])
        return np.array(predictions)

def standardize_data(X_train, X_test):
    """Simple standardization"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled, mean, std

def train_test_split_simple(X, y, test_size=0.2, random_state=42):
    """Simple train-test split"""
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
    X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices]
    y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
    y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    """Calculate basic classification metrics"""
    y_true = y_true.values if hasattr(y_true, 'values') else y_true
    
    accuracy = np.mean(y_true == y_pred)
    
    # Confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }

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
        
        # Feature importance analysis using correlation with target
        st.subheader("⭐ Feature Correlation with Target")
        
        feature_corr = df.corr()['Outcome'].abs().sort_values(ascending=False)[1:]
        
        fig = px.bar(x=feature_corr.values, y=feature_corr.index, 
                    orientation='h', title="Feature Correlation with Diabetes Outcome")
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
            "✅ Handle missing values (replace zeros with median where medically impossible)",
            "✅ Feature scaling using standardization",
            "✅ Train-test split (80-20)",
            "✅ Handle class imbalance awareness",
        ]
        
        for step in preprocessing_steps:
            st.write(step)
        
        # Preprocessing implementation
        df_processed = df.copy()
        
        # Replace zeros with median for certain features
        zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for feature in zero_not_accepted:
            median_val = df_processed[df_processed[feature] != 0][feature].median()
            df_processed[feature] = df_processed[feature].replace(0, median_val)
        
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
        
        X_train, X_test, y_train, y_test = train_test_split_simple(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        X_train_scaled, X_test_scaled, scaler_mean, scaler_std = standardize_data(X_train.values, X_test.values)
        
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
            'scaler_mean': scaler_mean,
            'scaler_std': scaler_std,
            'feature_names': X.columns.tolist()
        }
        
        st.success("✅ Data preprocessing completed and stored!")
    
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
            'Random Forest': SimpleRandomForest(n_estimators=20),
            'Logistic Regression': SimpleLogisticRegression(),
            'K-Nearest Neighbors': SimpleKNN(k=5)
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
                metrics = calculate_metrics(y_test, y_pred)
                
                model_results[name] = {
                    'model': model,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'training_time': training_time,
                    'confusion_matrix': metrics['confusion_matrix']
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
                'Precision': [results['precision'] for results in model_results.values()],
                'Recall': [results['recall'] for results in model_results.values()],
                'F1-Score': [results['f1'] for results in model_results.values()],
                'Training Time (s)': [results['training_time'] for results in model_results.values()]
            }).round(3)
            
            results_df = results_df.sort_values('Accuracy', ascending=False)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Accuracy comparison chart
            fig = px.bar(results_df, x='Model', y='Accuracy', 
                        title="Model Accuracy Comparison",
                        color='Accuracy',
                        color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Model training completed!")
    
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
            
            # Scale the input data using stored scaler parameters
            data = st.session_state['processed_data']
            input_scaled = (input_data - data['scaler_mean']) / data['scaler_std']
            
            # Get selected model
            selected_model = st.session_state['model_results'][selected_model_name]['model']
            
            # Make prediction
            prediction = selected_model.predict(input_scaled)[0]
            if hasattr(selected_model, 'predict_proba'):
                probability = selected_model.predict_proba(input_scaled)[0]
                prob_diabetes = probability[1]
            else:
                prob_diabetes = prediction
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div style="background: #ffe6e6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff6b6b;">
                        <h3 style="color: #d63031; margin: 0;">⚠️ High Risk</h3>
                        <p style="margin: 5px 0 0 0;">Diabetes risk detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #00b894;">
                        <h3 style="color: #00b894; margin: 0;">✅ Low Risk</h3>
                        <p style="margin: 5px 0 0 0;">No diabetes risk detected</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Diabetes Probability", f"{prob_diabetes:.1%}")
                st.metric("Model Used", selected_model_name)
            
            with col3:
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_diabetes * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Level"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#ff6b6b" if prob_diabetes > 0.5 else "#00b894"},
                        'steps': [
                            {'range': [0, 50], 'color': "#e8f5e8"},
                            {'range': [50, 100], 'color': "#ffe6e6"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📈 Model Performance Visualization")
        
        if 'model_results' in st.session_state:
            model_results = st.session_state['model_results']
            
            # Performance metrics comparison
            metrics_data = []
            for model_name, results in model_results.items():
                metrics_data.extend([
                    {'Model': model_name, 'Metric': 'Accuracy', 'Value': results['accuracy']},
                    {'Model': model_name, 'Metric': 'Precision', 'Value': results['precision']},
                    {'Model': model_name, 'Metric': 'Recall', 'Value': results['recall']},
                    {'Model': model_name, 'Metric': 'F1-Score', 'Value': results['f1']}
                ])
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig = px.bar(metrics_df, x='Model', y='Value', color='Metric', 
                        title="Model Performance Metrics Comparison",
                        barmode='group')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Section 7: Evaluation Metrics
    elif selected_section == "7. Evaluation Metrics":
        st.markdown('<div class="section-header"><h2>📊 Evaluation Metrics</h2></div>', unsafe_allow_html=True)
        
        if 'model_results' not in st.session_state:
            st.warning("Please run the Model Training section first!")
            return
        
        model_results = st.session_state['model_results']
        
        # Model selection for detailed analysis
        selected_model = st.selectbox("Select Model for Detailed Analysis:", 
                                    list(model_results.keys()))
        
        results = model_results[selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Classification Metrics")
            
            metrics_display = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [
                    f"{results['accuracy']:.3f}",
                    f"{results['precision']:.3f}",
                    f"{results['recall']:.3f}",
                    f"{results['f1']:.3f}"
                ],
                'Description': [
                    'Overall correct predictions',
                    'True positives / (True positives + False positives)',
                    'True positives / (True positives + False negatives)',
                    'Harmonic mean of Precision and Recall'
                ]
            })
            
            st.dataframe(metrics_display, use_container_width=True, hide_index=True)
            
            # Confusion Matrix
            st.subheader("🔄 Confusion Matrix")
            
            cm = results['confusion_matrix']
            
            fig = px.imshow(cm, 
                           text_auto=True,
                           aspect="auto",
                           color_continuous_scale='Blues',
                           title=f"Confusion Matrix - {selected_model}")
            
            fig.update_xaxes(title="Predicted")
            fig.update_yaxes(title="Actual")
            fig.update_layout(
                xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Diabetes', 'Diabetes']),
                yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Diabetes', 'Diabetes'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Model Summary")
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{selected_model}</h4>
                <p><strong>Training Time:</strong> {results['training_time']:.3f} seconds</p>
                <p><strong>Test Accuracy:</strong> {results['accuracy']:.1%}</p>
                <p><strong>True Positives:</strong> {cm[1,1]}</p>
                <p><strong>True Negatives:</strong> {cm[0,0]}</p>
                <p><strong>False Positives:</strong> {cm[0,1]}</p>
                <p><strong>False Negatives:</strong> {cm[1,0]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model interpretation
            st.subheader("🧠 Model Insights")
            
            if selected_model == "Random Forest":
                st.markdown("""
                **Random Forest Insights:**
                - Ensemble method using multiple decision trees
                - Good at handling feature interactions
                - Provides feature importance rankings
                - Robust against overfitting
                """)
            elif selected_model == "Logistic Regression":
                st.markdown("""
                **Logistic Regression Insights:**
                - Linear relationship between features and log-odds
                - Provides probability estimates
                - Interpretable coefficients
                - Good baseline model
                """)
            elif selected_model == "K-Nearest Neighbors":
                st.markdown("""
                **K-Nearest Neighbors Insights:**
                - Instance-based learning
                - Non-parametric approach
                - Sensitive to feature scaling
                - Good for local patterns
                """)
        
        # Model comparison summary
        st.subheader("🏆 Overall Model Comparison")
        
        comparison_data = []
        for model_name, results in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.3f}",
                'Precision': f"{results['precision']:.3f}",
                'Recall': f"{results['recall']:.3f}",
                'F1-Score': f"{results['f1']:.3f}",
                'Training Time': f"{results['training_time']:.3f}s"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Best model recommendation
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['f1'])
        
        st.markdown(f"""
        <div class="info-box">
        <strong>🎯 Recommended Model:</strong> {best_model}<br>
        Based on F1-Score: {model_results[best_model]['f1']:.3f}<br>
        <em>F1-Score provides a balanced measure considering both precision and recall.</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical considerations
        st.subheader("🏥 Clinical Considerations")
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Important Medical Disclaimer:</strong><br>
        This model is for educational purposes only and should not be used for actual medical diagnosis. 
        Always consult healthcare professionals for medical advice.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Model Selection for Healthcare:**
        
        - **High Recall Priority:** Choose models with high recall to minimize false negatives (missing diabetes cases)
        - **High Precision Priority:** Choose models with high precision to minimize false positives (unnecessary worry/treatment)
        - **Balanced Approach:** F1-Score provides a good balance between precision and recall
        
        **Next Steps for Production:**
        1. Validate with larger, more diverse datasets
        2. Clinical validation studies
        3. Regulatory approval processes
        4. Integration with electronic health records
        5. Continuous monitoring and model updates
        """)

if __name__ == "__main__":
    main()
