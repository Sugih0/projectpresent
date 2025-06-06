import streamlit as st
import pandas as pd
import numpy as np
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
    
    .success-box {
        background: #e8f5e8;
        border: 1px solid #00b894;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #00b894;
    }
    
    .risk-high {
        background: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        color: #d63031;
    }
    
    .risk-low {
        background: #e8f5e8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00b894;
        color: #00b894;
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

def create_risk_display(probability):
    """Create a text-based risk display"""
    # Create a simple progress bar using text
    bar_length = 20
    filled_length = int(bar_length * probability)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    if probability < 0.3:
        risk_level = "Low Risk 🟢"
        color_class = "success-box"
    elif probability < 0.7:
        risk_level = "Moderate Risk 🟡"
        color_class = "warning-box"
    else:
        risk_level = "High Risk 🔴"
        color_class = "warning-box"
    
    return f"""
    <div class="{color_class}">
    <h3 style="text-align: center;">{risk_level}</h3>
    <p style="text-align: center; font-family: monospace; font-size: 18px;">{bar}</p>
    <p style="text-align: center; font-size: 24px; font-weight: bold;">{probability:.1%}</p>
    </div>
    """

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
            # Dataset distribution using Streamlit chart
            outcome_counts = df['Outcome'].value_counts()
            chart_data = pd.DataFrame({
                'Category': ['No Diabetes', 'Diabetes'],
                'Count': [outcome_counts[0], outcome_counts[1]]
            })
            st.subheader("Dataset Distribution")
            st.bar_chart(chart_data.set_index('Category'))
    
    # Section 2: Dataset Overview
    elif selected_section == "2. Dataset Overview":
        st.markdown('<div class="section-header"><h2>📊 Dataset Overview</h2></div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Pima Indians Diabetes Dataset
        **Source:** National Institute of Diabetes and Digestive and Kidney Diseases
        
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
            outcome_counts = df['Outcome'].value_counts()
            chart_data = pd.DataFrame({
                'Category': ['No Diabetes', 'Diabetes'],
                'Count': [outcome_counts[0], outcome_counts[1]]
            })
            st.bar_chart(chart_data.set_index('Category'))
        
        with col2:
            st.metric("Diabetes Prevalence", f"{outcome_counts[1]/len(df)*100:.1f}%")
            st.metric("Class Ratio", f"1:{outcome_counts[0]/outcome_counts[1]:.1f}")
        
        # Feature distributions
        st.subheader("📊 Feature Distributions")
        
        # Select feature to analyze
        feature_to_analyze = st.selectbox("Select Feature to Analyze:", 
                                        ['Glucose', 'BMI', 'Age', 'BloodPressure'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{feature_to_analyze} - No Diabetes**")
            no_diabetes_data = df[df['Outcome']==0][feature_to_analyze]
            st.line_chart(no_diabetes_data.value_counts().sort_index())
        
        with col2:
            st.write(f"**{feature_to_analyze} - Diabetes**")
            diabetes_data = df[df['Outcome']==1][feature_to_analyze]
            st.line_chart(diabetes_data.value_counts().sort_index())
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        correlation_matrix = df.corr()
        st.dataframe(correlation_matrix, use_container_width=True)
        
        # Feature importance analysis using correlation with target
        st.subheader("⭐ Feature Correlation with Target")
        
        feature_corr = df.corr()['Outcome'].abs().sort_values(ascending=False)[1:]
        corr_data = pd.DataFrame({
            'Feature': feature_corr.index,
            'Correlation': feature_corr.values
        })
        st.bar_chart(corr_data.set_index('Feature'))
    
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
                y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred)
                
                model_results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'metrics': metrics,
                    'training_time': training_time
                }
                
                progress_bar.progress((i + 1) / len(models))
            
            status_text.text("Training completed!")
            st.session_state['model_results'] = model_results
            
            # Display results
            st.subheader("📊 Training Results")
            
            results_df = pd.DataFrame({
                'Model': list(model_results.keys()),
                'Accuracy': [results['metrics']['accuracy'] for results in model_results.values()],
                'Precision': [results['metrics']['precision'] for results in model_results.values()],
                'Recall': [results['metrics']['recall'] for results in model_results.values()],
                'F1-Score': [results['metrics']['f1'] for results in model_results.values()],
                'Training Time (s)': [results['training_time'] for results in model_results.values()]
            })
            
            st.dataframe(results_df.round(4), use_container_width=True, hide_index=True)
            
            # Model comparison visualization using Streamlit charts
            st.subheader("📈 Model Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Accuracy Comparison**")
                accuracy_data = pd.DataFrame({
                    'Model': list(model_results.keys()),
                    'Accuracy': [results['metrics']['accuracy'] for results in model_results.values()]
                })
                st.bar_chart(accuracy_data.set_index('Model'))
            
            with col2:
                st.write("**F1-Score Comparison**")
                f1_data = pd.DataFrame({
                    'Model': list(model_results.keys()),
                    'F1-Score': [results['metrics']['f1'] for results in model_results.values()]
                })
                st.bar_chart(f1_data.set_index('Model'))
            
            # Best model selection
            best_model_name = max(model_results.keys(), 
                                key=lambda x: model_results[x]['metrics']['f1'])
            
            st.markdown(f"""
            <div class="success-box">
            <strong>🏆 Best Performing Model: {best_model_name}</strong><br>
            F1-Score: {model_results[best_model_name]['metrics']['f1']:.4f}<br>
            Accuracy: {model_results[best_model_name]['metrics']['accuracy']:.4f}
            </div>
            """, unsafe_allow_html=True)
    
    # Section 6: Results & Predictions
    elif selected_section == "6. Results & Predictions":
        st.markdown('<div class="section-header"><h2>🎯 Results & Predictions</h2></div>', unsafe_allow_html=True)
        
        if 'model_results' not
        # Section 6: Results & Predictions (continued from where it left off)
    elif selected_section == "6. Results & Predictions":
        st.markdown('<div class="section-header"><h2>🎯 Results & Predictions</h2></div>', unsafe_allow_html=True)
        
        if 'model_results' not in st.session_state:
            st.warning("Please run the Model Training section first!")
            return
        
        model_results = st.session_state['model_results']
        data = st.session_state['processed_data']
        
        # Best model selection
        best_model_name = max(model_results.keys(), 
                            key=lambda x: model_results[x]['metrics']['f1'])
        best_model = model_results[best_model_name]['model']
        
        st.subheader(f"🏆 Using Best Model: {best_model_name}")
        
        # Individual prediction interface
        st.subheader("🔮 Make Individual Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0)
            pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        
        if st.button("🎯 Predict Diabetes Risk"):
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, pedigree, age]])
            
            # Scale input data
            scaler_mean = data['scaler_mean']
            scaler_std = data['scaler_std']
            input_scaled = (input_data - scaler_mean) / scaler_std
            
            # Make prediction
            prediction = best_model.predict(input_scaled)[0]
            prediction_proba = best_model.predict_proba(input_scaled)[0]
            
            # Display results
            diabetes_probability = prediction_proba[1]
            
            st.markdown(create_risk_display(diabetes_probability), unsafe_allow_html=True)
            
            # Additional insights
            if diabetes_probability > 0.7:
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ High Risk Detected</strong><br>
                Recommendations:
                <ul>
                <li>Consult with a healthcare professional immediately</li>
                <li>Consider diabetes screening tests</li>
                <li>Monitor blood glucose regularly</li>
                <li>Adopt a healthy diet and exercise routine</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            elif diabetes_probability > 0.3:
                st.markdown("""
                <div class="info-box">
                <strong>🟡 Moderate Risk</strong><br>
                Recommendations:
                <ul>
                <li>Schedule regular check-ups</li>
                <li>Maintain a healthy lifestyle</li>
                <li>Monitor your weight and BMI</li>
                <li>Stay physically active</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <strong>✅ Low Risk</strong><br>
                Keep up the good work! Continue maintaining:
                <ul>
                <li>Healthy diet</li>
                <li>Regular exercise</li>
                <li>Normal weight</li>
                <li>Regular health check-ups</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Batch predictions
        st.subheader("📊 Test Set Predictions Overview")
        
        y_test = data['y_test'].values if hasattr(data['y_test'], 'values') else data['y_test']
        test_predictions = model_results[best_model_name]['predictions']
        test_probabilities = model_results[best_model_name]['probabilities']
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': test_predictions,
            'Probability': test_probabilities[:, 1],
            'Correct': y_test == test_predictions
        })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Correct Predictions", f"{np.sum(results_df['Correct'])}/{len(results_df)}")
        with col2:
            st.metric("Accuracy", f"{np.mean(results_df['Correct']):.1%}")
        with col3:
            st.metric("High Risk Cases", f"{np.sum(test_probabilities[:, 1] > 0.7)}")
        
        # Show sample predictions
        st.subheader("🔍 Sample Predictions")
        
        # Select random samples to display
        sample_indices = np.random.choice(len(results_df), min(10, len(results_df)), replace=False)
        sample_df = results_df.iloc[sample_indices].copy()
        
        # Format for display
        display_df = sample_df.copy()
        display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.1%}")
        display_df['Status'] = display_df['Correct'].apply(lambda x: "✅ Correct" if x else "❌ Wrong")
        display_df = display_df[['Actual', 'Predicted', 'Probability', 'Status']]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Section 7: Evaluation Metrics
    elif selected_section == "7. Evaluation Metrics":
        st.markdown('<div class="section-header"><h2>📈 Evaluation Metrics</h2></div>', unsafe_allow_html=True)
        
        if 'model_results' not in st.session_state:
            st.warning("Please run the Model Training section first!")
            return
        
        model_results = st.session_state['model_results']
        
        # Model selection for detailed analysis
        selected_model = st.selectbox("Select Model for Detailed Analysis:", 
                                    list(model_results.keys()))
        
        model_data = model_results[selected_model]
        metrics = model_data['metrics']
        
        st.subheader(f"📊 Detailed Metrics for {selected_model}")
        
        # Key metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.3f}")
        
        # Confusion Matrix
        st.subheader("🎯 Confusion Matrix")
        
        cm = metrics['confusion_matrix']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display confusion matrix as a formatted table
            cm_df = pd.DataFrame(cm, 
                               columns=['Predicted: No Diabetes', 'Predicted: Diabetes'],
                               index=['Actual: No Diabetes', 'Actual: Diabetes'])
            st.dataframe(cm_df, use_container_width=True)
        
        with col2:
            # Confusion matrix interpretation
            tn, fp, fn, tp = cm.ravel()
            
            st.markdown(f"""
            **Matrix Interpretation:**
            - True Negatives (TN): {tn}
            - False Positives (FP): {fp}
            - False Negatives (FN): {fn}
            - True Positives (TP): {tp}
            
            **Clinical Impact:**
            - False Negatives: {fn} diabetes cases missed
            - False Positives: {fp} unnecessary concerns
            """)
        
        # Performance comparison across all models
        st.subheader("🏆 Model Performance Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'Accuracy': [results['metrics']['accuracy'] for results in model_results.values()],
            'Precision': [results['metrics']['precision'] for results in model_results.values()],
            'Recall': [results['metrics']['recall'] for results in model_results.values()],
            'F1-Score': [results['metrics']['f1'] for results in model_results.values()],
            'Training Time': [f"{results['training_time']:.3f}s" for results in model_results.values()]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Performance analysis
        st.subheader("🔍 Performance Analysis")
        
        best_accuracy = max(model_results.values(), key=lambda x: x['metrics']['accuracy'])
        best_precision = max(model_results.values(), key=lambda x: x['metrics']['precision'])
        best_recall = max(model_results.values(), key=lambda x: x['metrics']['recall'])
        best_f1 = max(model_results.values(), key=lambda x: x['metrics']['f1'])
        
        analysis_text = f"""
        **Model Performance Summary:**
        
        - **Best Accuracy**: {best_accuracy['metrics']['accuracy']:.3f} 
        - **Best Precision**: {best_precision['metrics']['precision']:.3f} (Fewer false positives)
        - **Best Recall**: {best_recall['metrics']['recall']:.3f} (Fewer missed diabetes cases)
        - **Best F1-Score**: {best_f1['metrics']['f1']:.3f} (Best balance)
        
        **Key Insights:**
        - High precision reduces unnecessary anxiety from false positives
        - High recall ensures fewer diabetes cases are missed
        - F1-score provides the best overall balance for medical applications
        """
        
        st.markdown(analysis_text)
        
        # Clinical recommendations
        st.subheader("🏥 Clinical Recommendations")
        
        st.markdown("""
        <div class="info-box">
        <strong>Model Deployment Considerations:</strong><br><br>
        
        <strong>For Screening Programs:</strong>
        <ul>
        <li>Prioritize high recall to catch all potential cases</li>
        <li>Accept some false positives for comprehensive screening</li>
        </ul>
        
        <strong>For Risk Assessment:</strong>
        <ul>
        <li>Balance precision and recall using F1-score</li>
        <li>Consider probability thresholds based on risk tolerance</li>
        </ul>
        
        <strong>For Clinical Decision Support:</strong>
        <ul>
        <li>Use model predictions as additional information, not replacement for clinical judgment</li>
        <li>Consider individual patient context and medical history</li>
        <li>Regular model retraining with new data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Model limitations
        st.subheader("⚠️ Model Limitations & Disclaimers")
        
        st.markdown("""
        <div class="warning-box">
        <strong>Important Limitations:</strong><br>
        <ul>
        <li>This model is for educational/demonstration purposes only</li>
        <li>Not validated for clinical use</li>
        <li>Based on limited historical data</li>
        <li>May not generalize to all populations</li>
        <li>Should not replace professional medical advice</li>
        <li>Regular retraining needed with new data</li>
        </ul>
        
        <strong>Disclaimer:</strong> Always consult healthcare professionals for medical decisions.
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
    <p>🏥 <strong>Diabetes Prediction ML Project</strong></p>
    <p>Complete end-to-end machine learning pipeline for educational purposes</p>
    <p><em>Always consult healthcare professionals for medical advice</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
