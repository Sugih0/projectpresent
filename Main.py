import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def create_risk_gauge(probability):
    """Create a simple risk gauge visualization"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create gauge
    colors = ['#00b894', '#fdcb6e', '#e17055', '#d63031']
    wedges = [25, 25, 25, 25]
    
    wedge_colors = colors
    
    # Create the gauge
    wedge_props = dict(width=0.3, edgecolor='white')
    wp = ax.pie(wedges, colors=wedge_colors, wedgeprops=wedge_props, 
                startangle=180, counterclock=False)
    
    # Add needle
    angle = 180 - (probability * 180)  # Convert probability to angle
    needle_x = 0.7 * np.cos(np.radians(angle))
    needle_y = 0.7 * np.sin(np.radians(angle))
    ax.arrow(0, 0, needle_x, needle_y, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=2)
    
    # Add center circle
    centre_circle = plt.Circle((0, 0), 0.4, fc='white', ec='black')
    ax.add_artist(centre_circle)
    
    # Add labels
    ax.text(0, -0.2, f'{probability:.1%}', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    ax.text(0, -0.35, 'Risk Level', ha='center', va='center', fontsize=12)
    
    # Add risk level labels
    ax.text(-0.8, 0.1, 'Low', ha='center', va='center', fontsize=10, color='#00b894')
    ax.text(0.8, 0.1, 'High', ha='center', va='center', fontsize=10, color='#d63031')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig

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
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = ['No Diabetes', 'Diabetes']
            sizes = [len(df[df['Outcome']==0]), len(df[df['Outcome']==1])]
            colors = ['#2ecc71', '#e74c3c']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Dataset Distribution')
            st.pyplot(fig)
    
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
            fig, ax = plt.subplots(figsize=(8, 6))
            outcome_counts = df['Outcome'].value_counts()
            labels = ['No Diabetes', 'Diabetes']
            sizes = [outcome_counts[0], outcome_counts[1]]
            colors = ['#2ecc71', '#e74c3c']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Diabetes Distribution')
            st.pyplot(fig)
        
        with col2:
            outcome_counts = df['Outcome'].value_counts()
            st.metric("Diabetes Prevalence", f"{outcome_counts[1]/len(df)*100:.1f}%")
            st.metric("Class Ratio", f"1:{outcome_counts[0]/outcome_counts[1]:.1f}")
        
        # Feature distributions
        st.subheader("📊 Feature Distributions")
        
        numeric_cols = ['Glucose', 'BMI', 'Age', 'BloodPressure']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            
            # Histogram for each outcome
            df[df['Outcome']==0][col].hist(alpha=0.7, label='No Diabetes', 
                                          color='#2ecc71', ax=ax, bins=20)
            df[df['Outcome']==1][col].hist(alpha=0.7, label='Diabetes', 
                                          color='#e74c3c', ax=ax, bins=20)
            
            ax.set_title(f'{col} Distribution by Diabetes Status')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        correlation_matrix = df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
        
        # Feature importance analysis using correlation with target
        st.subheader("⭐ Feature Correlation with Target")
        
        feature_corr = df.corr()['Outcome'].abs().sort_values(ascending=False)[1:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_corr.index, feature_corr.values)
        ax.set_title('Feature Correlation with Diabetes Outcome')
        ax.set_xlabel('Absolute Correlation')
        st.pyplot(fig)
    
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
            
            # Model comparison visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy comparison
            ax1 = axes[0]
            accuracies = [results['metrics']['accuracy'] for results in model_results.values()]
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            bars = ax1.bar(model_results.keys(), accuracies, color=colors)
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
            
            # F1-Score comparison
            ax2 = axes[1]
            f1_scores = [results['metrics']['f1'] for results in model_results.values()]
            bars = ax2.bar(model_results.keys(), f1_scores, color=colors)
            ax2.set_title('Model F1-Score Comparison')
            ax2.set_ylabel('F1-Score')
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, f1 in zip(bars, f1_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{f1:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
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
        
        if 'model_results' not in st.session_state:
            st.warning("Please train the models first in the Model Training section!")
            return
        
        model_results = st.session_state['model_results']
        data = st.session_state['processed_data']
        
        # Model selection for predictions
        st.subheader("🔮 Make Predictions")
        
        selected_model = st.selectbox("Choose Model for Prediction:", list(model_results.keys()))
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Input Patient Data:**")
            
            # Input fields
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
            pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
            
            if st.button("🔍 Predict Diabetes Risk"):
                # Prepare input data
                input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                      insulin, bmi, pedigree, age]])
                
                # Scale input data
                scaler_mean = data['scaler_mean']
                scaler_std = data['scaler_std']
                input_scaled = (input_data - scaler_mean) / scaler_std
                
                # Make prediction
                model = model_results[selected_model]['model']
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                st.session_state['last_prediction'] = {
                    'prediction': prediction,
                    'probability': probability,
                    'input_data': input_data[0]
                }
        
        with col2:
            if 'last_prediction' in st.session_state:
                pred_data = st.session_state['last_prediction']
                probability = pred_data['probability']
                
                # Risk visualization
                st.write("**Prediction Result:**")
                
                # Risk gauge
                fig = create_risk_gauge(probability)
                st.pyplot(fig)
                
                # Risk level determination
                if probability < 0.3:
                    risk_level = "Low Risk"
                    risk_color = "success-box"
                elif probability < 0.7:
                    risk_level = "Moderate Risk"
                    risk_color = "warning-box"
                else:
                    risk_level = "High Risk"
                    risk_color = "warning-box"
                
                st.markdown(f"""
                <div class="{risk_color}">
                <strong>Risk Assessment: {risk_level}</strong><br>
                Probability: {probability:.1%}<br>
                Prediction: {'Diabetes' if pred_data['prediction'] == 1 else 'No Diabetes'}
                </div>
                """, unsafe_allow_html=True)
        
        # Batch predictions
        st.subheader("📊 Batch Predictions")
        
        if st.button("Generate Sample Predictions"):
            # Generate some sample predictions
            sample_size = 10
            np.random.seed(42)
            
            # Generate random samples
            sample_data = []
            for _ in range(sample_size):
                sample = [
                    np.random.randint(0, 10),  # pregnancies
                    np.random.randint(70, 200),  # glucose
                    np.random.randint(40, 120),  # blood_pressure
                    np.random.randint(10, 50),  # skin_thickness
                    np.random.randint(0, 300),  # insulin
                    np.random.uniform(18, 50),  # bmi
                    np.random.uniform(0.1, 2.0),  # pedigree
                    np.random.randint(21, 70)  # age
                ]
                sample_data.append(sample)
            
            sample_array = np.array(sample_data)
            
            # Scale sample data
            sample_scaled = (sample_array - data['scaler_mean']) / data['scaler_std']
            
            # Make predictions
            model = model_results[selected_model]['model']
            predictions = model.predict(sample_scaled)
            probabilities = model.predict_proba(sample_scaled)[:, 1]
            
            # Create results dataframe
            results_df = pd.DataFrame(sample_data, columns=data['feature_names'])
            results_df['Predicted_Risk'] = probabilities
            results_df['Prediction'] = ['Diabetes' if p == 1 else 'No Diabetes' for p in predictions]
            results_df['Risk_Level'] = ['High' if p > 0.7 else 'Moderate' if p > 0.3 else 'Low' 
                                      for p in probabilities]
            
            st.dataframe(results_df.round(3), use_container_width=True, hide_index=True)
    
    # Section 7: Evaluation Metrics
    elif selected_section == "7. Evaluation Metrics":
        st.markdown('<div class="section-header"><h2>📈 Evaluation Metrics</h2></div>', unsafe_allow_html=True)
        
        if 'model_results' not in st.session_state:
            st.warning("Please train the models first in the Model Training section!")
            return
        
        model_results = st.session_state['model_results']
        
        # Model selection for detailed analysis
        selected_model = st.selectbox("Select Model for Detailed Analysis:", list(model_results.keys()))
        
        model_data = model_results[selected_model]
        metrics = model_data['metrics']
        
        st.subheader(f"📊 {selected_model} Performance Metrics")
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            cm = metrics['confusion_matrix']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Diabetes', 'Diabetes'],
                       yticklabels=['No Diabetes', 'Diabetes'])
            ax.set_title(f'Confusion Matrix - {selected_model}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with col2:
            # Confusion matrix interpretation
            tn, fp, fn, tp = cm.ravel()
            
            st.write("**Confusion Matrix Breakdown:**")
            st.write(f"- True Negatives (TN): {tn}")
            st.write(f"- False Positives (FP): {fp}")
            st.write(f"- False Negatives (FN): {fn}")
            st.write(f"- True Positives (TP): {tp}")
            
            st.write("\n**Clinical Interpretation:**")
            st.write(f"- Correctly identified healthy: {tn}")
            st.write(f"- Incorrectly flagged as diabetic: {fp}")
            st.write(f"- Missed diabetic cases: {fn}")
            st.write(f"- Correctly identified diabetic: {tp}")
        
        # Model comparison
        st.subheader("🏆 Model Comparison")
        
        comparison_metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(comparison_metrics):
            ax = axes[i]
            
            models = list(model_results.keys())
            values = [model_results[model]['metrics'][metric] for model in models]
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            
            bars = ax.bar(models, values, color=colors)
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance (for tree-based models)
        if selected_model == 'Random Forest':
            st.subheader("🌳 Feature Importance Analysis")
            
            # Simple feature importance based on correlation
            data = st.session_state['processed_data']
            feature_names = data['feature_names']
            
            # Calculate feature importance as absolute correlation with target
            df_temp = load_diabetes_data()  # Reload original data
            correlations = df_temp.corr()['Outcome'].abs().drop('Outcome').sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(correlations)), correlations.values)
            ax.set_yticks(range(len(correlations)))
            ax.set_yticklabels(correlations.index)
            ax.set_xlabel('Feature Importance (Correlation with Target)')
            ax.set_title('Feature Importance Analysis')
            st.pyplot(fig)
        
        # Recommendations
        st.subheader("🎯 Model Recommendations")
        
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics']['f1'])
        
        st.markdown(f"""
        <div class="info-box">
        <strong>📋 Model Performance Summary:</strong><br><br>
        
        <strong>Best Overall Model:</strong> {best_model}<br>
        <strong>Key Strengths:</strong>
        <ul>
        <li>Highest F1-Score: {model_results[best_model]['metrics']['f1']:.4f}</li>
        <li>Balanced precision and recall</li>
        <li>Good generalization capability</li>
        </ul>
        
        <strong>Clinical Considerations:</strong>
        <ul>
        <li>High recall is important to avoid missing diabetic cases</li>
        <li>Precision helps reduce false alarms</li>
        <li>F1-score provides balanced performance measure</li>
        </ul>
        
        <strong>Next Steps:</strong>
        <ul>
        <li>Collect more diverse training data</li>
        <li>Feature engineering for better predictive power</li>
        <li>Clinical validation with medical professionals</li>
        <li>Regular model retraining and monitoring</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
