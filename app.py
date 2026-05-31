import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Suppress DLL loading warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set page configuration
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .header {
        color: #2E86AB;
        text-align: center;
        padding: 20px 0;
    }
    .prediction-box {
        background-color: #A23B72;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessors from pickle files"""
    try:
        # Load the model
        with open('best_bean_classifier_xgb.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Title and description
st.markdown("<h1 class='header'>🫘 Dry Bean Type Classifier</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #555; margin-bottom: 30px;'>
    <p>Predict dry bean types based on physical characteristics using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Load model
model = load_model_and_preprocessors()

if model is None:
    st.error("Failed to load the model. Please ensure 'best_bean_classifier_xgb.pkl' is in the same directory.")
else:
    # Define feature names (based on your dataset after removing Solidity and ShapeFactor4)
    feature_names = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRatio', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
        'Extent', 'Roundness', 'Compactness', 'ShapeFactor1',
        'ShapeFactor2', 'ShapeFactor3'
    ]
    
    # Bean class names (7 classes)
    bean_classes = [
        'SEKER', 'BARBUNYA', 'BOMBAY', 'CALI',
        'HOROZ', 'SIRA', 'WHITE'
    ]
    
    # Create two columns for the main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Enter Bean Measurements")
        
        # Create input fields for each feature in an organized grid
        input_values = []
        
        # Organize features in 2 columns for better UI
        col_left, col_right = st.columns(2)
        
        with col_left:
            area = st.number_input("Area", min_value=0.0, value=50000.0, step=100.0, help="Pixel area of the bean")
            perimeter = st.number_input("Perimeter", min_value=0.0, value=500.0, step=10.0, help="Perimeter of the bean")
            major_axis = st.number_input("Major Axis Length", min_value=0.0, value=300.0, step=10.0)
            minor_axis = st.number_input("Minor Axis Length", min_value=0.0, value=200.0, step=10.0)
            aspect_ratio = st.number_input("Aspect Ratio", min_value=0.0, value=1.5, step=0.1)
            eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            convex_area = st.number_input("Convex Area", min_value=0.0, value=55000.0, step=100.0)
        
        with col_right:
            equiv_diameter = st.number_input("Equivalent Diameter", min_value=0.0, value=250.0, step=10.0)
            extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
            roundness = st.number_input("Roundness", min_value=0.0, value=0.8, step=0.05)
            compactness = st.number_input("Compactness", min_value=0.0, value=0.7, step=0.05)
            shape_factor1 = st.number_input("Shape Factor 1", min_value=0.0, value=0.5, step=0.05)
            shape_factor2 = st.number_input("Shape Factor 2", min_value=0.0, value=0.5, step=0.05)
            shape_factor3 = st.number_input("Shape Factor 3", min_value=0.0, value=0.5, step=0.05)
        
        input_values = [
            area, perimeter, major_axis, minor_axis, aspect_ratio,
            eccentricity, convex_area, equiv_diameter, extent, roundness,
            compactness, shape_factor1, shape_factor2, shape_factor3
        ]
    
    # Prediction section
    with col2:
        st.subheader("🎯 Prediction")
        
        if st.button("🔍 Predict Bean Type", use_container_width=True, type="primary"):
            # Prepare the input for the model
            input_array = np.array(input_values).reshape(1, -1)
            
            try:
                # Make prediction
                prediction = model.predict(input_array)[0]
                probabilities = model.predict_proba(input_array)[0]
                
                # Display prediction result
                predicted_class = bean_classes[int(prediction)]
                confidence = probabilities[int(prediction)] * 100
                
                st.markdown(f"""
                    <div class='prediction-box'>
                    <h2>{predicted_class}</h2>
                    <p>Confidence: {confidence:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display probability distribution
                st.subheader("📈 Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Bean Type': bean_classes,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                # Create a bar chart
                st.bar_chart(prob_df.set_index('Bean Type'))
                
                # Display detailed probabilities
                st.subheader("📋 Detailed Results")
                for bean_type, prob in zip(prob_df['Bean Type'], prob_df['Probability']):
                    st.write(f"**{bean_type}**: {prob*100:.2f}%")
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # Additional information
    st.divider()
    
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        ### Model Information
        - **Model Type**: XGBoost Classifier
        - **Number of Classes**: 7 bean types
        - **Features Used**: 14 physical bean characteristics
        - **Dataset**: Dry Bean Classification Dataset
        
        ### Bean Types
        1. **SEKER** - Turkish bean variety
        2. **BARBUNYA** - Red bean type
        3. **BOMBAY** - Dark round bean
        4. **CALI** - Light colored bean
        5. **HOROZ** - Horoz bean variety
        6. **SIRA** - Purple bean type
        7. **WHITE** - White bean variety
        
        ### Features Description
        - **Area**: Pixel area of the bean image
        - **Perimeter**: Boundary perimeter
        - **Major/Minor Axis**: Dimensions of the ellipse fitted to the bean
        - **Aspect Ratio**: Ratio of major to minor axis
        - **Eccentricity**: Measure of how elongated the bean is
        - **Convex Area**: Area of the convex hull
        - **Equivalent Diameter**: Diameter of a circle with the same area
        - **Extent**: Ratio of area to bounding rectangle
        - **Roundness**: Measure of how round the bean is
        - **Compactness**: Measure of bean compactness
        - **Shape Factors**: Additional geometric descriptors
        """)
    
    with st.expander("🔧 How to Use"):
        st.markdown("""
        1. **Enter Measurements**: Input the physical measurements of a dry bean in the left panel
        2. **Click Predict**: Press the "Predict Bean Type" button
        3. **View Results**: See the predicted bean type and confidence score
        4. **Analyze Probabilities**: Check the probability distribution across all bean types
        
        **Note**: All input values should be numeric and in appropriate ranges based on your bean measurements.
        """)
