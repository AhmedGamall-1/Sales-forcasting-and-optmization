import streamlit as st
import pandas as pd
from typing import Dict, Any

def display_prediction_result(result: Dict[str, Any]) -> None:
    """Display the prediction results."""
    st.success("Prediction successful!")
    
    # Create two columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        features_df = pd.DataFrame([result["features"]])
        st.dataframe(features_df)
    
    with col2:
        st.subheader("Prediction")
        st.metric(
            label="Predicted Sales",
            value=f"${result['predicted_sales']:,.2f}"
        )
        
        # Add a visual indicator
        st.progress(min(result['predicted_sales'] / 100000, 1.0)) 