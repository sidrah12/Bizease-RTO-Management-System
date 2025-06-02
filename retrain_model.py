from app import app, rto_predictor
import pandas as pd

# Create sample training data with balanced classes and Unknown values
sample_data = pd.DataFrame({
    'address_classification': ['Remote', 'Urban', 'Suburban', 'Unknown'] * 5,
    'status': ['Pending', 'Shipped', 'Delivered', 'Pending'] * 5,
    'payment_method': ['COD', 'Online', 'COD', 'UPI'] * 5,
    'quantity': [1, 2, 3, 1] * 5,
    'city': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'] * 5,
    'state': ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu'] * 5,
    'pincode': ['400001', '110001', '560001', '600001'] * 5,
    'rto_risk': ['High', 'Low', 'Medium', 'Medium'] * 5
})

with app.app_context():
    print("Training RTO model...")
    accuracy, precision, recall, f1 = rto_predictor.train(sample_data)
    print(f"Model metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    rto_predictor.save_model()
    print("Model saved successfully.") 