from app import app, db, Order
from rto_model import RTOPredictor
import pandas as pd

def main():
    with app.app_context():
        # Check total orders
        total_orders = db.session.query(Order).count()
        print(f"Total Orders in Database: {total_orders}")
        
        # Initialize the RTO predictor
        predictor = RTOPredictor()
        
        # Try to load existing model
        if predictor.load_model():
            print("Loaded existing model")
        else:
            print("No existing model found, will need to train new model")
        
        # Get all orders
        orders = Order.query.all()
        
        if orders:
            # Convert orders to DataFrame
            order_data = []
            for order in orders:
                order_data.append({
                    'address_classification': order.address_classification,
                    'status': order.status,
                    'payment_method': order.payment_method,
                    'quantity': order.quantity,
                    'city': order.city,
                    'state': order.state,
                    'pincode': order.pincode,
                    'rto_risk': order.rto_risk
                })
            
            df = pd.DataFrame(order_data)
            print("\nOrder Data Sample:")
            print(df.head())
            
            # Make predictions
            if len(df) > 0:
                predictions, probabilities = predictor.predict(df)
                print("\nPredictions:")
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    print(f"Order {i+1}:")
                    print(f"Predicted RTO Risk: {pred}")
                    print(f"Risk Probabilities: {prob}")
                    print()
        else:
            print("No orders found in database")

if __name__ == "__main__":
    main() 