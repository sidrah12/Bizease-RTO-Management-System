from app import app, db, Product, Order
from datetime import datetime, timedelta
import random

def seed_database():
    with app.app_context():
        print("Clearing existing data...")
        Order.query.delete()
        Product.query.delete()
        db.session.commit()

        print("Creating products...")
        products = [
            Product(
                title="iPhone 14 Pro",
                description="Latest Apple iPhone with 128GB storage and advanced camera system",
                sku="IP14P-128-BLK",
                category="Electronics",
                price=129999.00,
                stock=15,
                expiration_date=None
            ),
            Product(
                title="Samsung Galaxy S23",
                description="Premium Android smartphone with 256GB storage and 108MP camera",
                sku="SGS23-256-GRN",
                category="Electronics",
                price=89999.00,
                stock=8,
                expiration_date=None
            ),
            Product(
                title="Nike Air Max",
                description="Premium sports shoes with advanced air cushioning technology",
                sku="NAM-42-BLK",
                category="Footwear",
                price=8999.00,
                stock=25,
                expiration_date=None
            ),
            Product(
                title="Levi's 501 Jeans",
                description="Classic straight fit denim jeans in dark blue wash",
                sku="L501-32-BLU",
                category="Apparel",
                price=4999.00,
                stock=45,
                expiration_date=None
            ),
            Product(
                title="Sony WH-1000XM4",
                description="Premium noise-cancelling wireless headphones",
                sku="SWHXM4-BLK",
                category="Electronics",
                price=24999.00,
                stock=5,
                expiration_date=None
            ),
            Product(
                title="MacBook Air M2",
                description="Latest MacBook Air with M2 chip and 256GB storage",
                sku="MBA-M2-256",
                category="Electronics",
                price=114999.00,
                stock=12,
                expiration_date=None
            ),
            Product(
                title="Organic Coffee Beans",
                description="Premium Arabica coffee beans, 500g pack",
                sku="OCB-500G",
                category="Food",
                price=599.00,
                stock=100,
                expiration_date=datetime.now() + timedelta(days=180)
            ),
            Product(
                title="Vitamin C Supplements",
                description="1000mg Vitamin C tablets, 60 count",
                sku="VCS-60TAB",
                category="Health",
                price=499.00,
                stock=75,
                expiration_date=datetime.now() + timedelta(days=365)
            ),
            Product(
                title="Gaming Mouse",
                description="RGB gaming mouse with 16000 DPI optical sensor",
                sku="GM-RGB-BLK",
                category="Electronics",
                price=2999.00,
                stock=30,
                expiration_date=None
            ),
            Product(
                title="Yoga Mat",
                description="Professional non-slip yoga mat with carrying strap",
                sku="YM-PRO-PUR",
                category="Sports",
                price=1499.00,
                stock=20,
                expiration_date=None
            )
        ]

        for product in products:
            db.session.add(product)
        db.session.commit()

        print("Creating orders...")
        # Customer data for realistic orders
        customers = [
            ("Rahul Kumar", "rahul.k@email.com", "9876543210", "Delhi", "Delhi", "110001"),
            ("Priya Singh", "priya.s@email.com", "8765432109", "Mumbai", "Maharashtra", "400001"),
            ("Amit Sharma", "amit.s@email.com", "7654321098", "Bangalore", "Karnataka", "560001"),
            ("Sneha Patel", "sneha.p@email.com", "6543210987", "Ahmedabad", "Gujarat", "380001"),
            ("Karthik Raj", "karthik.r@email.com", "9876543211", "Chennai", "Tamil Nadu", "600001")
        ]

        # Payment methods
        payment_methods = ["UPI", "Credit Card", "Cash on Delivery", "Net Banking"]
        
        # Status options
        statuses = ["Pending", "Shipped", "Delivered", "Cancelled"]
        
        # RTO risk levels
        rto_risks = ["High", "Medium", "Low"]

        # Generate 20 orders
        orders = []
        current_date = datetime.now()
        
        for i in range(20):
            customer = random.choice(customers)
            product = random.choice(products)
            status = random.choice(statuses)
            
            # Adjust dates to be more realistic
            days_ago = random.randint(0, 30)
            order_date = current_date - timedelta(days=days_ago)
            
            # Set more realistic RTO risk based on payment method and location
            payment_method = random.choice(payment_methods)
            if payment_method == "Cash on Delivery":
                rto_risk = random.choice(["High", "Medium"])
            else:
                rto_risk = random.choice(["Medium", "Low"])

            # Calculate amount based on product price and quantity
            quantity = random.randint(1, 3)
            amount = product.price * quantity

            # Generate order number
            order_number = f"ORD{str(i+1).zfill(5)}"

            # Create address with both lines
            address_line1 = f"{random.randint(1, 100)}, {random.choice(['Main Road', 'Cross Street', 'Avenue'])}"
            address_line2 = random.choice(["Near Park", "Behind Mall", "Next to Hospital", ""])
            full_address = f"{address_line1}, {address_line2}" if address_line2 else address_line1

            order = Order(
                order_number=order_number,
                customer_name=customer[0],
                email=customer[1],
                phone=customer[2],
                address=full_address,
                city=customer[3],
                state=customer[4],
                pincode=customer[5],
                payment_method=payment_method,
                amount=amount,
                status=status,
                product_id=product.id,
                quantity=quantity,
                rto_risk=rto_risk,
                created_at=order_date,
                address_classification=random.choice(["Residential", "Commercial"]),
                barcode=f"BAR{str(i+1).zfill(5)}"
            )
            orders.append(order)

        for order in orders:
            db.session.add(order)
        db.session.commit()

        print("Seed data created successfully!")

if __name__ == "__main__":
    seed_database() 