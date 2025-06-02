import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from sqlalchemy import text

def upgrade():
    with app.app_context():
        try:
            # Drop existing tables
            db.session.execute(text("DROP TABLE IF EXISTS `order`"))
            db.session.execute(text("DROP TABLE IF EXISTS `product`"))
            
            # Create product table
            db.session.execute(text("""
                CREATE TABLE `product` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(100) NOT NULL,
                    description TEXT NOT NULL,
                    image VARCHAR(200),
                    sku VARCHAR(50) UNIQUE,
                    price FLOAT NOT NULL DEFAULT 0.0,
                    stock INT DEFAULT 0,
                    category VARCHAR(50),
                    dimensions VARCHAR(100),
                    weight FLOAT,
                    shelf_life INT,
                    expiration_date DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create order table without foreign key constraint initially
            db.session.execute(text("""
                CREATE TABLE `order` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    order_number VARCHAR(50) UNIQUE NOT NULL,
                    customer_name VARCHAR(100) NOT NULL,
                    email VARCHAR(120) NOT NULL,
                    phone VARCHAR(20) NOT NULL,
                    address TEXT NOT NULL,
                    city VARCHAR(50) NOT NULL,
                    state VARCHAR(50) NOT NULL,
                    pincode VARCHAR(10) NOT NULL,
                    payment_method VARCHAR(20) NOT NULL,
                    amount FLOAT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'Pending',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    product_id INT NOT NULL,
                    quantity INT DEFAULT 1,
                    address_classification VARCHAR(20),
                    rto_risk VARCHAR(20),
                    rto_probability FLOAT
                )
            """))
            
            # Insert sample products
            db.session.execute(text("""
                INSERT INTO product (id, title, description, price, stock, category)
                VALUES 
                (1, 'Sample Product 1', 'Description for product 1', 999.99, 10, 'Electronics'),
                (2, 'Sample Product 2', 'Description for product 2', 499.99, 5, 'Clothing'),
                (3, 'Sample Product 3', 'Description for product 3', 1499.99, 8, 'Home')
            """))
            
            # Insert sample orders
            db.session.execute(text("""
                INSERT INTO `order` (
                    order_number, customer_name, email, phone, address, city, state, pincode,
                    payment_method, amount, status, product_id, quantity, address_classification, rto_risk, rto_probability
                ) VALUES 
                ('ORD001', 'John Doe', 'john@example.com', '1234567890', '123 Main St', 'Mumbai', 'Maharashtra', '400001', 'COD', 999.99, 'Pending', 1, 1, 'Urban', 'Medium', 0.5),
                ('ORD002', 'Jane Smith', 'jane@example.com', '9876543210', '456 Rural Rd', 'Nagpur', 'Maharashtra', '440001', 'Online', 499.99, 'Shipped', 2, 2, 'Remote', 'High', 0.8),
                ('ORD003', 'Mike Johnson', 'mike@example.com', '5555555555', '789 Suburb Ave', 'Pune', 'Maharashtra', '411001', 'COD', 1499.99, 'Delivered', 3, 1, 'Suburban', 'Low', 0.2)
            """))
            
            # Add foreign key constraint after data is inserted
            db.session.execute(text("""
                ALTER TABLE `order`
                ADD CONSTRAINT fk_order_product
                FOREIGN KEY (product_id) REFERENCES product(id)
            """))
            
            db.session.commit()
            print("Migration completed successfully!")
        except Exception as e:
            print(f"Error during migration: {str(e)}")
            db.session.rollback()

def downgrade():
    with app.app_context():
        try:
            # Drop the table
            db.session.execute(text("DROP TABLE IF EXISTS `order`"))
            db.session.commit()
            print("Downgrade completed successfully!")
        except Exception as e:
            print(f"Error during downgrade: {str(e)}")
            db.session.rollback()

if __name__ == "__main__":
    upgrade() 