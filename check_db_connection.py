from app import app, db
from sqlalchemy import text

def check_database():
    with app.app_context():
        try:
            # Check if we can connect to the database
            with db.engine.connect() as conn:
                print("Successfully connected to the database")
                
                # Check if tables exist
                result = conn.execute(text("SHOW TABLES"))
                tables = [row[0] for row in result]
                print("\nExisting tables:")
                for table in tables:
                    print(f"- {table}")
                
                # Check if we have any data
                if 'order' in tables:
                    order_count = conn.execute(text("SELECT COUNT(*) FROM `order`")).scalar()
                    print(f"\nTotal orders in database: {order_count}")
                
                if 'product' in tables:
                    product_count = conn.execute(text("SELECT COUNT(*) FROM product")).scalar()
                    print(f"Total products in database: {product_count}")
            
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")

if __name__ == '__main__':
    check_database() 