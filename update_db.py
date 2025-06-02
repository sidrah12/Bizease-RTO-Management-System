from app import app, db
from sqlalchemy import text

def add_barcode_column():
    with app.app_context():
        try:
            # Add barcode column if it doesn't exist
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE `order` ADD COLUMN IF NOT EXISTS barcode VARCHAR(50) UNIQUE'))
                print("Successfully added barcode column to orders table")
                
                # Generate barcodes for existing orders
                conn.execute(text("""
                    UPDATE `order` 
                    SET barcode = CONCAT('BAR', id) 
                    WHERE barcode IS NULL
                """))
                print("Successfully updated existing orders with barcodes")
                
        except Exception as e:
            print(f"Error updating database: {str(e)}")

if __name__ == '__main__':
    add_barcode_column()