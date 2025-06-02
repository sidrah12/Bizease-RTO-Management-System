import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from sqlalchemy import text

def upgrade():
    with app.app_context():
        try:
            # Add new columns if they don't exist
            db.session.execute(text("""
                ALTER TABLE `order` 
                ADD COLUMN IF NOT EXISTS address_classification VARCHAR(20),
                ADD COLUMN IF NOT EXISTS rto_probability FLOAT
            """))
            
            # Update existing orders with default values
            db.session.execute(text("""
                UPDATE `order` 
                SET address_classification = 'Urban',
                    rto_risk = 'Medium',
                    rto_probability = 0.5
                WHERE address_classification IS NULL
            """))
            
            db.session.commit()
            print("Migration completed successfully!")
        except Exception as e:
            print(f"Error during migration: {str(e)}")
            db.session.rollback()

def downgrade():
    with app.app_context():
        try:
            # Remove columns
            db.session.execute(text("""
                ALTER TABLE `order` 
                DROP COLUMN IF EXISTS address_classification,
                DROP COLUMN IF EXISTS rto_probability
            """))
            
            db.session.commit()
            print("Downgrade completed successfully!")
        except Exception as e:
            print(f"Error during downgrade: {str(e)}")
            db.session.rollback()

if __name__ == "__main__":
    upgrade() 