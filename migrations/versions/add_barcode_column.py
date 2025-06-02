"""Add barcode column to orders

Revision ID: add_barcode_column
Revises: 
Create Date: 2024-03-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_barcode_column'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add barcode column to orders table
    op.add_column('order', sa.Column('barcode', sa.String(50), nullable=True, unique=True))

def downgrade():
    # Remove barcode column from orders table
    op.drop_column('order', 'barcode') 