from app import app, db
from flask_migrate import Migrate, upgrade

migrate = Migrate(app, db)

with app.app_context():
    print("Running database migration...")
    upgrade()
    print("Migration completed successfully.") 