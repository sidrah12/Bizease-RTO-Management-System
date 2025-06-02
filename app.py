from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import pandas as pd
from io import BytesIO
from rto_model import RTOPredictor

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/ecommerce_db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize RTO predictor
rto_predictor = RTOPredictor()

# Try to load existing model
if not rto_predictor.load_model():
    print("No existing RTO model found. A new model will be trained when data is available.")

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        print(f"Setting password for user {self.email}")
        print(f"Original password: {password}")
        # Use pbkdf2:sha256 method with a fixed salt for testing
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        print(f"Generated hash: {self.password_hash}")

    def check_password(self, password):
        print(f"Checking password for user {self.email}")
        print(f"Stored hash: {self.password_hash}")
        print(f"Input password: {password}")
        result = check_password_hash(self.password_hash, password)
        print(f"Password check result: {result}")
        return result

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    image = db.Column(db.String(200))
    sku = db.Column(db.String(50), unique=True, nullable=False)
    price = db.Column(db.Float, nullable=False, default=0.0)
    stock = db.Column(db.Integer, default=0)
    category = db.Column(db.String(50))
    dimensions = db.Column(db.String(100))
    weight = db.Column(db.Float)
    shelf_life = db.Column(db.Integer, nullable=False)  # Shelf life in days
    expiration_date = db.Column(db.DateTime)  # Calculated expiration date
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def calculate_expiration_date(self):
        if self.shelf_life:
            self.expiration_date = self.created_at + timedelta(days=self.shelf_life)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(50), unique=True, nullable=False)
    customer_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    address = db.Column(db.Text, nullable=False)
    city = db.Column(db.String(50), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    pincode = db.Column(db.String(10), nullable=False)
    payment_method = db.Column(db.String(20), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='Pending')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    barcode = db.Column(db.String(50), unique=True, nullable=True)  # Added barcode column
    
    # RTO-related fields
    address_classification = db.Column(db.String(20), nullable=True)  # Remote, Urban, Suburban
    rto_risk = db.Column(db.String(20), nullable=True)  # High, Medium, Low
    rto_probability = db.Column(db.Float, nullable=True)  # Probability score for RTO risk
    
    def __repr__(self):
        return f'<Order {self.order_number}>'

def predict_rto_risk(order_data):
    """Predict RTO risk using the trained model"""
    try:
        print("Starting RTO risk prediction...")
        # Convert order data to the format expected by the model
        if isinstance(order_data, dict):
            data = pd.DataFrame({
                'address_classification': [order_data.get('address_classification', 'Unknown')],
                'status': [order_data.get('status', 'Pending')],
                'payment_method': [order_data.get('payment_method', 'Unknown')],
                'quantity': [order_data.get('quantity', 1)],
                'city': [order_data.get('city', 'Unknown')],
                'state': [order_data.get('state', 'Unknown')],
                'pincode': [order_data.get('pincode', 'Unknown')]
            })
            print("Input data:", data)
        else:
            # If it's an Order object
            data = pd.DataFrame({
                'address_classification': [getattr(order_data, 'address_classification', 'Unknown')],
                'status': [getattr(order_data, 'status', 'Pending')],
                'payment_method': [getattr(order_data, 'payment_method', 'Unknown')],
                'quantity': [getattr(order_data, 'quantity', 1)],
                'city': [getattr(order_data, 'city', 'Unknown')],
                'state': [getattr(order_data, 'state', 'Unknown')],
                'pincode': [getattr(order_data, 'pincode', 'Unknown')]
            })
            print("Input data from Order object:", data)
        
        # Replace None values with 'Unknown'
        data = data.fillna('Unknown')
        
        if rto_predictor.model is None:
            print("No RTO model loaded - falling back to basic logic")
            raise Exception("RTO model not initialized")
            
        # Make prediction
        print("Making prediction with model...")
        predictions, probabilities = rto_predictor.predict(data)
        print(f"Model prediction: {predictions[0]}, Probabilities: {probabilities[0]}")
        
        # Return just the risk value
        return predictions[0]
    except Exception as e:
        print(f"Error predicting RTO risk: {str(e)}")
        print("Falling back to basic logic...")
        # Fallback to basic logic if model fails
        if isinstance(order_data, dict):
            address_class = order_data.get('address_classification')
            status = order_data.get('status')
            payment_method = order_data.get('payment_method')
        else:
            address_class = getattr(order_data, 'address_classification', None)
            status = getattr(order_data, 'status', None)
            payment_method = getattr(order_data, 'payment_method', None)
            
        print(f"Order details - Address Class: {address_class}, Status: {status}, Payment: {payment_method}")
        
        # Handle None or Unknown values
        address_class = 'Unknown' if address_class is None else address_class
        status = 'Pending' if status is None else status
        payment_method = 'Unknown' if payment_method is None else payment_method
        
        # Updated risk assessment logic
        if payment_method == 'COD':
            if address_class == 'Remote' or status == 'Pending':
                return 'High'
            else:
                return 'Medium'
        elif address_class == 'Remote':
            return 'High'
        elif status == 'Pending':
            return 'Medium'
        elif status == 'Shipped':
            return 'Medium'
        else:
            return 'Low'

def classify_address(address):
    # Placeholder logic to classify address as Urban or Remote
    urban_postal_codes = ['10001', '10002', '10003']  # Example urban postal codes
    remote_postal_codes = ['20001', '20002']  # Example remote postal codes
    # Extract postal code from address
    postal_code = address.split()[-1]  # Assuming postal code is the last part of the address
    if postal_code in urban_postal_codes:
        return 'Urban'
    elif postal_code in remote_postal_codes:
        return 'Remote'
    else:
        return 'Unknown'

@app.route('/')
@login_required
def dashboard():
    try:
        print("\n=== Dashboard Debug ===")
        
        # Basic counts
        products_count = Product.query.count()
        print(f"Products count: {products_count}")
        
        orders_count = Order.query.count()
        print(f"Orders count: {orders_count}")
        
        # Today's metrics
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        # Today's and yesterday's orders
        today_orders = Order.query.filter(
            db.func.date(Order.created_at) == today
        ).count()
        print(f"Today's orders: {today_orders}")
        
        yesterday_orders = Order.query.filter(
            db.func.date(Order.created_at) == yesterday
        ).count() or 1  # Avoid division by zero
        print(f"Yesterday's orders: {yesterday_orders}")
        
        # Today's and yesterday's revenue
        today_revenue = db.session.query(
            db.func.sum(Order.quantity * Product.price)
        ).join(Product).filter(
            db.func.date(Order.created_at) == today
        ).scalar() or 0
        print(f"Today's revenue: {today_revenue}")
        
        yesterday_revenue = db.session.query(
            db.func.sum(Order.quantity * Product.price)
        ).join(Product).filter(
            db.func.date(Order.created_at) == yesterday
        ).scalar() or 1  # Avoid division by zero
        print(f"Yesterday's revenue: {yesterday_revenue}")
        
        # Last 30 days metrics
        thirty_days_ago = today - timedelta(days=30)
        
        # Average order value (last 30 days)
        thirty_day_revenue = db.session.query(
            db.func.sum(Order.quantity * Product.price)
        ).join(Product).filter(
            db.func.date(Order.created_at) >= thirty_days_ago
        ).scalar() or 0
        print(f"30-day revenue: {thirty_day_revenue}")
        
        thirty_day_orders_count = Order.query.filter(
            db.func.date(Order.created_at) >= thirty_days_ago
        ).count() or 1  # Avoid division by zero
        print(f"30-day orders count: {thirty_day_orders_count}")
        
        avg_order_value = thirty_day_revenue / thirty_day_orders_count if thirty_day_orders_count > 0 else 0
        print(f"Average order value: {avg_order_value}")
        
        # RTO Rate calculation (last 30 days)
        high_rto_count = Order.query.filter(
            db.func.date(Order.created_at) >= thirty_days_ago,
            Order.rto_risk == 'High'
        ).count()
        print(f"High RTO count: {high_rto_count}")
        
        rto_rate = (high_rto_count / thirty_day_orders_count * 100) if thirty_day_orders_count > 0 else 0
        print(f"RTO rate: {rto_rate}")
        
        # Order status counts
        status_data = {
            status: Order.query.filter_by(status=status).count()
            for status in ['Pending', 'Shipped', 'Delivered', 'Cancelled']
        }
        print(f"Status data: {status_data}")
        
        # RTO risk distribution
        rto_data = {
            risk: Order.query.filter_by(rto_risk=risk).count()
            for risk in ['High', 'Medium', 'Low']
        }
        print(f"RTO data: {rto_data}")
        
        # Recent orders with product information
        recent_orders = db.session.query(Order, Product).join(
            Product, Order.product_id == Product.id
        ).order_by(Order.created_at.desc()).limit(10).all()
        print(f"Recent orders count: {len(recent_orders)}")
        
        # Low stock products (less than 10 units)
        low_stock_products = Product.query.filter(Product.stock < 10).all()
        print(f"Low stock products count: {len(low_stock_products)}")
        
        # Products expiring soon (within 30 days)
        expiry_threshold = datetime.now() + timedelta(days=30)
        expiring_products = Product.query.filter(
            Product.expiration_date <= expiry_threshold,
            Product.expiration_date >= datetime.now()
        ).all()
        print(f"Expiring products count: {len(expiring_products)}")
        
        # Last check timestamps
        low_stock_last_check = datetime.now().strftime('%Y-%m-%d %H:%M')
        expiry_last_check = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Maximum products (for progress bar)
        max_products = max(1000, products_count * 2)
        
        print("=== End Dashboard Debug ===\n")
        
        return render_template('dashboard.html',
                            products_count=products_count,
                            orders_count=orders_count,
                            today_orders=today_orders,
                            yesterday_orders=yesterday_orders,
                            today_revenue=today_revenue,
                            yesterday_revenue=yesterday_revenue,
                            avg_order_value=avg_order_value,
                            rto_rate=rto_rate,
                            recent_orders=recent_orders,
                            low_stock_products=low_stock_products,
                            expiring_products=expiring_products,
                            status_data=status_data,
                            rto_data=rto_data,
                            low_stock_last_check=low_stock_last_check,
                            expiry_last_check=expiry_last_check,
                            max_products=max_products)
                            
    except Exception as e:
        print(f"Dashboard Error: {str(e)}")
        # Return a basic version of the dashboard with minimal data
        return render_template('dashboard.html',
                            products_count=0,
                            orders_count=0,
                            today_orders=0,
                            yesterday_orders=1,
                            today_revenue=0,
                            yesterday_revenue=1,
                            avg_order_value=0,
                            rto_rate=0,
                            recent_orders=[],
                            low_stock_products=[],
                            expiring_products=[],
                            status_data={'Pending': 0, 'Shipped': 0, 'Delivered': 0, 'Cancelled': 0},
                            rto_data={'High': 0, 'Medium': 0, 'Low': 0},
                            low_stock_last_check=datetime.now().strftime('%Y-%m-%d %H:%M'),
                            expiry_last_check=datetime.now().strftime('%Y-%m-%d %H:%M'),
                            max_products=1000)

@app.route('/products')
def products():
    products = Product.query.all()
    return render_template('products.html', products=products)

@app.route('/products/add', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        try:
            # Debug print form data
            print("Form data:", request.form)
            
            # Get image if exists
            image = request.files.get('image')
            filename = None
            if image and image.filename:
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}"
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Get shelf_life from form, default to 0 if not provided
            shelf_life = int(request.form.get('shelf_life', 0))
            
            # Create product with all required fields
            product = Product(
                title=request.form.get('title'),
                description=request.form.get('description'),
                image=filename,
                sku=request.form.get('sku'),
                price=float(request.form.get('price', 0)),
                stock=int(request.form.get('stock', 0)),
                category=request.form.get('category'),
                dimensions=request.form.get('dimensions'),
                weight=float(request.form.get('weight', 0)),
                shelf_life=shelf_life,
                created_at=datetime.utcnow()  # Explicitly set created_at
            )
            
            # Calculate expiration date
            product.calculate_expiration_date()
            
            # Add to database
            db.session.add(product)
            db.session.commit()
            
            print("Product added successfully:", product.id)
            return redirect(url_for('products'))
            
        except Exception as e:
            db.session.rollback()
            print("Error adding product:", str(e))
            return render_template('add_product.html', error=str(e))
    
    return render_template('add_product.html')

@app.route('/products/edit/<int:id>', methods=['GET', 'POST'])
def edit_product(id):
    product = Product.query.get_or_404(id)
    if request.method == 'POST':
        product.title = request.form['title']
        product.description = request.form['description']
        image = request.files['image']
        if image:
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}"
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            product.image = filename
        product.sku = request.form['sku']
        product.stock = int(request.form['stock'])
        product.price = float(request.form['price'])
        product.category = request.form['category']
        product.dimensions = request.form['dimensions']
        product.weight = float(request.form['weight'])
        product.shelf_life = int(request.form['shelf_life'])
        product.calculate_expiration_date()
        db.session.commit()
        return redirect(url_for('products'))
    return render_template('edit_product.html', product=product)

@app.route('/products/delete/<int:id>', methods=['POST'])
def delete_product(id):
    product = Product.query.get_or_404(id)
    db.session.delete(product)
    db.session.commit()
    return redirect(url_for('products'))

@app.route('/products/<int:id>')
def product_detail(id):
    product = Product.query.get_or_404(id)
    return render_template('product_detail.html', product=product)

@app.route('/orders')
def orders():
    orders = Order.query.all()
    return render_template('orders.html', orders=orders)

@app.route('/add_order', methods=['GET', 'POST'])
def add_order():
    if request.method == 'POST':
        try:
            # Get the product
            product = Product.query.get(request.form['product_id'])
            if not product:
                raise ValueError("Product not found")
            
            # Check stock availability
            order_quantity = int(request.form['quantity'])
            if product.stock < order_quantity:
                raise ValueError(f"Insufficient stock. Available: {product.stock}, Requested: {order_quantity}")
            
            # Generate a unique order number and barcode
            timestamp = int(datetime.now().timestamp())
            order_number = 'ORD' + str(timestamp)
            barcode = 'BAR' + str(timestamp)
            
            # Create order with basic information
            order = Order(
                order_number=order_number,
                barcode=barcode,  # Add barcode
                customer_name=request.form['customer_name'],
                email=request.form['email'],
                phone=request.form['phone'],
                address=request.form['address'],
                city=request.form['city'],
                state=request.form['state'],
                pincode=request.form['pincode'],
                payment_method=request.form['payment_method'],
                quantity=order_quantity,
                product_id=product.id,
                amount=float(request.form.get('amount', 0)),
                status='Pending'  # Set default status
            )
            
            # Classify address and predict RTO risk
            order.address_classification = classify_address(order.address)
            rto_prediction = predict_rto_risk(order)
            order.rto_risk = rto_prediction
            
            # Update product stock
            product.stock -= order_quantity
            
            # Save to database
            db.session.add(order)
            db.session.commit()
            
            print(f"Order created successfully: {order_number}, Barcode: {barcode}")
            return redirect(url_for('orders'))
            
        except ValueError as e:
            db.session.rollback()
            print("Validation error:", str(e))
            return render_template('add_order.html', products=Product.query.all(), error=str(e))
        except Exception as e:
            db.session.rollback()
            print("Error creating order:", str(e))
            return render_template('add_order.html', products=Product.query.all(), error=str(e))
    
    products = Product.query.all()
    return render_template('add_order.html', products=products)

@app.route('/orders/edit/<int:id>', methods=['GET', 'POST'])
def edit_order(id):
    order = Order.query.get_or_404(id)
    if request.method == 'POST':
        try:
            # Debug print form data
            print("Form data:", request.form)
            
            # Validate required fields
            required_fields = ['customer_name', 'email', 'phone', 'address', 
                             'city', 'state', 'pincode', 'payment_method', 'quantity', 
                             'product_id', 'status']
            
            for field in required_fields:
                if field not in request.form or not request.form[field]:
                    raise ValueError(f"Missing required field: {field}")
            
            # Update order fields
            order.customer_name = request.form['customer_name']
            order.email = request.form['email']
            order.phone = request.form['phone']
            order.address = request.form['address']
            order.city = request.form['city']
            order.state = request.form['state']
            order.pincode = request.form['pincode']
            order.payment_method = request.form['payment_method']
            order.quantity = int(request.form['quantity'])
            order.product_id = int(request.form['product_id'])
            order.status = request.form['status']
            order.amount = float(request.form.get('amount', 0))
            
            # Update address classification and RTO risk
            order.address_classification = classify_address(order.address)
            rto_prediction = predict_rto_risk(order)
            order.rto_risk = rto_prediction
            order.rto_probability = 0.8  # Assuming a default confidence
            
            db.session.commit()
            print("Order updated successfully:", order.id)
            return redirect(url_for('orders'))
            
        except ValueError as e:
            db.session.rollback()
            print("Validation error:", str(e))
            return render_template('edit_order.html', order=order, products=Product.query.all(), error=str(e))
        except Exception as e:
            db.session.rollback()
            print("Error updating order:", str(e))
            return render_template('edit_order.html', order=order, products=Product.query.all(), error=str(e))
    
    products = Product.query.all()
    return render_template('edit_order.html', order=order, products=products)

@app.route('/orders/delete/<int:id>', methods=['POST'])
def delete_order(id):
    order = Order.query.get_or_404(id)
    db.session.delete(order)
    db.session.commit()
    return redirect(url_for('orders'))

@app.route('/orders/<int:id>')
def order_detail(id):
    order = Order.query.get_or_404(id)
    return render_template('order_detail.html', order=order)

@app.route('/rto_checker', methods=['GET', 'POST'])
def rto_checker():
    if request.method == 'POST':
        file = request.files['file']
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            # Read the file and process orders
            df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
            rto_results = []
            for index, row in df.iterrows():
                order = {
                    'first_name': row['first_name'],
                    'last_name': row['last_name'],
                    'email': row['email'],
                    'phone': row['phone'],
                    'address_line1': row['address_line1'],
                    'address_line2': row['address_line2'],
                    'city': row['city'],
                    'state': row['state'],
                    'pincode': row['pincode'],
                    'payment_method': row['payment_method'],
                    'quantity': row['quantity'],
                    'status': row['status']
                }
                print(order)  # Debugging: print the order dictionary
                order['address_classification'] = classify_address(order['address_line1'])
                order['rto_risk'] = predict_rto_risk(order)
                rto_results.append(order)
            return render_template('rto_checker.html', rto_results=rto_results)
    return render_template('rto_checker.html')

@app.route('/download_rto_template')
def download_rto_template():
    # Create a sample template with required fields
    template_data = {
        'first_name': ['John', 'Jane', 'Michael'],
        'last_name': ['Doe', 'Smith', 'Johnson'],
        'email': ['john.doe@example.com', 'jane.smith@example.com', 'michael.johnson@example.com'],
        'phone': ['1234567890', '0987654321', '5555555555'],
        'address_line1': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
        'address_line2': ['Apt 4B', '', 'Suite 100'],
        'city': ['New York', 'Los Angeles', 'Chicago'],
        'state': ['NY', 'CA', 'IL'],
        'pincode': ['10001', '90001', '60601'],
        'payment_method': ['Credit Card', 'PayPal', 'Bank Transfer'],
        'quantity': [1, 2, 3],
        'status': ['Pending', 'Shipped', 'Delivered']
    }
    
    # Create DataFrame
    df = pd.DataFrame(template_data)
    
    # Create a BytesIO object to store the CSV
    output = BytesIO()
    
    # Write the DataFrame to the BytesIO object as CSV
    df.to_csv(output, index=False)
    
    # Seek to the beginning of the BytesIO object
    output.seek(0)
    
    # Return the CSV file as a download
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='rto_template.csv'
    )

@app.route('/barcode_scanner', methods=['GET', 'POST'])
def barcode_scanner():
    if request.method == 'POST':
        barcode_data = request.form['barcode']
        # Find the order by barcode
        order = Order.query.filter_by(barcode=barcode_data).first()
        
        if order:
            if order.status == 'Cancelled':
                return render_template('_order_details.html', order=order)
            else:
                return render_template('_error_message.html', 
                                    error='Only cancelled orders can be restocked')
        else:
            return render_template('_error_message.html', 
                                error='Order not found')
    
    return render_template('barcode_scanner.html')

@app.route('/restock/<int:order_id>', methods=['POST'])
def restock(order_id):
    try:
        # Get the order
        order = Order.query.get_or_404(order_id)
        
        # Check if order is cancelled
        if order.status != 'Cancelled':
            return render_template('_error_message.html', 
                                error='Only cancelled orders can be restocked')
        
        # Get the product
        product = Product.query.get(order.product_id)
        if not product:
            raise ValueError("Product not found")
        
        # Update order status and invalidate barcode
        order.status = 'Return-Restocked'
        order.barcode = f"RESTOCKED_{order_id}"  # Set to a special value that can never be used for new orders
        
        # Restock the product
        product.stock += order.quantity
        
        # Commit changes
        db.session.commit()
        
        return render_template('_success_message.html', 
                            message=f'Order #{order.order_number} has been successfully restocked')
        
    except Exception as e:
        db.session.rollback()
        return render_template('_error_message.html', error=str(e))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        print(f"\n=== Login Attempt ===")
        print(f"Email: {email}")
        print(f"Password: {password}")
        print(f"Remember: {remember}")
        
        user = User.query.filter_by(email=email).first()
        
        if user:
            print(f"User found: {user.email}")
            print(f"User's stored hash: {user.password_hash}")
            if user.check_password(password):
                print("Password correct, logging in")
                login_user(user, remember=remember)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('dashboard'))
            else:
                print("Password incorrect")
        else:
            print(f"No user found with email: {email}")
        
        flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/create_admin', methods=['GET', 'POST'])
def create_admin():
    if User.query.filter_by(is_admin=True).first():
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        print(f"\n=== Admin Creation Attempt ===")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Password: {password}")
        
        if not all([username, email, password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('create_admin'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('create_admin'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('create_admin'))
        
        user = User(username=username, email=email, is_admin=True)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        print(f"Admin user created successfully: {user.email}")
        print(f"Stored password hash: {user.password_hash}")
        
        flash('Admin user created successfully', 'success')
        return redirect(url_for('login'))
    
    return render_template('create_admin.html')

# Protect all routes that require authentication
@app.before_request
def before_request():
    # List of routes that don't require authentication
    public_routes = ['login', 'static', 'create_admin']
    
    if not current_user.is_authenticated and request.endpoint not in public_routes:
        return redirect(url_for('login', next=request.url))

# Context processor to provide notification data to all templates
@app.context_processor
def inject_notifications():
    if current_user.is_authenticated:
        # Get low stock notifications
        low_stock_products = Product.query.filter(Product.stock <= 10).all()
        low_stock_notifications = []
        for product in low_stock_products:
            low_stock_notifications.append({
                'type': 'low_stock',
                'title': f'Low Stock Alert: {product.title}',
                'message': f'Only {product.stock} units remaining in stock.',
                'time': 'Just now',
                'read': False
            })
        
        # Get expiry notifications
        today = datetime.now().date()
        expiry_threshold = today + timedelta(days=30)  # Products expiring in the next 30 days
        expiring_products = Product.query.filter(
            Product.expiration_date.isnot(None),
            Product.expiration_date <= expiry_threshold
        ).all()
        expiry_notifications = []
        for product in expiring_products:
            # Convert expiration_date to date if it's a datetime
            expiry_date = product.expiration_date.date() if isinstance(product.expiration_date, datetime) else product.expiration_date
            days_until_expiry = (expiry_date - today).days
            expiry_notifications.append({
                'type': 'expiry',
                'title': f'Expiry Alert: {product.title}',
                'message': f'Product expires in {days_until_expiry} days.',
                'time': 'Just now',
                'read': False
            })
        
        # Combine all notifications
        all_notifications = low_stock_notifications + expiry_notifications
        
        # Sort by time (most recent first)
        all_notifications.sort(key=lambda x: x['time'], reverse=True)
        
        # Limit to 5 notifications for the dropdown
        recent_notifications = all_notifications[:5]
        
        return {
            'notifications': recent_notifications,
            'notifications_count': len(all_notifications)
        }
    return {'notifications': [], 'notifications_count': 0}

@app.route('/notifications')
@login_required
def notifications():
    # Get low stock notifications
    low_stock_products = Product.query.filter(Product.stock <= 10).all()
    low_stock_notifications = []
    for product in low_stock_products:
        low_stock_notifications.append({
            'type': 'low_stock',
            'title': f'Low Stock Alert: {product.title}',
            'message': f'Only {product.stock} units remaining in stock.',
            'time': 'Just now',
            'read': False
        })
    
    # Get expiry notifications
    today = datetime.now().date()
    expiry_threshold = today + timedelta(days=30)  # Products expiring in the next 30 days
    expiring_products = Product.query.filter(
        Product.expiration_date.isnot(None),
        Product.expiration_date <= expiry_threshold
    ).all()
    expiry_notifications = []
    for product in expiring_products:
        # Convert expiration_date to date if it's a datetime
        expiry_date = product.expiration_date.date() if isinstance(product.expiration_date, datetime) else product.expiration_date
        days_until_expiry = (expiry_date - today).days
        expiry_notifications.append({
            'type': 'expiry',
            'title': f'Expiry Alert: {product.title}',
            'message': f'Product expires in {days_until_expiry} days.',
            'time': 'Just now',
            'read': False
        })
    
    # Combine all notifications
    all_notifications = low_stock_notifications + expiry_notifications
    
    # Sort by time (most recent first)
    all_notifications.sort(key=lambda x: x['time'], reverse=True)
    
    return render_template('notifications.html', notifications=all_notifications)

@app.route('/mark_notifications_read', methods=['POST'])
@login_required
def mark_notifications_read():
    # In a real application, you would update the read status in the database
    # For this example, we'll just return a success response
    return jsonify({'success': True})

@app.route('/train_rto_model', methods=['POST'])
@login_required
def train_rto_model():
    """Train the RTO prediction model using historical data"""
    try:
        # Get historical orders with known RTO outcomes
        historical_orders = Order.query.filter(Order.rto_risk.isnot(None)).all()
        
        if len(historical_orders) < 10:
            return jsonify({
                'success': False,
                'message': 'Not enough historical data to train the model. Need at least 10 orders.'
            })
        
        # Prepare training data
        training_data = {
            'address_classification': [],
            'status': [],
            'payment_method': [],
            'quantity': [],
            'city': [],
            'state': [],
            'pincode': [],
            'rto_risk': []
        }
        
        for order in historical_orders:
            training_data['address_classification'].append(order.address_classification or 'Unknown')
            training_data['status'].append(order.status)
            training_data['payment_method'].append(order.payment_method)
            training_data['quantity'].append(order.quantity)
            training_data['city'].append(order.city)
            training_data['state'].append(order.state)
            training_data['pincode'].append(order.pincode)
            training_data['rto_risk'].append(order.rto_risk)
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_data)
        
        # Train the model with hyperparameter tuning
        accuracy, precision, recall, f1 = rto_predictor.train(
            training_df, 
            model_type='random_forest',
            tune_hyperparameters=True
        )
        
        # Save the trained model
        rto_predictor.save_model()
        
        # Get feature importance
        feature_importance = rto_predictor.get_feature_importance(top_n=5)
        
        # Create feature importance plot
        plot_path = 'static/images/rto_feature_importance.png'
        rto_predictor.plot_feature_importance(save_path=plot_path)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'feature_importance': feature_importance.to_dict('records'),
            'plot_path': plot_path
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        })

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    with app.app_context():
        try:
            # Create tables if they don't exist
            db.create_all()
            print("Database tables initialized")
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            print("Please check your database connection and configuration")
    
    app.run(debug=True, port=5004)
