from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

# server , library
import os
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use('Agg') #use non-gui backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
#scikit-learn

app=Flask(__name__)
app.secret_key = 'Nisarga@011'
#data base configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/ml_app'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']= False
#initialize SQlALchemy
db = SQLAlchemy(app)
#initialize Bcrypt
bcrypt = Bcrypt(app)
#define a database model-User model
class User(db.Model):
    id = db.Column(db.Integer , primary_key = True)
    name = db.Column(db.String(100), unique = True , nullable = False)
    email = db.Column(db.String(100), unique = True , nullable = False)
    password = db.Column(db.String(100), unique = False , nullable = False)                  

@app.route('/')
def index():
    #return "Flask is working!"
    return render_template('index.html')
@app.route('/register' , methods=['GET','POST'])
def register():
    if request.method=='POST':
        #read the inputs from registration fprm
        name=request.form['name']
        email=request.form['email']
        password=request.form['password']
        #password encryption
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        #create a new user 
        new_user = User(name=name,email=email,password=hashed_password)
        #saving the user details to database table
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration sucessfull !! Please log in.','success')
            return redirect(url_for("login"))
        except:
           flash('Error:name or email already exists.','danger') 
    return render_template('register.html')
        
          
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        #read the email and password from login form
        email=request.form['email']
        password=request.form['password']
        #query the user by email
        user=User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password,password):
            #sucessful login
            session['user_id']=user.id
            session['name']=user.name
            flash('Login Successfull!','success')
          #redirect to dashboard
            return redirect(url_for('dashboard'))
        else:
            #login failed
               flash('Invalid email or password,Please try again.','danger')
    return render_template('login.html')
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.' ,name=name)
        return redirect(url_for('login'))
    name=session['name']
    return render_template('dashboard.html',name=name)

@app.route('/logout', methods = ['POST'])
def logout():
    #remove user_id and name keys
    #from session dictionary
    session.pop('user_id', None)
    session.pop('name', None)
    flash('You have been logged out.' ,'info')
    return redirect(url_for('login'))

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.getcwd()), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower()in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global uploaded_data
    file_uploaded = False
    columns, data = [],[]
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            file.save(file_path)
            file_uploaded = True
            flash('File uploaded successfully!', 'success')

            try:
                uploaded_data = pd.read_csv(file_path)
                columns = uploaded_data.columns.tolist()
                data = uploaded_data.head(10).values.tolist()
                return render_template('upload.html', file_uploaded=file_uploaded,columns=columns,data=data)

            except Exception as e:
                flash(f'Error reading the file: {e}', 'danger')
                return redirect(request.url)
        flash('Invalid file type. Please upload a CSV file.','danger') 
        return redirect(request.url)

    return render_template('upload.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    global uploaded_data
    if uploaded_data is None:
        flash('No dataset uploaded.', 'warning')
        return redirect(url_for('upload'))
    try:
        data = uploaded_data.copy()
        data.columns = data.columns.str.strip()
        # Check target column
        if 'Loan_Status' not in data.columns:
            flash('Dataset must contain Loan_Status as the target column.', 'danger')
            return redirect(url_for('upload'))

        # Convert Loan_Status to binary
        data['Loan_Status'] = data['Loan_Status'].apply(
            lambda x: 1 if str(x).strip().upper() == 'Y' else 0
        )
        
        selected_features = [
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Term_Months',
            'Credit_History',
            'Employment_Type',
            'Dependents',
            'Property_Area'
        ]

        # Reduce dataset to only these features + target
        data = data[selected_features + ['Loan_Status']]
        
        # Encode categorical columns
        from sklearn.preprocessing import LabelEncoder
        for col in selected_features:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))

        # Split into features & target
        X = data[selected_features]
        y = data['Loan_Status']

        # Convert all to numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Save model
        import os, joblib
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(rf_model, "models/model.pkl")

        # Evaluate model
        from sklearn.metrics import accuracy_score, confusion_matrix
        y_pred = rf_model.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)

        # Confusion matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Eligible', 'Eligible'],
            yticklabels=['Not Eligible', 'Eligible']
        )

        plt.title(f"Confusion Matrix (Accuracy: {acc}%)")
        plt.xlabel("Detection")
        plt.ylabel("Actual")

        cm_path = os.path.join('static', 'images', 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

        flash(f"Training Successful! Accuracy: {acc}%", "success")
        return render_template('train.html', accuracy=acc, confusion_matrix=cm_path)

    except Exception as e:
        flash(f"Error during training: {e}", 'danger')
        return redirect(url_for('upload'))


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        inputs = {
            "ApplicantIncome": float(request.form['ApplicantIncome']),
            "CoapplicantIncome": float(request.form['CoapplicantIncome']),
            "LoanAmount": float(request.form['LoanAmount']),
            "Loan_Term_Months": int(request.form['Loan_Term_Months']),
            "Credit_History": int(request.form['Credit_History']),
            "Employment_Type": int(request.form['Employment_Type']),
            "Dependents": int(request.form['Dependents']),
            "Property_Area": int(request.form['Property_Area']),
        }
        input_values = list(inputs.values())
        model = joblib.load("models/model.pkl")
        detection = model.predict([input_values])[0]
        result = "Eligible for Loan" if detection == 1 else "Not Eligible for Loan"
        return render_template('result.html', inputs=inputs, result=result)
    return render_template('detect.html')
                                                      
                                 
                              
              
                                                  
            
                                            
                        
                                                 
                    
        
              


            

                                               
                
            



        
if __name__ == '__main__':
    #ensure tables are created
    with app.app_context():
        db.create_all()
    #run the app
    app.run(debug=False, host='0.0.0.0', port=5000)









    
