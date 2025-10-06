from flask import Flask, render_template, request, redirect, session
import pymysql
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 🔴 ADD THIS LINE BEFORE importing pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from collections import defaultdict
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


app = Flask(__name__)
app.secret_key = "supersecretkey"

model = None
X_train = X_test = y_train = y_test = None
column_names = []
logreg = svm = rf = knn = dt = xgb = None

# DB connection
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='1234',
    database="lung_cancer_db", 
    cursorclass=pymysql.cursors.DictCursor
)

@app.route('/')
def home(): return redirect('/login')

@app.route('/sign', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, password))
            connection.commit()
        return redirect('/login')
    return render_template('sign.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
            user = cursor.fetchone()
            if user:
                session['email'] = email
                session['user_id'] = user['id']
                return redirect('/home')
            else:
                return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/home')
def homepage():
    if 'email' not in session:
        return redirect('/login')
    return render_template('home.html')

@app.route('/fileup', methods=['GET', 'POST'])
def fileup():
    if 'email' not in session:
        return redirect('/login')

    global X_train, X_test, y_train, y_test, model
    global logreg, svm, rf, knn, dt, xgb
    global column_names

    table_html = ''
    accuracy = None
    message = None

    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file:
                message = "Please upload a file."
                return render_template('fileup.html', message=message, table_html=table_html, accuracy=accuracy)

            df = pd.read_csv(file)
            column_names = list(df.columns[:-1])
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            table_html = df.to_html(classes='table table-bordered', header="true", index=False)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize models
            logreg = LogisticRegression(max_iter=1000)
            svm = SVC(probability=True)
            rf = RandomForestClassifier()
            knn = KNeighborsClassifier()
            dt = DecisionTreeClassifier()
            xgb = XGBClassifier(verbosity=0)

            # Train individual models
            for clf in [logreg, svm, rf, knn, dt, xgb]:
                clf.fit(X_train, y_train)

            # Train VotingClassifier
            model = VotingClassifier(estimators=[
                ('lr', logreg), ('svm', svm), ('rf', rf),
                ('knn', knn), ('xgb', xgb), ('dt', dt)
            ], voting='soft')
            model.fit(X_train, y_train)

            # Accuracy scores
            accs = {
                'LogisticRegression': accuracy_score(y_test, logreg.predict(X_test)) * 100,
                'SVM': accuracy_score(y_test, svm.predict(X_test)) * 100,
                'RandomForest': accuracy_score(y_test, rf.predict(X_test)) * 100,
                'KNN': accuracy_score(y_test, knn.predict(X_test)) * 100,
                'DecisionTree': accuracy_score(y_test, dt.predict(X_test)) * 100,
                'XGBoost': accuracy_score(y_test, xgb.predict(X_test)) * 100,
                'VotingClassifier': accuracy_score(y_test, model.predict(X_test)) * 100,
            }

            accuracy = accs['VotingClassifier']
            session['trained'] = True  # ✅ Only now enable buttons on home

            # Save chart
            os.makedirs('static/graphs', exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.bar(accs.keys(), accs.values(), color='skyblue')
            plt.xticks(rotation=45)
            plt.ylabel('Accuracy (%)')
            plt.title('Model Accuracy Comparison')
            plt.tight_layout()
            plt.savefig('static/graphs/acc_cmp.png')
            plt.close()

            message = "Model trained successfully and chart saved!"

        except Exception as e:
            message = f"Error: {e}"

    return render_template('fileup.html', message=message, table_html=table_html, accuracy=accuracy)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global column_names, X_train, y_train
    error = None

    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        try:
            # 🔧 Step 1: Collect input row
            row = []
            for col in column_names:
                val = request.form.get(col)
                value = float(val) if val and val.strip() != '' else 0.0
                row.append(value)

            # 🔧 Step 2: Create DataFrame with column names
            input_df = pd.DataFrame([row], columns=column_names)

            # 🔧 Step 3: Auto-calculate actual result (risk logic)
            try:
                age = float(request.form.get("AGE", 0))
                smoking = int(request.form.get("SMOKING", 0))
                alcohol_consuming = int(request.form.get("ALCOHOL_CONSUMING", 0))
                anxiety = int(request.form.get("ANXIETY", 0))
                chest_pain = int(request.form.get("CHEST_PAIN", 0))
                chronic_disease = int(request.form.get("CHRONIC_DISEASE", 0))
                yellow_fingers = int(request.form.get("YELLOW_FINGERS", 0))
                allergy = int(request.form.get("ALLERGY", 0))
                wheezing = int(request.form.get("WHEEZING", 0))
                shortness_of_breath = int(request.form.get("SHORTNESS_OF_BREATH", 0))
                swallowing_difficulty = int(request.form.get("SWALLOWING_DIFFICULTY", 0))
                
                risk_score = 0
                if age >= 55: risk_score += 1
                if smoking: risk_score += 1
                if alcohol_consuming: risk_score += 1
                if anxiety: risk_score += 1
                if chest_pain: risk_score += 1
                if chronic_disease: risk_score += 1
                if yellow_fingers: risk_score += 1
                if allergy: risk_score += 1
                if wheezing: risk_score += 1
                if shortness_of_breath: risk_score += 1
                if swallowing_difficulty: risk_score += 1
                        

                actual = 1 if risk_score >= 5 else 0
            except Exception:
                actual = 0

            # 🔧 Step 4: Train models
            logreg = LogisticRegression(solver='liblinear')
            svm = SVC(probability=True)
            rf = RandomForestClassifier()
            knn = KNeighborsClassifier()
            dt = DecisionTreeClassifier()
            xgb = XGBClassifier(verbosity=0)

            for model in [logreg, svm, rf, knn, dt, xgb]:
                model.fit(X_train, y_train)

            voting_clf = VotingClassifier(
                estimators=[('lr', logreg), ('svm', svm), ('rf', rf),
                            ('knn', knn), ('xgb', xgb), ('dt', dt)],
                voting='soft'
            )
            voting_clf.fit(X_train, y_train)
            session['trained'] = True
            # 🔧 Step 5: Make predictions using input_df
            preds = {
                'LogisticRegression': int(logreg.predict(input_df)[0]),
                'SVM': int(svm.predict(input_df)[0]),
                'RandomForest': int(rf.predict(input_df)[0]),
                'KNN': int(knn.predict(input_df)[0]),
                'DecisionTree': int(dt.predict(input_df)[0]),
                'XGBoost': int(xgb.predict(input_df)[0]),
                'VotingClassifier': int(voting_clf.predict(input_df)[0])
            }

            # 🔧 Step 6: Store into database
            db = pymysql.connect(host="localhost", 
                                 user="root", password="1234",
                                 database="lung_cancer_db",  cursorclass=pymysql.cursors.DictCursor)
            cursor = db.cursor()
            user_id = session.get('user_id', 1)
            for model_name, pred in preds.items():
                cursor.execute("""
                    INSERT INTO manual_predictions (user_id, model_name, input_data, prediction_result, actual_result)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, model_name, str(row), pred, actual))
            db.commit()
            cursor.close()
            db.close()

            
            # Final prediction message  preds['VotingClassifier'],actual
            risk_text = "High Risk" if actual == 1 else "Low Risk"
            return render_template('result.html', risk_text=risk_text)

        except Exception as e:
            error = f"Prediction failed: {e}"

    # Default GET or on error
    return render_template('predict.html', column_names=column_names, error=error)

@app.route('/performance')
def performance():
    if 'email' not in session:
        return redirect('/login')

    # ✅ Step 1: Fetch ALL VotingClassifier manual predictions (not just current user)
    db = pymysql.connect(host="localhost", user="root", password="1234", database="lung_cancer_db",  cursorclass=pymysql.cursors.DictCursor)
    cursor = db.cursor()

    cursor.execute("""
        SELECT prediction_result, actual_result 
        FROM manual_predictions 
        WHERE model_name = 'VotingClassifier'
    """)
    data = cursor.fetchall()
    cursor.close() 
    db.close()

    if not data:
        return "No predictions found for VotingClassifier."

    # ✅ Step 2: Prepare true & predicted values
    y_pred = [int(row['prediction_result']) for row in data]
    y_true = [int(row['actual_result']) for row in data]

    # ✅ Step 3: Calculate performance metrics
    accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
    precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
    recall = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
    f1 = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)

    # ✅ Step 4: Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")

    os.makedirs("static/graphs", exist_ok=True)
    chart_path = "static/graphs/performance_conf_matrix.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("performance.html",
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1=f1,
                           chart_path=chart_path,
                           username="All Users")

@app.route('/for', methods=['GET', 'POST'])
def forgot_password():
    message = None
    if request.method == 'POST':
        email = request.form['email']
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()
            if user:
                session['reset_email'] = email
                return redirect('/reset-password')
            else:
                message = "Email not found."
    return render_template('for.html', message=message)

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_email' not in session:
        return redirect('/for')
    message = None
    if request.method == 'POST':
        new_password = request.form['password']
        with connection.cursor() as cursor:
            cursor.execute("UPDATE users SET password=%s WHERE email=%s", (new_password, session['reset_email']))
            connection.commit()
        session.pop('reset_email', None)
        return redirect('/login')
    return render_template('reset.html')

@app.route('/my-predictions')
def my_predictions():
    if 'email' not in session:
        return redirect('/login')
    db = pymysql.connect(
        host="localhost",
        user="root",
        password="1234",
        database="lung_cancer_db", 
        cursorclass=pymysql.cursors.DictCursor
    )
    cursor = db.cursor()
    cursor.execute(
        "SELECT m.id, u.email, m.model_name, m.input_data, m.prediction_result, m.actual_result, m.created_at"
       " FROM manual_predictions m"
       " JOIN users u ON m.user_id = u.id"
       " WHERE m.model_name= 'VotingClassifier'"
        "ORDER BY m.created_at DESC"
        )
    predictions = cursor.fetchall()
    cursor.close()
    db.close() 
    return render_template('my_predictions.html', predictions=predictions)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)
