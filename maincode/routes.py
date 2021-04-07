from flask import render_template, url_for, flash, redirect, request
from maincode import app, db, bcrypt
from maincode.forms import RegistrationForm, LoginForm
from maincode.models import User
from flask_login import login_user
from model_files.model_funct import predict_share,show_anal

stk=''

@app.route('/',methods=[ 'POST','GET'])
@app.route('/home',methods=['POST','GET'])
def home():
    global stk
    if request.method == "POST":
        stk = request.form["stock"]
        show_anal(stk)
        return redirect(url_for("analysis"))
    else:
        return render_template('home.html')

@app.route("/predict")
def predict():
    # return f"<h1>{stk}</h1>"
    return render_template('predict.html')

@app.route("/analysis")
def analysis():
    # return f"<h1>{stk}</h1>"
    return render_template('analysis.html')

@app.route('/about')
def about():
   return render_template('about.html',title='About Us')


@app.route('/paid_home',methods=['POST','GET'])
def paid_home():
    global stk
    if request.method == "POST":
        stk = request.form["stock"]
        predict_share(stk)
        return redirect(url_for("predict"))
    else:
        return render_template('paid_home.html')
   

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, number=form.number.data,password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data}! You are now able to login.', 'success')

        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('paid_home'))
        else:    
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)