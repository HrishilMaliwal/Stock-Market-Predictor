from flask import render_template, url_for, flash, redirect
from maincode import app, db, bcrypt
from maincode.forms import RegistrationForm, LoginForm
from maincode.models import User
from flask_login import login_user

@app.route('/')
@app.route('/home')
def home():
   return render_template('home.html')

@app.route('/about')
def about():
   return render_template('about.html',title='About Us')

@app.route('/paid_home')
def paid_home():
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