from flask import render_template, url_for, flash, redirect
from maincode import app
from maincode.forms import RegistrationForm, LoginForm
from maincode.models import User

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
        flash(f'Account created for {form.username.data}!', 'success')

        return redirect(url_for('paid_home'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'sharvilkotian99@paisamilega.com' and form.password.data == 'paisa':
            flash('You have been logged in!', 'success')
            return redirect(url_for('paid_home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)