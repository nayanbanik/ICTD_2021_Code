from flask import Flask, request, url_for, jsonify
from flask import render_template
import ptorsum

app=Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""
    if request.method == 'POST':
        data = request.form['nsbody']
        data = ptorsum.cleaning_text(data)
#	print(data)
        msg = ptorsum.eval_one(data)
        print(msg)
        return render_template('Home.html', dummy=msg)
    else:
        return render_template('Home.html')

@app.route('/Features.html')
def features():
    return render_template('Features.html')

@app.route('/Contact.html')
def contact():
    return render_template('Contact.html')