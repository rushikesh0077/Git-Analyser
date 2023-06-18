import flask
from flask import Flask, request, render_template
from model import repo_analyser
import os
from dotenv import load_dotenv
import openai
load_dotenv()

app = Flask(__name__)

openai.api_key = os.environ['OPENAI_API_KEY']
path = "codes"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analysis', methods=['POST'])
def analysis():
    if request.method == 'POST':
        print('****************in post************')
        link = request.form['link']
        print(link)
        obj = repo_analyser(str(link))
        justification,repo_link = obj.final_answer()
        print(justification)
        repo_link = repo_link.split('[')[-1][:-1].split('/raw')[0]
        print('link:', repo_link)
        return render_template('index_results.html', justification = justification, link = repo_link)
    else:
        return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)