from flask import Flask, render_template, url_for, request, redirect
from model.neural_network import predict_protein
import webbrowser
import threading
from tech.file_runner import resource_path

app = Flask(__name__)
PORT = 5000
URL = f"http://127.0.0.1:{PORT}"

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/process_form', methods=['POST'])
def process_form():
    if request.method == "POST":
        sequence = request.form.get('sequence', '').upper().strip(" ")

        result = predict_protein(sequence, 3)
        normalized = [(str(name), int(score*100)) for name, score in result]
        sum = 0
        for name, score in normalized:
            sum+= score 
        text = ", ".join(f"{str(name)}: {score}%" for name, score in normalized)
        if text:
            return redirect(url_for('result', content = text))
        return render_template('home.html')
    return render_template('home.html')

@app.route('/result', methods = ['GET'])
def result():
    msg = request.args.get('content', None)
    return render_template('result.html', result=msg)

@app.route('/help', methods = ["GET"])
def help():
    return render_template("help.html")

def open_browser():
    threading.Timer(1.0, lambda: webbrowser.open_new(URL)).start()

if __name__ == "__main__":
    open_browser()
    app.run(port=PORT, debug=False)