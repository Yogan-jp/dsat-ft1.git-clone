from flask import Flask, render_template, request
import joblib
from groq import Groq
import os
import re  # Added for tag removal

# Set Groq API key
os.environ['GROQ_API_KEY'] = os.getenv("groq")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    q = request.form.get("q")
    # db
    return render_template("main.html")

@app.route("/llama", methods=["GET", "POST"])
def llama():
    return render_template("llama.html")

@app.route("/llama_reply", methods=["GET", "POST"])
def llama_reply():
    q = request.form.get("q")
    # Load model
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return render_template("llama_reply.html", r=completion.choices[0].message.content)

@app.route("/deepseek", methods=["GET", "POST"])
def deepseek():
    return render_template("deepseek.html")

@app.route("/deepseek_reply", methods=["GET", "POST"])
def deepseek_reply():
    q = request.form.get("q")
    # Load model
    client = Groq()
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    # Get and clean the response
    response = completion.choices[0].message.content
    response = re.sub(r"<think>\s*</think>", "", response).strip()
    
    return render_template("deepseek_reply.html", r=response)

@app.route("/dbs", methods=["GET", "POST"])
def dbs():
    return render_template("dbs.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    q = float(request.form.get("q"))
    # Load model
    model = joblib.load("dbs.jl")
    # Make prediction
    pred = model.predict([[q]])
    return render_template("prediction.html", r=pred)

if __name__ == "__main__":
    app.run()
