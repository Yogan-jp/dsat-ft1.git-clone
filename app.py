from flask import Flask, render_template, request
import joblib
from groq import Groq
import requests
import os
import re

# === Read from deployed environment variables ===
GROQ_API_KEY = os.getenv("groq")  # you named it "groq" in the Render environment
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY (env: groq) is not set.")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set.")

# === Flask App ===
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    q = request.form.get("q")
    return render_template("main.html")

@app.route("/llama", methods=["GET", "POST"])
def llama():
    return render_template("llama.html")

@app.route("/llama_reply", methods=["GET", "POST"])
def llama_reply():
    q = request.form.get("q")
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": q}]
    )
    return render_template("llama_reply.html", r=completion.choices[0].message.content)

@app.route("/deepseek", methods=["GET", "POST"])
def deepseek():
    return render_template("deepseek.html")

@app.route("/deepseek_reply", methods=["GET", "POST"])
def deepseek_reply():
    q = request.form.get("q")
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": q}]
    )
    response = completion.choices[0].message.content
    response = re.sub(r"<think>\s*</think>", "", response).strip()
    return render_template("deepseek_reply.html", r=response)

@app.route("/dbs", methods=["GET", "POST"])
def dbs():
    return render_template("dbs.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    q = float(request.form.get("q"))
    model = joblib.load("dbs.jl")
    pred = model.predict([[q]])
    return render_template("prediction.html", r=pred)

@app.route("/telegram", methods=["GET", "POST"])
def telegram():
    domain_url = 'https://dsat-ft1-git-clone.onrender.com'

    # Delete existing webhook
    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    requests.post(delete_webhook_url, json={"url": domain_url, "drop_pending_updates": True})

    # Set new webhook
    set_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={domain_url}/webhook"
    webhook_response = requests.post(set_webhook_url, json={"url": domain_url, "drop_pending_updates": True})

    status = "The telegram bot is running. Please check @gemini_tt_bot" if webhook_response.status_code == 200 else "Failed to start the telegram bot."
    return render_template("telegram.html", status=status)

@app.route("/stop_telegram", methods=["GET", "POST"])
def stop_telegram():
    domain_url = 'https://dsat-ft1-git-clone.onrender.com'

    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    webhook_response = requests.post(delete_webhook_url, json={"url": domain_url, "drop_pending_updates": True})

    status = "The telegram bot is stopped." if webhook_response.status_code == 200 else "Failed to stop the telegram bot."
    return render_template("telegram.html", status=status)

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    update = request.get_json()
    if "message" in update and "text" in update["message"]:
        chat_id = update["message"]["chat"]["id"]
        query = update["message"]["text"]

        client = Groq(api_key=GROQ_API_KEY)
        completion_ds = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": query}]
        )
        response_message = completion_ds.choices[0].message.content

        send_message_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(send_message_url, json={"chat_id": chat_id, "text": response_message})

@app.route('/sepia', methods=['GET', 'POST'])
def sepia():
    return render_template("sepia_hf.html")
    return 'ok', 200

if __name__ == "__main__":
    app.run()
