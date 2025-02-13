from flask import Flask, request, render_template
import requests

app = Flask(__name__)

AZURE_FUNCTION_URL = "https://articlesreco.azurewebsites.net/api/product_get"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = request.form["user_id"]
        response = requests.get(f"{AZURE_FUNCTION_URL}?user_id={user_id}")
        if response.status_code == 200:
            recommendations = response.json()
            return render_template("index.html", recommendations=recommendations)
        else:
            return render_template("index.html", error="Failed to fetch recommendations.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)