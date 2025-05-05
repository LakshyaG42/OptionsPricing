from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

# Models
from models.black_scholes import price_option


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/price_black_scholes", methods=["POST"])
def price():
    data = request.json
    S = data["S"]
    K = data["K"]
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    option_type = data.get("option_type", "call")
    price = price_option(S, K, T, r, sigma, option_type)
    return jsonify({"price": price})

if __name__ == "__main__":
    app.run(debug=True)
