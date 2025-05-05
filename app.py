from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/price", methods=["POST"])
def price():
    data = request.json
    return jsonify({"price": 5.72})

if __name__ == "__main__":
    app.run(debug=True)
