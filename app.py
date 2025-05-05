from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


import json
import plotly
import plotly.graph_objs as go

# Models
from models.black_scholes import price_option


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/plot", methods=["POST"])
def route_plot():
    data = request.json
    model = data.get("model", "black_scholes")

    if model == "black_scholes":
        return jsonify(plot_black_scholes(data))
    
    elif model == "binomial":
        # return jsonify(plot_binomial(data))
        return jsonify({"error": "Binomial plot not implemented yet."})
    
    elif model == "monte_carlo":
        # return jsonify(plot_monte_carlo(data))
        return jsonify({"error": "Monte Carlo plot not implemented yet."})

    elif model == "pde":
        # return jsonify(plot_pde(data))
        return jsonify({"error": "PDE plot not implemented yet."})

    else:
        return jsonify({"error": f"Unknown model: {model}"}), 400

def plot_black_scholes(data):
    K = data["K"]
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    option_type = data["option_type"]

    S_range = [i for i in range(50, 151, 5)]
    prices = [price_option(S, K, T, r, sigma, option_type) for S in S_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines+markers', name='Option Price'))
    fig.update_layout(title='Black-Scholes Option Price',
                      xaxis_title='Stock Price (S)',
                      yaxis_title='Option Price')

    price_at_S = price_option(data["S"], K, T, r, sigma, option_type)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return {"plot": graphJSON, "price": price_at_S}

if __name__ == "__main__":
    app.run(debug=True)
