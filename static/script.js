const modelParamMap = {
    black_scholes: [],
    binomial: ["n_steps"],
    monte_carlo: ["n_steps", "n_simulations"],
    pde: ["x_max", "n_t"],
    all: ["n_steps", "n_simulations"] // For Binomial and MC components of "All Models"
  };

  let lastSuccessfulPayload = null; // Variable to store the last successful payload
  let lastSuccessfulHistoricalPayload = null; // Cache for historical mode

  // Helper function for deep object comparison
  function deepEqual(obj1, obj2) {
    if (obj1 === obj2) return true;
    if (obj1 == null || typeof obj1 !== "object" || obj2 == null || typeof obj2 !== "object") return false;

    const keys1 = Object.keys(obj1);
    const keys2 = Object.keys(obj2);

    if (keys1.length !== keys2.length) return false;

    for (const key of keys1) {
      if (!keys2.includes(key) || !deepEqual(obj1[key], obj2[key])) return false;
    }

    return true;
  }

  const rerunButton = document.getElementById("rerunButton");
  const mainContainer = document.querySelector(".main-container");
  const manualForm = document.getElementById("optionForm");
  const historicalForm = document.getElementById("historicalForm");
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const historicalModelSelect = document.getElementById("model_historical");
  const historicalExerciseStyleSelect = document.getElementById("exercise_style_historical");

  // Get references to the new display areas
  const standardPlotDisplayArea = document.getElementById("standard-plot-display-area");
  const mcPrimaryAnalyticsDisplayArea = document.getElementById("monte-carlo-primary-analytics-display-area");
  const allModelsDetailsArea = document.getElementById("all-models-details-area");
  const analyticsSummaryContainer = document.getElementById("analytics-summary-container");

  function handleModeChange() {
    const selectedMode = document.querySelector('input[name="mode"]:checked').value;
    // Clear all plots and results when mode changes
    Plotly.purge("plot");
    Plotly.purge("gbm_plot_div");
    Plotly.purge("terminal_prices_histogram_div");
    Plotly.purge("error_bar_chart_div");

    standardPlotDisplayArea.style.display = "none";
    mcPrimaryAnalyticsDisplayArea.style.display = "none";
    allModelsDetailsArea.style.display = "none";
    analyticsSummaryContainer.style.display = "none";

    document.getElementById("gbm_analytics_summary_div").innerHTML = "";
    document.getElementById("pricing_table_div").innerHTML = "";
    document.getElementById("output").innerText = "";
    mainContainer.classList.remove("plot-active");
    rerunButton.style.display = "none";

    if (selectedMode === 'manual') {
        manualForm.style.display = 'block';
        historicalForm.style.display = 'none';
        mainContainer.classList.add('manual-active');
        mainContainer.classList.remove('historical-active');
        clearHistoricalResults();
        updateHistoricalVisibleInputs();
    } else { // historical mode
        manualForm.style.display = 'none';
        historicalForm.style.display = 'block';
        mainContainer.classList.remove('manual-active');
        mainContainer.classList.add('historical-active');
        updateVisibleInputs();
        updateHistoricalVisibleInputs();
    }
  }

  modeRadios.forEach(radio => {
      radio.addEventListener('change', handleModeChange);
  });

  function updateVisibleInputs() {
    const selectedModelEl = document.getElementById("model_manual");
    const exerciseStyleSelectEl = document.getElementById("exercise_style_manual");
    if (!selectedModelEl || !exerciseStyleSelectEl) {
        console.warn("Manual form elements not found for updateVisibleInputs");
        return;
    }

    const selectedModel = selectedModelEl.value;
    const allFields = ["n_steps", "n_simulations", "x_max", "n_t"];
    const americanOption = exerciseStyleSelectEl.querySelector('option[value="american"]');

    allFields.forEach(field => {
        const el = document.getElementById(`${field}_container`);
        if (el) { 
             if (modelParamMap[selectedModel]?.includes(field)) {
                el.style.display = "block";
            } else {
                el.style.display = "none";
            }
        } else {
           console.warn(`Element container not found: ${field}_container`);
        }
    });

    // Hide American option for Black-Scholes/PDE in manual mode
    if (americanOption) {
        if (selectedModel === "black_scholes" || selectedModel === "pde") {
            americanOption.style.display = "none";
            if (exerciseStyleSelectEl.value === "american") {
                exerciseStyleSelectEl.value = "european"; 
            }
        } else {
            americanOption.style.display = "";
        }
    }

    if (selectedModel !== "monte_carlo") {
        rerunButton.style.display = "none";
    }
}

// --- New Function for Historical Form Visibility ---
function updateHistoricalVisibleInputs() {
    const selectedModelEl = document.getElementById("model_historical");
    const exerciseStyleSelectEl = document.getElementById("exercise_style_historical");

    // Check if elements exist before proceeding (important for initial load/mode switching)
    if (!selectedModelEl || !exerciseStyleSelectEl) {
        console.warn("Historical form elements not found for updateHistoricalVisibleInputs");
        return;
    }

    const selectedModel = selectedModelEl.value;
    const americanOption = exerciseStyleSelectEl.querySelector('option[value="american"]');

    if (americanOption) {
        if (selectedModel === "black_scholes" || selectedModel === "pde") {
            americanOption.style.display = "none";
            if (exerciseStyleSelectEl.value === "american") {
                exerciseStyleSelectEl.value = "european";
            }
        } else {
            americanOption.style.display = "";
        }
    }
    // Hide rerun button if historical model is not Monte Carlo
    if (selectedModel !== "monte_carlo") {
        rerunButton.style.display = "none";
    }
    // If it is Monte Carlo, visibility is handled by handleHistoricalFormSubmit
}

  const manualModelSelect = document.getElementById("model_manual");
  if (manualModelSelect) {
      manualModelSelect.addEventListener("change", updateVisibleInputs);
  }
  // Add listener for historical model change
  if (historicalModelSelect) {
      historicalModelSelect.addEventListener("change", updateHistoricalVisibleInputs);
  }

  window.addEventListener("DOMContentLoaded", () => {
      updateVisibleInputs();
      updateHistoricalVisibleInputs(); 
  });

  async function handleManualFormSubmit() {
    const marketPriceInput = document.getElementById("market_price");
    let marketPrice = null;
    if (marketPriceInput && marketPriceInput.value.trim() !== "") {
        const parsedMarketPrice = parseFloat(marketPriceInput.value);
        if (!isNaN(parsedMarketPrice)) {
            marketPrice = parsedMarketPrice;
        }
    }

    const payload = {
        S: parseFloat(document.getElementById("S").value),
        K: parseFloat(document.getElementById("K_manual").value),
        T: parseFloat(document.getElementById("T").value),
        r: parseFloat(document.getElementById("r").value),
        sigma: parseFloat(document.getElementById("sigma").value),
        option_type: document.getElementById("option_type_manual").value, 
        model: document.getElementById("model_manual").value, 
        exercise_style: document.getElementById("exercise_style_manual").value,
        market_price: marketPrice // This can be null if not provided or invalid
    };

    // Add model-specific parameters
    if (document.getElementById("n_steps_container")?.style.display !== "none")
        payload.n_steps = parseInt(document.getElementById("n_steps").value);
    if (document.getElementById("n_simulations_container")?.style.display !== "none")
        payload.n_simulations = parseInt(document.getElementById("n_simulations").value);
    if (document.getElementById("x_max_container")?.style.display !== "none")
        payload.x_max = parseFloat(document.getElementById("x_max").value);
    if (document.getElementById("n_t_container")?.style.display !== "none")
        payload.n_t = parseInt(document.getElementById("n_t").value);

    if (deepEqual(payload, lastSuccessfulPayload)) {
        console.log("Manual payload hasn't changed. Using cached result.");
        if (lastSuccessfulPayload) mainContainer.classList.add("plot-active");
        if (payload.model === "monte_carlo" && (document.getElementById('gbm_plot_div').children.length > 0 || document.getElementById('terminal_prices_histogram_div').children.length > 0) ) {
             rerunButton.style.display = "block";
        }
        return;
    }

    Plotly.purge("plot");
    Plotly.purge("gbm_plot_div");
    Plotly.purge("terminal_prices_histogram_div");
    Plotly.purge("error_bar_chart_div");

    standardPlotDisplayArea.style.display = "none";
    mcPrimaryAnalyticsDisplayArea.style.display = "none";
    allModelsDetailsArea.style.display = "none";
    analyticsSummaryContainer.style.display = "none";

    document.getElementById("gbm_analytics_summary_div").innerHTML = "";
    document.getElementById("pricing_table_div").innerHTML = "";
    mainContainer.classList.remove("plot-active");
    rerunButton.style.display = "none";
    document.getElementById("output").innerText = "Calculating...";

    if (payload.model === "all") {
        try {
            const res = await fetch("/plot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const result = await res.json();

            if (result.error) {
                document.getElementById("output").innerText = `Error: ${result.error}`;
                lastSuccessfulPayload = null;
                return;
            }
            let contentDisplayed = false;
            if (result.combined_plot) {
                const combinedPlotData = JSON.parse(result.combined_plot);
                standardPlotDisplayArea.style.display = "block"; // Make parent visible BEFORE plotting
                Plotly.newPlot("plot", combinedPlotData.data, combinedPlotData.layout); // Simpler call
                requestAnimationFrame(() => Plotly.Plots.resize(document.getElementById('plot')));
                contentDisplayed = true;
            }
            if (result.pricing_table || result.error_bar_chart) {
                allModelsDetailsArea.style.display = "block";
                if (result.pricing_table && result.pricing_table.length > 0) {
                    const tableDiv = document.getElementById("pricing_table_div");
                    let tableHTML = "<table><thead><tr><th>Model</th><th>Price</th></tr></thead><tbody>";
                    result.pricing_table.forEach(row => {
                        tableHTML += `<tr><td>${row.Model}</td><td>${row.Price}</td></tr>`;
                    });
                    tableHTML += "</tbody></table>";
                    tableDiv.innerHTML = tableHTML;
                    contentDisplayed = true;
                } else {
                    document.getElementById("pricing_table_div").innerHTML = "";
                }
                if (result.error_bar_chart) {
                    const errorChartDiv = document.getElementById("error_bar_chart_div");
                    const errorPlotData = JSON.parse(result.error_bar_chart);
                    Plotly.newPlot(errorChartDiv, errorPlotData.data, errorPlotData.layout);
                    requestAnimationFrame(() => Plotly.Plots.resize(errorChartDiv));
                    contentDisplayed = true;
                } else {
                    Plotly.purge("error_bar_chart_div");
                }
            }
            if(contentDisplayed) mainContainer.classList.add("plot-active");
            else mainContainer.classList.remove("plot-active");
            
            // Final resize after layout changes
            requestAnimationFrame(() => {
                if (document.getElementById('plot').children.length > 0) Plotly.Plots.resize(document.getElementById('plot'));
                if (document.getElementById('error_bar_chart_div').children.length > 0) Plotly.Plots.resize(document.getElementById('error_bar_chart_div'));
            });

            document.getElementById("output").innerText = contentDisplayed ? "All models processed." : "No data to display for 'All Models'.";
            lastSuccessfulPayload = payload;
        } catch (err) {
            document.getElementById("output").innerText = "Failed to fetch data for all models.";
            lastSuccessfulPayload = null;
            console.error("Error fetching all models data:", err);
        }
        return;
    }

    try {
        const res = await fetch("/plot", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await res.json();

        if (result.error) {
            document.getElementById("output").innerText = `Error: ${result.error}`;
            lastSuccessfulPayload = null;
            return;
        }

        let contentDisplayed = false;
        let mcPlotsExist = false;

        if (payload.model === "monte_carlo") {
            mcPrimaryAnalyticsDisplayArea.style.display = "block";
            if (result.gbm_simulation_plot) {
                try {
                    const gbmSimPlot = JSON.parse(result.gbm_simulation_plot);
                    Plotly.newPlot("gbm_plot_div", gbmSimPlot.data, gbmSimPlot.layout, {responsive: true});
                    requestAnimationFrame(() => Plotly.Plots.resize(document.getElementById('gbm_plot_div')));
                    mcPlotsExist = true; contentDisplayed = true;
                } catch (e) { console.error("Error rendering GBM simulation plot:", e); Plotly.purge("gbm_plot_div"); }
            } else { Plotly.purge("gbm_plot_div"); }

            if (result.terminal_prices_histogram_plot) {
                try {
                    const termPriceHistPlot = JSON.parse(result.terminal_prices_histogram_plot);
                    Plotly.newPlot("terminal_prices_histogram_div", termPriceHistPlot.data, termPriceHistPlot.layout, {responsive: true});
                    requestAnimationFrame(() => Plotly.Plots.resize(document.getElementById('terminal_prices_histogram_div')));
                    mcPlotsExist = true; contentDisplayed = true;
                } catch (e) { console.error("Error rendering Terminal Prices histogram:", e); Plotly.purge("terminal_prices_histogram_div"); }
            } else { Plotly.purge("terminal_prices_histogram_div"); }
            
            const summaryDiv = document.getElementById("gbm_analytics_summary_div");
            summaryDiv.innerHTML = ""; 
            if (result.gbm_summary_stats) {
                let summaryHTML = "<h4>GBM Simulation Analytics</h4>";
                for (const [key, value] of Object.entries(result.gbm_summary_stats)) {
                    const readableKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    summaryHTML += `<p><strong>${readableKey}:</strong> ${value}</p>`;
                }
                summaryDiv.innerHTML = summaryHTML;
                analyticsSummaryContainer.style.display = "block";
                contentDisplayed = true;
            }
            rerunButton.style.display = mcPlotsExist ? "block" : "none";
        } else { // For Black-Scholes, Binomial, PDE
            if (result.plot) {
                try {
                    const plotData = JSON.parse(result.plot); // Use a different variable name
                    standardPlotDisplayArea.style.display = "block"; // Make parent visible BEFORE plotting
                    Plotly.newPlot("plot", plotData.data, plotData.layout); // Simpler call
                    requestAnimationFrame(() => Plotly.Plots.resize(document.getElementById('plot')));
                    contentDisplayed = true;
                } catch (e) { console.error("Error rendering main plot:", e); Plotly.purge("plot");}
            } else { Plotly.purge("plot"); }
        }

        if(contentDisplayed) mainContainer.classList.add("plot-active");
        else mainContainer.classList.remove("plot-active");

        // Final resize after layout changes
        requestAnimationFrame(() => {
            if (document.getElementById('plot').children.length > 0) Plotly.Plots.resize(document.getElementById('plot'));
            if (document.getElementById('gbm_plot_div').children.length > 0) Plotly.Plots.resize(document.getElementById('gbm_plot_div'));
            if (document.getElementById('terminal_prices_histogram_div').children.length > 0) Plotly.Plots.resize(document.getElementById('terminal_prices_histogram_div'));
        });

        let priceText = "";
        if (result.price !== undefined && result.price !== null) {
            priceText = `Model Price: $${result.price.toFixed(4)}`;
        }
        if (payload.model === "monte_carlo" && result.monte_carlo_price_from_analytics_sim !== undefined && result.monte_carlo_price_from_analytics_sim !== null) {
             // For MC, the primary price IS the sim price.
             priceText = `MC (Sim) Price: $${result.monte_carlo_price_from_analytics_sim.toFixed(4)}`;
        } else if (result.monte_carlo_price_from_analytics_sim !== undefined && result.monte_carlo_price_from_analytics_sim !== null) {
            // This case should ideally not happen if backend is correct, but as a fallback
            if (priceText) priceText += " | ";
            priceText += `MC (Sim) Price: $${result.monte_carlo_price_from_analytics_sim.toFixed(4)}`;
        }
        document.getElementById("output").innerText = priceText || (contentDisplayed ? "Plot(s) generated." : "No price or plot returned.");
        lastSuccessfulPayload = payload;

    } catch (err) {
        document.getElementById("output").innerText = "Failed to fetch data.";
        lastSuccessfulPayload = null;
        console.error(err);
    }
}

async function handleHistoricalFormSubmit() {
    const payload = {
        ticker: document.getElementById("ticker").value,
        quote_date: document.getElementById("quote_date").value,
        expiry_date: document.getElementById("expiry_date").value,
        K: parseFloat(document.getElementById("K_historical").value),
        option_type: document.getElementById("option_type_historical").value,
        model: document.getElementById("model_historical").value,
        exercise_style: document.getElementById("exercise_style_historical").value,
        volatility_source: document.getElementById("volatility_source").value // Added
    };

    if (!payload.ticker || !payload.quote_date || !payload.expiry_date || !payload.K) {
        document.getElementById("output").innerText = "Error: Please fill all historical fields.";
        return;
    }
    if (payload.expiry_date <= payload.quote_date) {
        document.getElementById("output").innerText = "Error: Expiry date must be after quote date.";
        return;
    }
    if (deepEqual(payload, lastSuccessfulHistoricalPayload)) {
        console.log("Historical payload hasn't changed. Using cached result.");
        if (document.getElementById('plot').children.length > 0 || mcPrimaryAnalyticsDisplayArea.style.display === 'block') {
             mainContainer.classList.add("plot-active");
        }
        // Visibility for rerun button on cached result
        if (payload.model === "monte_carlo" &&
            (mcPrimaryAnalyticsDisplayArea.style.display === 'block' ||
             (document.getElementById('gbm_plot_div').children.length > 0 ||
              document.getElementById('terminal_prices_histogram_div').children.length > 0))) {
            rerunButton.style.display = "block";
        } else { 
            rerunButton.style.display = "none";
        }
        return;
    }

    Plotly.purge("plot");
    Plotly.purge("gbm_plot_div");
    Plotly.purge("terminal_prices_histogram_div");
    // Plotly.purge("error_bar_chart_div"); // Not typically used for historical single model

    standardPlotDisplayArea.style.display = "none";
    mcPrimaryAnalyticsDisplayArea.style.display = "none";
    allModelsDetailsArea.style.display = "none"; // Should be hidden for single historical model
    analyticsSummaryContainer.style.display = "none";

    document.getElementById("gbm_analytics_summary_div").innerHTML = "";
    // document.getElementById("pricing_table_div").innerHTML = ""; // Not typically used
    mainContainer.classList.remove("plot-active");
    document.getElementById("output").innerText = "Fetching historical data...";
    rerunButton.style.display = "none"; // Explicitly hide at start of new fetch


    try {
        const res = await fetch("/historical_price", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await res.json();

        if (result.error) {
            document.getElementById("output").innerText = `Error: ${result.error}`;
            lastSuccessfulHistoricalPayload = null;
            return;
        }

        document.getElementById("hist_S").innerText = result.S?.toFixed(2) ?? 'N/A';
        document.getElementById("hist_sigma").innerText = result.sigma?.toFixed(4) ?? 'N/A';
        document.getElementById("hist_T").innerText = result.T?.toFixed(4) ?? 'N/A';
        document.getElementById("hist_r").innerText = result.r?.toFixed(4) ?? 'N/A';
        document.getElementById("hist_vol_source").innerText = result.volatility_source ? result.volatility_source.charAt(0).toUpperCase() + result.volatility_source.slice(1) : 'Constant';

        let contentDisplayed = false;
        let mcPlotsExistHist = false; // Specific flag for historical MC plots

        if (payload.model === "monte_carlo") {
            mcPrimaryAnalyticsDisplayArea.style.display = "block";
            if (result.gbm_simulation_plot) {
                try {
                    const gbmSimPlot = JSON.parse(result.gbm_simulation_plot);
                    Plotly.newPlot("gbm_plot_div", gbmSimPlot.data, gbmSimPlot.layout, {responsive: true});
                    requestAnimationFrame(() => Plotly.Plots.resize(document.getElementById('gbm_plot_div')));
                    if (document.getElementById('gbm_plot_div').children.length > 0) mcPlotsExistHist = true;
                    contentDisplayed = true;
                } catch (e) { console.error("Error rendering GBM sim (hist):", e); Plotly.purge("gbm_plot_div");}
            } else { Plotly.purge("gbm_plot_div"); }

            if (result.terminal_prices_histogram_plot) {
                try {
                    const termPriceHistPlot = JSON.parse(result.terminal_prices_histogram_plot);
                    Plotly.newPlot("terminal_prices_histogram_div", termPriceHistPlot.data, termPriceHistPlot.layout, {responsive: true});
                    requestAnimationFrame(() => Plotly.Plots.resize(document.getElementById('terminal_prices_histogram_div')));
                    if (document.getElementById('terminal_prices_histogram_div').children.length > 0) mcPlotsExistHist = true;
                    contentDisplayed = true;
                } catch (e) { console.error("Error rendering Term Price Hist (hist):", e); Plotly.purge("terminal_prices_histogram_div");}
            } else { Plotly.purge("terminal_prices_histogram_div"); }

            const summaryDivHist = document.getElementById("gbm_analytics_summary_div");
            summaryDivHist.innerHTML = "";
            if (result.gbm_summary_stats) {
                let summaryHTMLHist = "<h4>GBM Simulation Analytics</h4>";
                for (const [key, value] of Object.entries(result.gbm_summary_stats)) {
                    const readableKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    summaryHTMLHist += `<p><strong>${readableKey}:</strong> ${value}</p>`;
                }
                summaryDivHist.innerHTML = summaryHTMLHist;
                analyticsSummaryContainer.style.display = "block";
                contentDisplayed = true; // This ensures contentDisplayed is true if summary stats are shown
            }
            rerunButton.style.display = mcPlotsExistHist ? "block" : "none";
        } else { // BS, Binomial, PDE for historical
            if (result.plot) {
                try {
                    const plotData = JSON.parse(result.plot);
                    standardPlotDisplayArea.style.display = "block"; // Make parent visible BEFORE plotting
                    Plotly.newPlot("plot", plotData.data, plotData.layout); // Simpler call
                    requestAnimationFrame(() => Plotly.Plots.resize(document.getElementById('plot')));
                    contentDisplayed = true;
                } catch (plotError) { console.error("Error plotting (hist):", plotError); Plotly.purge("plot");}
            } else { Plotly.purge("plot"); }
            rerunButton.style.display = "none"; // Ensure hidden for non-MC historical models
        }
        
        if(contentDisplayed) mainContainer.classList.add("plot-active");
        else mainContainer.classList.remove("plot-active");

        // Final resize after layout changes
        requestAnimationFrame(() => {
            if (document.getElementById('plot').children.length > 0) Plotly.Plots.resize(document.getElementById('plot'));
            if (document.getElementById('gbm_plot_div').children.length > 0) Plotly.Plots.resize(document.getElementById('gbm_plot_div'));
            if (document.getElementById('terminal_prices_histogram_div').children.length > 0) Plotly.Plots.resize(document.getElementById('terminal_prices_histogram_div'));
        });

        let priceTextHist = "";
        if (result.price !== undefined && result.price !== null) {
            priceTextHist = `Model Price: $${result.price.toFixed(4)}`;
        }
        // MC price for historical is already in result.price if MC is selected
        document.getElementById("output").innerText = priceTextHist || (contentDisplayed ? "Plot(s) generated." : "No price or plot returned.");
        lastSuccessfulHistoricalPayload = payload;

    } catch (err) {
        document.getElementById("output").innerText = "Failed to fetch historical price or analytics.";
        lastSuccessfulHistoricalPayload = null;
        rerunButton.style.display = "none"; // Hide on error
        console.error("Error in handleHistoricalFormSubmit:", err);
    }
}

function clearHistoricalResults() {
     document.getElementById("hist_S").innerText = '--';
     document.getElementById("hist_sigma").innerText = '--';
     document.getElementById("hist_T").innerText = '--';
     document.getElementById("hist_r").innerText = '--';
     document.getElementById("hist_vol_source").innerText = '--'; // Clear vol source display
}


// --- Event Listeners ---
document.addEventListener("DOMContentLoaded", () => {
  // --- Set Date Defaults and Constraints ---
  const today = new Date();
  const todayString = today.toISOString().split('T')[0]; // Format as YYYY-MM-DD
  const yesterdayString = new Date(today.setDate(today.getDate() - 1)).toISOString().split('T')[0]; // Yesterday's date
  const quoteDateInput = document.getElementById("quote_date");
  if (quoteDateInput) {
      quoteDateInput.max = todayString; // Set max date to today
      quoteDateInput.value = yesterdayString; // Set default value to today
  }

  const expiryDateInput = document.getElementById("expiry_date");
  if (expiryDateInput) {
      const nextYear = new Date(today);
      nextYear.setFullYear(today.getFullYear() + 1);
      const nextYearString = nextYear.toISOString().split('T')[0];
      expiryDateInput.value = nextYearString; // Set default expiry to one year from today
  }
  // --- End Date Setup ---

  // Initial setup
  handleModeChange(); // Set initial visibility based on default checked radio

  // Listener for manual model change
  const manualModelSelect = document.getElementById("model_manual");
  if (manualModelSelect) {
      manualModelSelect.addEventListener("change", updateVisibleInputs);
  } else {
      console.error("Manual model select not found!");
  }

  // Add listener for historical model change
  if (historicalModelSelect) {
      historicalModelSelect.addEventListener("change", updateHistoricalVisibleInputs);
  } else {
      console.error("Historical model select not found!");
  }

  // Manual form submission listener
  if (manualForm) {
      manualForm.addEventListener("submit", (e) => {
          e.preventDefault();
          handleManualFormSubmit();
      });
  } else {
       console.error("Manual form not found!");
  }

  // Historical form submission listener
  if (historicalForm) {
      historicalForm.addEventListener("submit", (e) => {
          e.preventDefault();
          handleHistoricalFormSubmit();
      });
  } else {
       console.error("Historical form not found!");
  }

  // Rerun button click listener (only relevant for manual Monte Carlo)
  if (rerunButton) {
      rerunButton.addEventListener("click", () => {
          const currentMode = document.querySelector('input[name="mode"]:checked').value;
          if (currentMode === 'manual') {
              const selectedModel = document.getElementById("model_manual").value;
              if (selectedModel === "monte_carlo") {
                  lastSuccessfulPayload = null; 
                  console.log("Rerunning Monte Carlo simulation (manual)...");
                  handleManualFormSubmit(); 
              }
          } 
          // Rerun for historical Monte Carlo
          else if (currentMode === 'historical') {
             const selectedModel = document.getElementById("model_historical").value;
             if (selectedModel === "monte_carlo") {
                 lastSuccessfulHistoricalPayload = null;
                 console.log("Rerunning Monte Carlo simulation (historical)...");
                 handleHistoricalFormSubmit();
             }
          }
      });
  }
});