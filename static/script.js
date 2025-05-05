const modelParamMap = {
    black_scholes: [],
    binomial: ["n_steps"],
    monte_carlo: ["n_steps", "n_simulations"],
    pde: ["x_max", "n_t"]
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

  function handleModeChange() {
    const selectedMode = document.querySelector('input[name="mode"]:checked').value;
    if (selectedMode === 'manual') {
        manualForm.style.display = 'block';
        historicalForm.style.display = 'none';
        mainContainer.classList.add('manual-active');
        clearHistoricalResults();
        updateHistoricalVisibleInputs(); 
    } else { 
      // historical mode
        manualForm.style.display = 'none';
        historicalForm.style.display = 'block';
        mainContainer.classList.remove('manual-active');
        mainContainer.classList.add('historical-active');

        // Clear manual plot/results if switching away
        Plotly.purge("plot");
        document.getElementById("output").innerText = "";
        mainContainer.classList.remove("plot-active"); 
        rerunButton.style.display = 'none';
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
    const payload = {
        S: parseFloat(document.getElementById("S").value),
        K: parseFloat(document.getElementById("K_manual").value),
        T: parseFloat(document.getElementById("T").value),
        r: parseFloat(document.getElementById("r").value),
        sigma: parseFloat(document.getElementById("sigma").value),
        option_type: document.getElementById("option_type_manual").value, 
        model: document.getElementById("model_manual").value, 
        exercise_style: document.getElementById("exercise_style_manual").value
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

    // --- Caching Logic (Manual) ---
    if (deepEqual(payload, lastSuccessfulPayload)) {
        console.log("Manual payload hasn't changed. Using cached result.");
        // Ensure plot-active class and rerun button visibility are correct
        if (lastSuccessfulPayload) mainContainer.classList.add("plot-active");
        if (payload.model === "monte_carlo") rerunButton.style.display = "block";
        else rerunButton.style.display = "none";
        return;
    }
    // --- End Caching Logic ---

    try {
        document.getElementById("output").innerText = "Calculating...";
        Plotly.purge("plot");
        rerunButton.style.display = "none";
        mainContainer.classList.remove("plot-active");

        const res = await fetch("/plot", { // Send to original endpoint
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await res.json();

        if (result.error) {
            document.getElementById("output").innerText = `Error: ${result.error}`;
            lastSuccessfulPayload = null;
            Plotly.purge("plot");
            rerunButton.style.display = "none";
            mainContainer.classList.remove("plot-active");
            return;
        }

        // Plotting logic (assuming /plot returns plot JSON)
        if (result.plot) {
            const plot = JSON.parse(result.plot);
            Plotly.newPlot("plot", plot.data, plot.layout);
            mainContainer.classList.add("plot-active");
        } else {
             Plotly.purge("plot");
             mainContainer.classList.remove("plot-active");
        }

        // Display price
        if (result.price !== undefined) {
            document.getElementById("output").innerText = `Price: $${result.price.toFixed(4)}`;
        } else {
            document.getElementById("output").innerText = result.plot ? "Plot generated." : "No price returned.";
        }

        lastSuccessfulPayload = payload;
        // Show rerun button only for Monte Carlo model
        if (payload.model === "monte_carlo" && result.plot) {
            rerunButton.style.display = "block";
        } else {
            rerunButton.style.display = "none";
        }

    } catch (err) {
        document.getElementById("output").innerText = "Failed to fetch plot.";
        lastSuccessfulPayload = null;
        rerunButton.style.display = "none";
        mainContainer.classList.remove("plot-active");
        console.error(err);
    }
}

// Historical Form Submission
async function handleHistoricalFormSubmit() {
    const payload = {
        ticker: document.getElementById("ticker").value,
        quote_date: document.getElementById("quote_date").value,
        expiry_date: document.getElementById("expiry_date").value,
        K: parseFloat(document.getElementById("K_historical").value), // Use historical ID
        option_type: document.getElementById("option_type_historical").value, // Use historical ID
        model: document.getElementById("model_historical").value, // Use historical ID
        exercise_style: document.getElementById("exercise_style_historical").value // Added
    };

    if (!payload.ticker || !payload.quote_date || !payload.expiry_date || !payload.K) {
        document.getElementById("output").innerText = "Error: Please fill all historical fields.";
        return;
    }
     if (payload.expiry_date <= payload.quote_date) {
        document.getElementById("output").innerText = "Error: Expiry date must be after quote date.";
        return;
    }

     // --- Caching Logic (Historical) ---
     if (deepEqual(payload, lastSuccessfulHistoricalPayload)) {
        console.log("Historical payload hasn't changed. Using cached result.");
        if (document.getElementById('plot').children.length > 0) { // Check if plot div has content
             mainContainer.classList.add("plot-active");
        }
        return;
     }
     // --- End Caching Logic ---

    try {
        document.getElementById("output").innerText = "Fetching historical data and calculating...";
        clearHistoricalResults(); // Clear previous calculated params
        Plotly.purge("plot"); // Clear plot area initially
        mainContainer.classList.remove("plot-active"); // Remove manual plot class

        const res = await fetch("/historical_price", { // Send to NEW endpoint
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await res.json();

        if (result.error) {
            document.getElementById("output").innerText = `Error: ${result.error}`;
            lastSuccessfulHistoricalPayload = null;
            Plotly.purge("plot"); // Ensure plot is cleared on error
            mainContainer.classList.remove("plot-active");
            return;
        }

        // Display calculated parameters
        document.getElementById("hist_S").innerText = result.S?.toFixed(2) ?? 'N/A';
        document.getElementById("hist_sigma").innerText = result.sigma?.toFixed(4) ?? 'N/A';
        document.getElementById("hist_T").innerText = result.T?.toFixed(4) ?? 'N/A';
        document.getElementById("hist_r").innerText = result.r?.toFixed(4) ?? 'N/A';

        // Display price
        document.getElementById("output").innerText = `Historical Price: $${result.price?.toFixed(4) ?? 'N/A'}`;

        // Plotting logic for historical data
        if (result.plot) {
            try {
                const plot = JSON.parse(result.plot);
                Plotly.newPlot("plot", plot.data, plot.layout);
                mainContainer.classList.add("plot-active"); // Add class to adjust layout
            } catch (plotError) {
                console.error("Error parsing or plotting historical plot data:", plotError);
                Plotly.purge("plot"); // Clear plot on error
                mainContainer.classList.remove("plot-active");
            }
        } else {
             Plotly.purge("plot"); // Clear plot if no plot data received
             mainContainer.classList.remove("plot-active");
        }


        lastSuccessfulHistoricalPayload = payload; // Cache successful historical payload

    } catch (err) {
        document.getElementById("output").innerText = "Failed to fetch historical price.";
        lastSuccessfulHistoricalPayload = null;
        Plotly.purge("plot"); // Ensure plot is cleared on fetch error
        mainContainer.classList.remove("plot-active");
        console.error(err);
    }
}

function clearHistoricalResults() {
     document.getElementById("hist_S").innerText = '--';
     document.getElementById("hist_sigma").innerText = '--';
     document.getElementById("hist_T").innerText = '--';
     document.getElementById("hist_r").innerText = '--';
}


// --- Event Listeners ---
document.addEventListener("DOMContentLoaded", () => {
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
          if (document.querySelector('input[name="mode"]:checked').value === 'manual') {
              lastSuccessfulPayload = null; // Clear manual cache
              console.log("Rerunning Monte Carlo simulation...");
              handleManualFormSubmit(); // Resubmit manual form
          }
      });
  }
});