const modelParamMap = {
    black_scholes: [],
    binomial: ["n_steps"],
    monte_carlo: ["n_steps", "n_simulations"],
    pde: ["x_max", "n_t"]
  };

  let lastSuccessfulPayload = null; // Variable to store the last successful payload

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

  const rerunButton = document.getElementById("rerunButton"); // Get the button
  const mainContainer = document.querySelector(".main-container"); // Get the main container

  function updateVisibleInputs() {
      const selectedModel = document.getElementById("model").value;
      const allFields = ["n_steps", "n_simulations", "x_max", "n_t"];
      const exerciseStyleSelect = document.getElementById("exercise_style");
      const americanOption = exerciseStyleSelect.querySelector('option[value="american"]');

      allFields.forEach(field => {
        const el = document.getElementById(`${field}_container`);
        if (modelParamMap[selectedModel].includes(field)) {
          el.style.display = "block";
        } else {
          el.style.display = "none";
        }
      });

      // Hide American option for Black-Scholes
      if (selectedModel === "black_scholes" || selectedModel === "pde") {
        americanOption.style.display = "none";
        if (exerciseStyleSelect.value === "american") {
          exerciseStyleSelect.value = "european";
        }
      } else {
        americanOption.style.display = "";
      }

      // Hide rerun button if model changes
      if (selectedModel !== "monte_carlo") {
          rerunButton.style.display = "none";
      }
    }

    document.getElementById("model").addEventListener("change", updateVisibleInputs);
    window.addEventListener("DOMContentLoaded", updateVisibleInputs);

    // Function to handle the form submission logic
    async function handleFormSubmit() {
      const payload = {
          S: parseFloat(document.getElementById("S").value),
          K: parseFloat(document.getElementById("K").value),
          T: parseFloat(document.getElementById("T").value),
          r: parseFloat(document.getElementById("r").value),
          sigma: parseFloat(document.getElementById("sigma").value),
          option_type: document.getElementById("option_type").value,
          model: document.getElementById("model").value,
          exercise_style: document.getElementById("exercise_style").value
        };

      // Add model-specific parameters only if they are visible
      if (document.getElementById("n_steps_container").style.display !== "none")
        payload.n_steps = parseInt(document.getElementById("n_steps").value);

      if (document.getElementById("n_simulations_container").style.display !== "none")
        payload.n_simulations = parseInt(document.getElementById("n_simulations").value);

      if (document.getElementById("x_max_container").style.display !== "none")
        payload.x_max = parseFloat(document.getElementById("x_max").value);

      if (document.getElementById("n_t_container").style.display !== "none")
        payload.n_t = parseInt(document.getElementById("n_t").value);

      // --- Caching Logic ---
      if (deepEqual(payload, lastSuccessfulPayload)) {
        console.log("Payload hasn't changed. Using cached result.");
        // Ensure rerun button visibility is correct even when cached
        if (payload.model === "monte_carlo") {
            rerunButton.style.display = "block";
        } else {
            rerunButton.style.display = "none";
        }
        return; // Exit without fetching
      }
      // --- End Caching Logic ---

      try {
        document.getElementById("output").innerText = "Calculating...";
        Plotly.purge("plot");
        rerunButton.style.display = "none"; // Hide rerun button during calculation
        mainContainer.classList.remove("plot-active");

        const res = await fetch("/plot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const result = await res.json();

        if (result.error) {
          document.getElementById("output").innerText = `Error: ${result.error}`;
          lastSuccessfulPayload = null;
          Plotly.purge("plot");
          rerunButton.style.display = "none"; // Hide on error
          mainContainer.classList.remove("plot-active");
          return;
        }

        const plot = JSON.parse(result.plot);
        Plotly.newPlot("plot", plot.data, plot.layout);
        mainContainer.classList.add("plot-active");

        if (result.price !== undefined) {
          document.getElementById("output").innerText = `Price: $${result.price.toFixed(4)}`;
        } else {
           document.getElementById("output").innerText = "Plot generated.";
        }

        lastSuccessfulPayload = payload; // Cache the successful payload

        // Show rerun button ONLY if the model was Monte Carlo
        if (payload.model === "monte_carlo") {
          rerunButton.style.display = "block";
        } else {
          rerunButton.style.display = "none";
        }

      } catch (err) {
        document.getElementById("output").innerText = "Failed to fetch plot.";
        lastSuccessfulPayload = null;
        rerunButton.style.display = "none"; // Hide on fetch error
        mainContainer.classList.remove("plot-active");
        console.error(err);
      }
    }

    // Original form submit listener
    document.getElementById("optionForm").addEventListener("submit", (e) => {
      e.preventDefault();
      handleFormSubmit(); // Call the submission logic function
    });

    // Rerun button click listener
    rerunButton.addEventListener("click", () => {
      lastSuccessfulPayload = null; // Clear cache to force refetch
      console.log("Rerunning Monte Carlo simulation...");
      handleFormSubmit(); // Call the submission logic function again
    });