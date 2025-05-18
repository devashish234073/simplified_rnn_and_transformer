function rnn() {
  let W_in = randomBetween(-0.1, 0.1);
  let W_hid = randomBetween(-0.1, 0.1);
  let W_out = randomBetween(-0.1, 0.1);
  let h = 0;

  function predictNext(input) {
    h = Math.tanh(W_in * input + W_hid * h);
    return W_out * h;
  }

  function trainOnInputs(inputs, targets) {
    let itrCount = 100;
    let stepSize = 0.01;
    for (let j = 0; j < itrCount; j++) {
      h = 0;
      for (let i = 0; i < inputs.length; i++) {
        const x = inputs[i];
        const yA = targets[i];
        const y = predictNext(x);
        const difference = yA - y;

        // Calculate how to adjust the weightss
        const adjustAmount = -2 * difference * h;
        W_out -= stepSize * adjustAmount;

        const adjustH = W_out * (1 - h ** 2) * (-2 * difference);

        // Update the weights
        W_in -= stepSize * adjustH * inputs[i];
        W_hid -= stepSize * adjustH * h;

        //log(JSON.stringify({h, adjustAmount, adjustH, W_in, W_hid, W_out}));

        if (j % 20 === 0) {
          log(`Iteration ${j}, Input: ${inputs[i]}, Target: ${targets[i]}, Predicted: ${y}, Difference: ${difference}`);
        }
      }
    }
  }

  function train(sequence) {
    const inputs = sequence.slice(0, -1);    // All except last [0.1, 0.2,...0.7]
    const targets = sequence.slice(1);       // All except first [0.2, 0.3,...0.8]

    log("Training...");
    trainOnInputs(inputs, targets);

    // Test the trained predictor
    log("\nFinal Predictions:");
    inputs.forEach((num, i) => {
      // Reset h for clean prediction
      h = 0;
      const prediction = predictNext(num);
      log(
        `Input: ${num.toFixed(1)} | Predicted: ${prediction.toFixed(2)} | Actual: ${targets[i].toFixed(1)}`
      );
    });

    // Predict beyond training data
    log("\nFuture Prediction:");
    const lastInput = sequence[sequence.length - 1]; // Last input in the sequence
    h = 0;
    const futurePred = predictNext(lastInput);
    log(`After ${lastInput.toFixed(1)} → Predicted: ${futurePred.toFixed(2)}`);
    const predictionFunction = function (input) {
      h = 0; // Reset h for clean prediction
      return predictNext(input);
    };
    return predictionFunction;
  }
  return {train};
}

function randomBetween(min, max) {
  return Math.random() * (max - min) + min;
}

function appendMessageToList(message) {
  const li = _("li");
  li.textContent = message;
  logUl.appendChild(li);
}

function log(message) {
  if (logUl) {
    appendMessageToList(message);
  } else {
    console.log(message);
  }
}

function $(selector) {
  return document.querySelector(selector);
}

function _(tag) {
  return document.createElement(tag);
}