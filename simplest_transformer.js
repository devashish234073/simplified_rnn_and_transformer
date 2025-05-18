function transformer() {
    let queryWeight = randomBetween(-0.1, 0.1);
    let keyWeight = randomBetween(-0.1, 0.1);
    let valueWeight = randomBetween(-0.1, 0.1);
    let outputWeight = randomBetween(-0.1, 0.1);

    function softmax(scores) {
        const maxScore = Math.max(...scores); // prevent large exponentials
        const expScores = scores.map(score => Math.exp(score - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        return expScores.map(score => score / sumExp);
    }

    function getAttentionWeights(inputs) {
        const scores = [];
        for (let i = 0; i < inputs.length; i++) {
            scores.push(inputs[i] * keyWeight);
        }
        return softmax(scores); // convert to attention weights
    }

    function predictNext(inputs) {
        const attention = getAttentionWeights(inputs);
        let weightedSum = 0;
        for (let i = 0; i < inputs.length; i++) {
            weightedSum += inputs[i] * valueWeight * attention[i];
        }
        return weightedSum * outputWeight;
    }

    function trainOnInputs(inputs, targets) {
        let learningRate = 0.01;
        let itrCount = 100;
        for (let j = 0; j < itrCount; j++) {
            let totalError = 0;
            for (let i = 0; i < inputs.length; i++) {
                let x = inputs[i];
                let yA = targets[i];
                const y = predictNext(x);
                const error = y - yA;

                const attention = getAttentionWeights(x);
                const weightedInputSum = x.reduce(
                    (sum, input, i) => sum + input * attention[i],
                    0
                );

                // Update weights to reduce prediction error
                outputWeight -= learningRate * error * weightedInputSum;
                valueWeight -= learningRate * error * outputWeight * weightedInputSum;

                // Simplified update for query and key weights
                const sharedGradient = error * outputWeight * valueWeight;
                queryWeight -= learningRate * sharedGradient * 0.1;
                keyWeight -= learningRate * sharedGradient * 0.1;
                totalError += error * error;
            }
            if (j % 10 === 0) {
                console.log(`iteration ${j}, Avg Error: ${(totalError / inputs.length).toFixed(4)}`);
            }
        }
    }

    function train(sequence) {
        const inputs = [];
        const targets = [];

        for (let i = 0; i < sequence.length - 3; i++) {
            inputs.push([
                sequence[i],
                sequence[i + 1],
                sequence[i + 2]
            ]);
            targets.push(sequence[i + 3]);
        }

        log("Training...");
        trainOnInputs(inputs, targets);

        // Test the trained predictor
        log("\nFinal Predictions:");
        inputs.forEach((numArray, i) => {
            // Reset h for clean prediction
            const prediction = predictNext(numArray);
            log(
                `Input: ${numArray} | Predicted: ${prediction.toFixed(2)} | Actual: ${targets[i].toFixed(1)}`
            );
        });

        // Predict beyond training data
        log("\nFuture Prediction:");
        const lastInputArr = inputs[inputs.length - 1]; // Last input in the sequence
        const futurePred = predictNext(lastInputArr);
        log(`After ${lastInputArr} → Predicted: ${futurePred.toFixed(2)}`);
        const predictionFunction = function (input) {
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