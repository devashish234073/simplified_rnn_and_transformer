function vectoredTransformer() {
    // Config (adjustable)
    const dModel = 64;       // Embedding dimension
    const nHeads = 4;        // Number of attention heads
    const seqLength = 3;     // Input sequence length (for positional encoding)
    const vocabSize = 10;    // Example vocabulary size (0-9)

    // Initialize weights
    const embedding = randomMatrix(vocabSize, dModel);
    const posEncoding = getPositionalEncoding(seqLength, dModel);
    const Wq = randomMatrix(dModel, dModel);
    const Wk = randomMatrix(dModel, dModel);
    const Wv = randomMatrix(dModel, dModel);
    const Wo = randomMatrix(dModel, dModel);
    const dense = randomMatrix(dModel, vocabSize);

    // --- Transformer Operations ---
    function scaledDotProductAttention(Q, K, V) {
        const scores = Q.map(q =>
            K.map(k => dotProduct(q, k) / Math.sqrt(dModel))
        );
        const weights = softmax(scores);
        return weights.map(row =>
            V.reduce((sum, v, j) =>
                sum.map((s, i) => s + row[j] * v[i]),
                Array(V[0].length).fill(0)
            )
        );
    }

    function multiHeadAttention(x) {
        const headSize = dModel / nHeads;
        const outputs = [];
        for (let h = 0; h < nHeads; h++) {
            const Q = x.map(vec => matMul(vec, Wq).slice(h * headSize, (h + 1) * headSize));
            const K = x.map(vec => matMul(vec, Wk).slice(h * headSize, (h + 1) * headSize));
            const V = x.map(vec => matMul(vec, Wv).slice(h * headSize, (h + 1) * headSize));
            outputs.push(scaledDotProductAttention(Q, K, V));
        }
        return x.map((_, i) =>
            outputs.reduce((concat, head) => concat.concat(head[i]), [])
        );
    }

    function layerNorm(x) {
        const mean = x.reduce((a, b) => a + b, 0) / x.length;
        const std = Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length);
        return x.map(val => (val - mean) / (std + 1e-6));
    }

    // --- Core Functions (Same as Your Original) ---
    function predictNext(inputs) {
        // Embed inputs and add positional encoding
        const embedded = inputs.map((val, pos) =>
            embedding[val].map((e, i) => e + posEncoding[pos][i])
        );

        // Attention + residual
        const attnOutput = multiHeadAttention(embedded);
        const residual = embedded.map((vec, i) =>
            vec.map((v, j) => v + attnOutput[i][j])
        );
        const norm = layerNorm(residual[residual.length - 1]); // Last token

        // Project to vocabulary
        const logits = matMul(norm, dense);
        return softmax(logits);
    }

    function trainOnInputs(inputs, targets) {
        const learningRate = 0.01;
        for (let i = 0; i < inputs.length; i++) {
            const x = inputs[i];
            const yTrue = targets[i];
            const yPred = predictNext(x);

            // Simplified training: Cross-entropy gradient
            const error = yPred.map((p, j) => p - (j === yTrue ? 1 : 0));

            // Backprop through attention (simplified)
            // (In practice, use automatic differentiation or full backprop)
            // ...
        }
    }

    function train(sequence) {
        const inputs = [];
        const targets = [];

        // Prepare sliding window (as in your original code)
        for (let i = 0; i < sequence.length - 3; i++) {
            inputs.push([sequence[i], sequence[i + 1], sequence[i + 2]]);
            targets.push(sequence[i + 3]);
        }

        log("Training...");
        trainOnInputs(inputs, targets);

        // Return prediction function (same as yours)
        return function predict(input) {
            return predictNext(input);
        };
    }

    // --- Helper Functions (Same as Your Original) ---
    function randomMatrix(rows, cols) {
        return Array(rows).fill().map(() =>
            Array(cols).fill().map(() => randomBetween(-0.1, 0.1))
        );
    }

    function getPositionalEncoding(seqLength, dModel) {
        const posEnc = Array(seqLength).fill().map(() => Array(dModel).fill(0));
        for (let pos = 0; pos < seqLength; pos++) {
            for (let i = 0; i < dModel; i++) {
                const angle = pos / Math.pow(10000, 2 * i / dModel);
                posEnc[pos][i] = (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
            }
        }
        return posEnc;
    }

    function dotProduct(a, b) {
        return a.reduce((sum, v, i) => sum + v * b[i], 0);
    }

    function matMul(a, b) {
        return b[0].map((_, j) => a.reduce((sum, _, i) => sum + a[i] * b[i][j], 0));
    }

    function softmax(arr) {
        const exps = arr.map(x => Math.exp(x - Math.max(...arr)));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    // Return public methods (same as yours)
    return { train };
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