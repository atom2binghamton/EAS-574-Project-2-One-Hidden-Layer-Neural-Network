import numpy as np
from sklearn.datasets import fetch_openml

try:
    import resource
    def get_peak_memory_mb():
        """Return peak memory usage in MB (macOS/Linux)."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 * 1024)
except ImportError:
    import psutil
    def get_peak_memory_mb():
        """Return current memory usage in MB (Windows)."""
        return psutil.Process().memory_info().rss / (1024 * 1024)


def sigmoid(z):
    """Numerically stable sigmoid activation."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def forward(X, W1, b1, W2, b2):
    """
    Forward pass through the network.

    Args:
        X:  Input data, shape (n_samples, 784)
        W1: Hidden weights, shape (784, hidden_size)
        b1: Hidden biases, shape (hidden_size,)
        W2: Output weights, shape (hidden_size, 1)
        b2: Output bias, shape (1,)

    Returns:
        y_hat: Predictions, shape (n_samples, 1)
        cache: Tuple (Z1, A1, Z2) for backprop
    """
    Z1 = X @ W1 + b1            # (n, hidden_size)
    A1 = np.tanh(Z1)             # (n, hidden_size)
    Z2 = A1 @ W2 + b2            # (n, 1)
    y_hat = sigmoid(Z2)          # (n, 1)
    cache = (Z1, A1, Z2)
    return y_hat, cache


def compute_loss(y_true, y_hat):
    """
    Binary cross-entropy loss.

    Args:
        y_true: Labels, shape (n_samples, 1)
        y_hat:  Predictions, shape (n_samples, 1)

    Returns:
        loss: Scalar
    """
    n = y_true.shape[0]
    y_hat_clipped = np.clip(y_hat, 1e-7, 1.0 - 1e-7)
    loss = -np.mean(y_true * np.log(y_hat_clipped) +
                    (1.0 - y_true) * np.log(1.0 - y_hat_clipped))
    return float(loss)


def backward(X, y, y_hat, cache, W2):
    """
    Backward pass to compute gradients.

    Args:
        X, y, y_hat: Data and predictions
        cache: (Z1, A1, Z2) from forward pass
        W2: Output weights (needed for backprop)

    Returns:
        grads: Dict with 'dW1', 'db1', 'dW2', 'db2'
    """
    n = X.shape[0]
    Z1, A1, Z2 = cache

    # Output layer error, shape (n, 1)
    delta2 = y_hat - y

    dW2 = (1.0 / n) * (A1.T @ delta2)          # (hidden_size, 1)
    db2 = (1.0 / n) * np.sum(delta2, axis=0)   # (1,)

    # Hidden layer error
    delta1 = (delta2 @ W2.T) * (1.0 - A1 ** 2)  # (n, hidden_size)

    dW1 = (1.0 / n) * (X.T @ delta1)            # (784, hidden_size)
    db1 = (1.0 / n) * np.sum(delta1, axis=0)    # (hidden_size,)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


def train(X, y, hidden_size=64, learning_rate=0.1, iterations=1000):
    """
    Train the neural network.

    Returns:
        W1, b1, W2, b2: Trained parameters
        losses: List of loss values per iteration
    """
    n_features = X.shape[1]
    rng = np.random.default_rng(42)

    W1 = (rng.standard_normal((n_features, hidden_size)) * 0.01).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    W2 = (rng.standard_normal((hidden_size, 1)) * 0.01).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)

    losses = []

    for _ in range(iterations):
        y_hat, cache = forward(X, W1, b1, W2, b2)
        loss = compute_loss(y, y_hat)
        losses.append(loss)

        grads = backward(X, y, y_hat, cache, W2)

        W1 -= learning_rate * grads['dW1']
        b1 -= learning_rate * grads['db1']
        W2 -= learning_rate * grads['dW2']
        b2 -= learning_rate * grads['db2']

    return W1, b1, W2, b2, losses


def gradient_check(X, y, W1, b1, W2, b2, param_name, i, j, epsilon=1e-7):
    """
    Compare analytical gradient to numerical gradient for one element.

    Args:
        X, y: Data
        W1, b1, W2, b2: Current parameters
        param_name: 'W1', 'b1', 'W2', or 'b2'
        i, j: Indices of the element to check (j=None for biases)
        epsilon: Perturbation size

    Returns:
        analytical: Gradient from backprop
        numerical:  Gradient from finite differences
        difference: |analytical - numerical|
    """
    # Upcast to float64 just for the gradient check because numerical is only coming out to 0
    W1 = W1.astype(np.float64)
    b1 = b1.astype(np.float64)
    W2 = W2.astype(np.float64)
    b2 = b2.astype(np.float64)
    X  = X.astype(np.float64)
    y  = y.astype(np.float64)

    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    param = params[param_name]

    # Analytical gradient calculation
    y_hat, cache = forward(X, W1, b1, W2, b2)
    grads = backward(X, y, y_hat, cache, W2)
    grad_map = {'W1': grads['dW1'], 'b1': grads['db1'],
                'W2': grads['dW2'], 'b2': grads['db2']}
    if j is None:
        analytical = float(grad_map[param_name][i])
    else:
        analytical = float(grad_map[param_name][i, j])

    # Numerical gradient calculation
    original = param[i] if j is None else param[i, j]

    if j is None:
        param[i] = original + epsilon
    else:
        param[i, j] = original + epsilon
    y_hat_plus, _ = forward(X, W1, b1, W2, b2)
    loss_plus = compute_loss(y, y_hat_plus)

    if j is None:
        param[i] = original - epsilon
    else:
        param[i, j] = original - epsilon
    y_hat_minus, _ = forward(X, W1, b1, W2, b2)
    loss_minus = compute_loss(y, y_hat_minus)

    # Restore original value
    if j is None:
        param[i] = original
    else:
        param[i, j] = original

    numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
    difference = abs(analytical - numerical)

    return analytical, numerical, difference

# Main used for testing purposes and json file creation is AI generated
if __name__ == "__main__":
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    LEARNING_RATES = [0.5, 1.0]
    HIDDEN_SIZES   = [64, 128, 256]
    ITERATIONS     = 1000

    # Load MNIST dataset
    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)

    # Split 60,000 train, 10,000 test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # 1 = even, 0 = odd
    y_train = (y_train % 2 == 0).astype(np.float32).reshape(-1, 1)
    y_test  = (y_test  % 2 == 0).astype(np.float32).reshape(-1, 1)

    # Training
    run_data = {hs: {} for hs in HIDDEN_SIZES}

    for hs in HIDDEN_SIZES:
        for lr in LEARNING_RATES:
            print(f"Training hidden_size={hs}, learning_rate={lr}...")
            W1, b1, W2, b2, losses = train(X_train, y_train,
                                            hidden_size=hs,
                                            learning_rate=lr,
                                            iterations=ITERATIONS)
            peak_mem = get_peak_memory_mb()
            y_hat, _ = forward(X_test, W1, b1, W2, b2)
            test_acc = float(((y_hat >= 0.5).astype(np.float32) == y_test).mean())
            y_hat_train, _ = forward(X_train, W1, b1, W2, b2)
            train_acc = float(((y_hat_train >= 0.5).astype(np.float32) == y_train).mean())
            print(f"  final_loss={losses[-1]:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  peak_memory={peak_mem:.1f} MB")

            run_data[hs][lr] = {
                'losses':         losses,
                'train_acc':      train_acc,
                'test_acc':       test_acc,
                'peak_memory_mb': peak_mem,
                'params':         (W1, b1, W2, b2),
            }

    # Select best learning rate for each hidden size
    json_results = {}
    best_curves  = {}
    best_lr_per_hs = {}

    for hs in HIDDEN_SIZES:
        best_lr = min(LEARNING_RATES, key=lambda lr: run_data[hs][lr]['losses'][-1])
        best_lr_per_hs[hs] = best_lr
        d = run_data[hs][best_lr]
        json_results[str(hs)] = {
            "train_loss":     round(float(d['losses'][-1]), 6),
            "train_acc":      round(d['train_acc'], 6),
            "test_acc":       round(d['test_acc'], 6),
            "peak_memory_mb": round(d['peak_memory_mb'], 2),
        }
        best_curves[hs] = d['losses']

    # Pick best size
    best_hs = min(HIDDEN_SIZES, key=lambda hs: json_results[str(hs)]['train_loss'])
    best_lr = best_lr_per_hs[best_hs]
    W1, b1, W2, b2 = run_data[best_hs][best_lr]['params']

    # Gradient checks
    gc_checks = [
        ('W1', 100, 30,  'W1_100_30'),
        ('W1', 500, 50,  'W1_500_50'),
        ('W2', 30,  0,   'W2_30_0'),
        ('b1', 10,  None,'b1_10'),
    ]
    gradient_check_results = {}
    print(f"\nGradient check (hidden_size={best_hs}, lr={best_lr}):")
    print(f"{'Parameter':<14} {'Analytical':>14} {'Numerical':>14} {'Difference':>14}")
    for param_name, i, j, key in gc_checks:
        analytical, numerical, difference = gradient_check(
            X_train, y_train, W1, b1, W2, b2, param_name, i, j)
        gradient_check_results[key] = float(difference)
        idx = f"[{i}]" if j is None else f"[{i},{j}]"
        print(f"{param_name+idx:<14} {analytical:>14.6e} {numerical:>14.6e} {difference:>14.2e}")

    # Save results
    output = {
        "learning_rates_tested": LEARNING_RATES,
        "hidden_sizes_tested":   HIDDEN_SIZES,
        "results":               json_results,
        "gradient_check":        gradient_check_results,
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=4)
    print("\nResults saved to results.json")

    # Plot learning curves
    ref_hs = HIDDEN_SIZES[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for lr in LEARNING_RATES:
        ax1.plot(run_data[ref_hs][lr]['losses'], label=f"lr={lr}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss vs. Iteration (learning rate comparison)")
    ax1.legend()

    for hs, losses in best_curves.items():
        ax2.plot(losses, label=f"hidden_size={hs} (lr={best_lr_per_hs[hs]})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs. Iteration (hidden size comparison)")
    ax2.legend()

    # Save plots to png file
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    print("Plot saved to learning_curves.png")
