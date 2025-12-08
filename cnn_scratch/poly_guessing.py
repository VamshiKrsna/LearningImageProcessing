import numpy as np

np.random.seed(0)
N = 200
true_a, true_b, true_c = 2.5, -1.5, 3

x = np.random.uniform(-5, 5, size=N)
sigma = 0.5
y = true_a * x**2 + true_b * x + true_c + np.random.normal(0, sigma, size=N)

# design matrix
X_raw = np.vstack([x**2, x, np.ones_like(x)]).T  # shape (N,3)

# --- scale features: mean 0, std 1 for columns 0 and 1; don't scale constant column ---
X = X_raw.copy().astype(float)
for j in [0,1]:
    mu = X[:, j].mean()
    s = X[:, j].std()
    if s == 0:
        s = 1.0
    X[:, j] = (X[:, j] - mu) / s


# closed form (sanity)
theta_hat = np.linalg.pinv(X_raw.T @ X_raw) @ X_raw.T @ y
print("Closed-form theta (original basis):", theta_hat)

# Gradient descent on the SCALED features
theta = np.zeros(3)
eta = 1e-4       
max_iter = 20000
prev_loss = np.inf

for epoch in range(max_iter):
    residual = y - X @ theta
    loss = (residual**2).mean()         
    grad = -2.0 * (X.T @ residual) / N  
    # safety checks
    if not np.isfinite(loss) or np.any(~np.isfinite(grad)):
        print("Numeric problem at epoch", epoch)
        break
    theta = theta - eta * grad

    # monitor occasionally
    if epoch % 2000 == 0:
        print(f"iter {epoch:5d} loss={loss:.6e} | theta={theta}")

mu0 = X_raw[:,0].mean(); s0 = X_raw[:,0].std()
mu1 = X_raw[:,1].mean(); s1 = X_raw[:,1].std()

a_est = theta[0] / s0
b_est = theta[1] / s1
c_est = theta[2] - (theta[0]*mu0)/s0 - (theta[1]*mu1)/s1

print("GD estimated (converted to original basis):", [a_est, b_est, c_est])
print("True:", [true_a, true_b, true_c])

# ridge (closed form) as before (on original basis)
lam = 1e-3
theta_ridge = np.linalg.inv(X_raw.T @ X_raw + N*lam*np.eye(3)) @ X_raw.T @ y
print("Ridge theta (original basis):", theta_ridge)
