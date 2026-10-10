import math
import numpy as np
import matplotlib.pyplot as plt

# Gamma distribution parameters
alpha = 3.0       # shape
theta = 2.0       # scale

# Exponential proposal parameter
lam = 0.25        # rate
proposal_scale = 1 / lam

# Calculate c
delta = 1 / theta - lam
x_mode = (alpha - 1) / delta

c = (
    x_mode ** (alpha - 1)
    * math.exp(-delta * x_mode)
    / (math.gamma(alpha) * theta ** alpha * lam)
)

print(f"x_mode = {x_mode:.4f}")
print(f"c = {c:.4f}")
print(f"理论接受率 = {1 / c:.4f}")

# Target Gamma density
def gamma_density(x):
    x = np.asarray(x)
    return np.where(
        x >= 0,
        x ** (alpha - 1)
        * np.exp(-x / theta)
        / (math.gamma(alpha) * theta ** alpha),
        0.0
    )

# Acceptance-Rejection sampling
def gamma_accept_reject(n, seed=1234):
    rng = np.random.default_rng(seed)

    samples = []
    total_proposals = 0
    total_accepted = 0

    while len(samples) < n:
        batch_size = max(1000, n - len(samples))

        # Generate proposal samples from Exp(rate=0.25)
        y = rng.exponential(scale=proposal_scale, size=batch_size)

        # Generate U(0,1)
        u = rng.random(batch_size)

        # Calculate f(y) / (c * g(y))
        accept_prob = (
            y ** (alpha - 1)
            * np.exp(-delta * y)
            / (
                math.gamma(alpha)
                * theta ** alpha
                * lam
                * c
            )
        )

        accept = u <= accept_prob

        samples.extend(y[accept].tolist())
        total_proposals += batch_size
        total_accepted += np.sum(accept)

    samples = np.asarray(samples[:n])
    acceptance_rate = total_accepted / total_proposals

    return samples, acceptance_rate


# Generate 10,000 Gamma samples
samples, acceptance_rate = gamma_accept_reject(10000)

print(f"实际接受率 = {acceptance_rate:.4f}")
print(f"样本均值 = {samples.mean():.4f}")
print(f"样本方差 = {samples.var():.4f}")

# Theoretical values:
# E[X] = alpha * theta = 6
# Var(X) = alpha * theta^2 = 12
print("理论均值 = 6")
print("理论方差 = 12")

# Plot the simulated samples and theoretical densities
x = np.linspace(0, 30, 500)

gamma_curve = gamma_density(x)
proposal_curve = lam * np.exp(-lam * x)

plt.figure(figsize=(9, 6))

plt.hist(
    samples,
    bins=60,
    density=True,
    alpha=0.6,
    color="skyblue",
    label="Simulated samples"
)

# Target Gamma density
plt.plot(
    x,
    gamma_curve,
    color="red",
    linewidth=2,
    label=r"$\operatorname{Gamma}(3,2)$ target density"
)

# Proposal Exp(0.25) density
plt.plot(
    x,
    proposal_curve,
    color="green",
    linestyle="--",
    linewidth=2,
    label=r"$\operatorname{Exp}(0.25)$ proposal density"
)

# Envelope c * g(x)
plt.plot(
    x,
    c * proposal_curve,
    color="gray",
    linestyle=":",
    linewidth=2,
    label=r"$c g(x)$ envelope"
)

plt.xlabel("x")
plt.ylabel("Density")
plt.title("Acceptance-Rejection Sampling for Gamma(3, 2)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()