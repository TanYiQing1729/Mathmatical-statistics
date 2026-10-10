import numpy as np
import matplotlib.pyplot as plt
from math import comb


rng = np.random.default_rng(42)
N = 10000


def standard_normal_pdf(x):
    """Standard normal density."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def numerical_standard_normal_cdf(
    left=-8.0,
    right=8.0,
    grid_size=200001,
):
    """Numerically integrate the standard normal PDF to obtain its CDF."""
    grid = np.linspace(left, right, grid_size)
    dx = grid[1] - grid[0]
    pdf = standard_normal_pdf(grid)

    # Cumulative trapezoidal integration, without requiring SciPy.
    cdf = np.empty_like(grid)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * dx)
    cdf /= cdf[-1]

    return grid, cdf


def normal_inverse_transform_newton(
    uniforms,
    cdf_grid,
    cdf_values,
    tol=1e-10,
    max_iter=20,
):
    """Generate normal samples by numerical CDF inversion and Newton iteration.

    The CDF is obtained by numerical integration of the PDF. Newton-Raphson
    solves F(x) - u = 0, using the PDF as the derivative F'(x).
    """
    uniforms = np.asarray(uniforms, dtype=float)
    if np.any(~np.isfinite(uniforms)) or np.any((uniforms <= 0) | (uniforms >= 1)):
        raise ValueError("uniforms must contain finite values in (0, 1)")

    # Interpolation provides a stable starting point for Newton iteration.
    x = np.interp(uniforms, cdf_values, cdf_grid)

    for _ in range(max_iter):
        cdf_x = np.interp(x, cdf_grid, cdf_values)
        pdf_x = standard_normal_pdf(x)
        correction = (cdf_x - uniforms) / pdf_x
        next_x = x - correction
        next_x = np.clip(next_x, cdf_grid[0], cdf_grid[-1])

        if np.max(np.abs(next_x - x)) < tol:
            x = next_x
            break
        x = next_x

    return x


def normal_box_muller(n, rng):
    """Generate standard normal samples with the Box-Muller transform."""
    pair_count = (n + 1) // 2
    u1 = np.clip(rng.random(pair_count), np.finfo(float).tiny, 1.0)
    u2 = rng.random(pair_count)

    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    z1 = radius * np.cos(angle)
    z2 = radius * np.sin(angle)

    return np.concatenate((z1, z2))[:n]


def inverse_transform_discrete(u, support, pmf):
    """离散分布的逆变换法：X = min{k: F(k) >= U}."""
    u = np.asarray(u, dtype=float)
    support = np.asarray(support)
    pmf = np.asarray(pmf, dtype=float)

    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u >= 1)):
        raise ValueError("u must contain finite values in [0, 1)")
    if support.ndim != 1 or pmf.ndim != 1 or support.size != pmf.size:
        raise ValueError(
            "support and pmf must be one-dimensional and have equal length"
        )
    if np.any(pmf < 0) or pmf.sum() > 1.0 + 1e-10:
        raise ValueError("pmf must be nonnegative and have total mass at most one")

    cdf = np.cumsum(pmf)
    if cdf[-1] + 1e-15 < np.max(u):
        raise ValueError("pmf does not cover all sampled uniform values")

    # searchsorted 找到第一个满足 F(k) >= u 的位置
    indices = np.searchsorted(cdf, u, side="left")
    return support[indices]


def poisson_pmf_until(lam, largest_u):
    """递推泊松分布的 PMF，直到累计概率覆盖本批次的 U。"""
    pmf = [np.exp(-lam)]
    total = pmf[0]
    k = 0

    while total < largest_u:
        k += 1
        # P(X=k) = P(X=k-1) * lambda / k
        next_probability = pmf[-1] * lam / k
        pmf.append(next_probability)
        total += next_probability

    # 省略的尾部概率小于本批次均匀随机数所需覆盖的范围
    return np.asarray(pmf, dtype=float)


def negative_binomial_pmf_until(r, p, largest_u):
    """递推负二项分布的 PMF，直到累计概率覆盖本批次的 U。"""
    pmf = [p**r]
    total = pmf[0]
    k = 0

    while total < largest_u:
        k += 1
        # P(X=k) / P(X=k-1) = ((k+r-1)/k) * (1-p)
        next_probability = pmf[-1] * (k + r - 1) / k * (1 - p)
        pmf.append(next_probability)
        total += next_probability

    return np.asarray(pmf, dtype=float)


# ---------- 参数与数据 ----------
# 伯努利分布
p_bern = 0.3
# 理论概率：P(X=1)=p，P(X=0)=1-p
# CDF：F(x) = 0 (x < 0)，1-p (0 <= x < 1)，1 (x >= 1)
bernoulli_u = rng.random(N)
bernoulli_data = (bernoulli_u > 1 - p_bern).astype(int)

# 均匀分布
a = 0
b = 1
# 理论密度：f(x) = 1/(b-a)，a <= x <= b
# CDF：F(x) = 0 (x < a)，(x-a)/(b-a) (a <= x <= b)，1 (x > b)
# 逆变换：F^{-1}(u) = a + (b-a)u
uniform_u = rng.random(N)
uniform_data = a + (b - a) * uniform_u

# 指数分布
lam = 2
# 理论密度：f(x) = lam * exp(-lam*x)，x >= 0
# CDF：F(x) = 1 - exp(-lam*x)，x >= 0
# 逆变换：F^{-1}(u) = -log(1-u)/lam
exponential_u = rng.random(N)
exponential_data = -np.log1p(-exponential_u) / lam

# 二项分布
n = 10
p = 0.4
# 理论概率：P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
# CDF：F(k) = sum_{j=0}^k C(n,j) * p^j * (1-p)^(n-j)
# 先计算 PMF，再根据离散分布的 CDF 使用逆变换法
binomial_u = rng.random(N)
binomial_support = np.arange(n + 1)
binomial_pmf = np.array([
    comb(n, k) * p**k * (1 - p)**(n - k)
    for k in binomial_support
])
binomial_data = inverse_transform_discrete(
    binomial_u,
    binomial_support,
    binomial_pmf,
)

# 负二项分布
# X 表示第 r 次成功之前的失败次数
r = 5      # 需要成功 r 次
p_nb = 0.5 # 每次试验的成功概率
# 理论概率：P(X=k) = C(k+r-1,k) * p_nb^r * (1-p_nb)^k
# CDF：F(k) = sum_{j=0}^k C(j+r-1,j) * p_nb^r * (1-p_nb)^j
negative_binomial_u = rng.random(N)
negative_binomial_pmf = negative_binomial_pmf_until(
    r,
    p_nb,
    largest_u=float(negative_binomial_u.max()),
)
negative_binomial_support = np.arange(negative_binomial_pmf.size)
negative_binomial_data = inverse_transform_discrete(
    negative_binomial_u,
    negative_binomial_support,
    negative_binomial_pmf,
)

# 泊松分布
lam_pois = 5   # 泊松分布的参数 lambda
# 理论概率：P(X=k) = lam_pois^k * exp(-lam_pois) / k!
# CDF：F(k) = sum_{j=0}^k lam_pois^j * exp(-lam_pois) / j!
# 先递推 PMF，再根据离散分布的 CDF 使用逆变换法
poisson_u = rng.random(N)
poisson_pmf_values = poisson_pmf_until(
    lam_pois,
    largest_u=float(poisson_u.max()),
)
poisson_support = np.arange(poisson_pmf_values.size)
poisson_data = inverse_transform_discrete(
    poisson_u,
    poisson_support,
    poisson_pmf_values,
)


# 正态分布：方法 1，数值积分 CDF 后用 Newton 逆变换
mu_normal = 0.0
sigma_normal = 1.0
normal_cdf_grid, normal_cdf_values = numerical_standard_normal_cdf()
normal_inverse_u = rng.random(N)
standard_normal_inverse = normal_inverse_transform_newton(
    normal_inverse_u,
    normal_cdf_grid,
    normal_cdf_values,
)
normal_inverse_data = (
    mu_normal + sigma_normal * standard_normal_inverse
)

# 正态分布：方法 2，Box-Muller 变换
normal_box_muller_standard = normal_box_muller(N, rng)
normal_box_muller_data = (
    mu_normal + sigma_normal * normal_box_muller_standard
)


# ---------- 理论均值与样本均值 ----------
print(f"Bernoulli sample mean:          {bernoulli_data.mean():.4f} (theory {p_bern:.4f})")
print(f"Uniform sample mean:            {uniform_data.mean():.4f} (theory {(a + b) / 2:.4f})")
print(f"Exponential sample mean:        {exponential_data.mean():.4f} (theory {1 / lam:.4f})")
print(f"Binomial sample mean:           {binomial_data.mean():.4f} (theory {n * p:.4f})")
print(
    "Negative binomial sample mean: "
    f"{negative_binomial_data.mean():.4f} "
    f"(theory {r * (1 - p_nb) / p_nb:.4f})"
)
print(f"Poisson sample mean:            {poisson_data.mean():.4f} (theory {lam_pois:.4f})")
print(
    "Normal inverse-transform mean:  "
    f"{normal_inverse_data.mean():.4f} "
    f"(theory {mu_normal:.4f})"
)
print(
    "Normal inverse-transform var:   "
    f"{normal_inverse_data.var():.4f} "
    f"(theory {sigma_normal**2:.4f})"
)
print(
    "Normal Box-Muller mean:          "
    f"{normal_box_muller_data.mean():.4f} "
    f"(theory {mu_normal:.4f})"
)
print(
    "Normal Box-Muller var:           "
    f"{normal_box_muller_data.var():.4f} "
    f"(theory {sigma_normal**2:.4f})"
)


# ---------- 比较两种正态模拟方法 ----------
x_normal = np.linspace(
    mu_normal - 4 * sigma_normal,
    mu_normal + 4 * sigma_normal,
    400,
)
normal_density = (
    np.exp(-0.5 * ((x_normal - mu_normal) / sigma_normal) ** 2)
    / (sigma_normal * np.sqrt(2 * np.pi))
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(
    normal_inverse_data,
    bins=50,
    density=True,
    color="steelblue",
    edgecolor="white",
    alpha=0.7,
    label="Numerical CDF + Newton inverse"
)
axes[0].plot(
    x_normal,
    normal_density,
    "r-",
    linewidth=2,
    label="Theoretical PDF"
)
axes[0].set(
    title="Normal simulation by numerical inverse transform",
    xlabel="Value",
    ylabel="Probability density",
)
axes[0].grid(axis="y", alpha=0.3)
axes[0].legend()

axes[1].hist(
    normal_box_muller_data,
    bins=50,
    density=True,
    color="darkorange",
    edgecolor="white",
    alpha=0.7,
    label="Box-Muller transform"
)
axes[1].plot(
    x_normal,
    normal_density,
    "r-",
    linewidth=2,
    label="Theoretical PDF"
)
axes[1].set(
    title="Normal simulation by Box-Muller",
    xlabel="Value",
    ylabel="Probability density",
)
axes[1].grid(axis="y", alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
