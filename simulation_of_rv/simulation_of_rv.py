import numpy as np
import matplotlib.pyplot as plt
from math import comb, factorial

rng = np.random.default_rng(42)
N = 10000


# ---------- 参数与数据 ----------
# 伯努利分布
p_bern = 0.3
bernoulli_data = rng.binomial(n=1, p=p_bern, size=N)

# 均匀分布
a = 0
b = 1
uniform_data = rng.uniform(a, b, size=N)

# 指数分布
lam = 2
exponential_data = rng.exponential(scale=1 / lam, size=N)

# 二项分布
n = 10
p = 0.4
binomial_data = rng.binomial(n=n, p=p, size=N)

# 负二项分布
r = 5      # 需要成功 r 次
p_nb = 0.5 # 每次试验的成功概率
negative_binomial_data = rng.negative_binomial(n=r, p=p_nb, size=N)

# 泊松分布
lam_pois = 5   # 泊松分布的参数 lambda
poisson_data = rng.poisson(lam=lam_pois, size=N)

# 正态分布：使用 NumPy 的正态随机数生成器，得到
# Y~N(mu_normal, sigma_normal^2)。
mu_normal = 0
sigma_normal = 1
normal_data = rng.normal(loc=mu_normal, scale=sigma_normal, size=N)

# ---------- 绘图 ----------
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
axes = axes.ravel()   # 展平成一维数组，索引 0~8

# 1. 伯努利分布
axes[0].hist(
    bernoulli_data,
    bins=np.arange(-0.5, 2.5, 1),
    weights=np.ones(N) / N,
    color="crimson",
    edgecolor="white",
    alpha=0.65,
    label="Sample histogram",
)

# 理论概率：P(X=1)=p，P(X=0)=1-p
k_bern = np.array([0, 1])
pmf_bern = np.array([1 - p_bern, p_bern])

axes[0].plot(
    k_bern, pmf_bern, "r-o",
    linewidth=2,
    markersize=5,
    label="Theoretical PMF",
)

axes[0].set(
    title=f"Bernoulli: p = {p_bern}",
    xlabel="Value",
    ylabel="Probability",
    xlim=(-0.5, 1.5),
)
axes[0].set_xticks([0, 1])

# 2. 均匀分布
axes[1].hist(
    uniform_data,
    bins=np.linspace(a, b, 31),
    density=True,
    color="steelblue",
    edgecolor="white",
    alpha=0.65,
    label="Sample histogram",
)

# 理论密度：f(x) = 1/(b-a)，a <= x <= b
x = np.linspace(a, b, 400)
y = np.ones_like(x) / (b - a)
axes[1].plot(x, y, "r-", linewidth=2, label="Theoretical PDF")

axes[1].set(
    title="Uniform: U(0,1)",
    xlabel="Value",
    ylabel="Probability density",
    xlim=(a, b),
)

# 3. 指数分布
axes[2].hist(
    exponential_data,
    bins=40,
    density=True,
    color="darkorange",
    edgecolor="white",
    alpha=0.65,
    label="Sample histogram",
)

# 理论密度：f(x) = lam * exp(-lam*x)，x >= 0
x = np.linspace(0, exponential_data.max(), 400)
y = lam * np.exp(-lam * x)
axes[2].plot(x, y, "r-", linewidth=2, label="Theoretical PDF")

axes[2].set(
    title="Exponential: lambda = 2",
    xlabel="Value",
    ylabel="Probability density",
    xlim=(0, exponential_data.max()),
)

# 4. 二项分布
axes[3].hist(
    binomial_data,
    bins=np.arange(-0.5, n + 1.5, 1),
    weights=np.ones(N) / N,
    color="seagreen",
    edgecolor="white",
    alpha=0.65,
    label="Sample histogram",
)

# 理论概率：P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
k = np.arange(n + 1)
pmf = np.array([
    comb(n, j) * p**j * (1 - p)**(n - j)
    for j in range(n + 1)
])

axes[3].plot(
    k, pmf, "r-o",
    linewidth=2,
    markersize=5,
    label="Theoretical PMF",
)

axes[3].set(
    title="Binomial: n = 10, p = 0.4",
    xlabel="Number of successes",
    ylabel="Probability",
    xlim=(-0.5, n + 0.5),
)
axes[3].set_xticks(k)

# 5. 负二项分布
axes[4].hist(
    negative_binomial_data,
    bins=np.arange(-0.5, negative_binomial_data.max() + 1.5, 1),
    weights=np.ones(N) / N,
    color="purple",
    edgecolor="white",
    alpha=0.65,
    label="Sample histogram",
)

# 理论概率：P(X=k) = C(k+r-1, k) * p^r * (1-p)^k
k_nb = np.arange(negative_binomial_data.max() + 1)
pmf_nb = np.array([
    comb(j + r - 1, j) * p_nb**r * (1 - p_nb)**j
    for j in k_nb
])

axes[4].plot(
    k_nb, pmf_nb, "r-o",
    linewidth=2,
    markersize=5,
    label="Theoretical PMF",
)

axes[4].set(
    title="Negative Binomial: r = 5, p = 0.5",
    xlabel="Number of failures before r successes",
    ylabel="Probability",
    xlim=(-0.5, negative_binomial_data.max() + 0.5),
)
# 失败次数范围较大时避免 x 轴刻度太密
axes[4].set_xticks(np.arange(0, negative_binomial_data.max() + 1, 5))

# 6. 泊松分布
axes[5].hist(
    poisson_data,
    bins=np.arange(-0.5, poisson_data.max() + 1.5, 1),
    weights=np.ones(N) / N,
    color="goldenrod",
    edgecolor="white",
    alpha=0.65,
    label="Sample histogram",
)

# 理论概率：P(X=k) = lam^k * e^(-lam) / k!
k_pois = np.arange(poisson_data.max() + 1)
pmf_pois = np.array([
    lam_pois**j * np.exp(-lam_pois) / factorial(j)
    for j in k_pois
])

axes[5].plot(
    k_pois, pmf_pois, "r-o",
    linewidth=2,
    markersize=5,
    label="Theoretical PMF",
)

axes[5].set(
    title="Poisson: lambda = 5",
    xlabel="Number of events",
    ylabel="Probability",
    xlim=(-0.5, poisson_data.max() + 0.5),
)
axes[5].set_xticks(k_pois[::2])   # 每 2 个取一个刻度，避免太密

# 7. 正态分布（通过逆变换法模拟）
axes[6].hist(
    normal_data,
    bins=50,
    density=True,
    color="slateblue",
    edgecolor="white",
    alpha=0.65,
    label="Sample histogram",
)

# 理论密度：f(y) = 1/(sigma*sqrt(2*pi)) * exp(-((y-mu)/sigma)^2/2)
x_normal = np.linspace(
    mu_normal - 4 * sigma_normal,
    mu_normal + 4 * sigma_normal,
    400,
)
y_normal = (
    np.exp(-0.5 * ((x_normal - mu_normal) / sigma_normal) ** 2)
    / (sigma_normal * np.sqrt(2 * np.pi))
)
axes[6].plot(x_normal, y_normal, "r-", linewidth=2, label="Theoretical PDF")
axes[6].set(
    title=f"Normal: N({mu_normal}, {sigma_normal}^2)",
    xlabel="Value",
    ylabel="Probability density",
    xlim=(mu_normal - 4 * sigma_normal, mu_normal + 4 * sigma_normal),
)

# 暂时没有第 8、9 种分布，隐藏空白子图。
axes[7].axis("off")
axes[8].axis("off")

for ax in axes[:7]:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()
