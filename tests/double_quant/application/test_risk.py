import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class DataPreparation:
    def download(self):
        ...

    def grouped_by_risk(self):
        ...

class TestDistribution:
    def test_var_es(self):
        # ==========================================
        # 1. 设置绘图风格 (学术论文风格)
        # ==========================================
        sns.set_style("white")  # 简洁白底
        plt.rcParams["font.family"] = "serif"  # 使用衬线字体 (如 Times New Roman)
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # ==========================================
        # 2. 数据生成 (模拟损失分布)
        # ==========================================
        # 假设这是一个损失分布 (Loss Distribution)，均值为0，标准差为1
        # X轴正方向代表损失金额
        mu, sigma = 0, 1
        x = np.linspace(-3, 4.5, 1000)
        y = norm.pdf(x, mu, sigma)

        # ==========================================
        # 3. 计算 VaR 和 ES (置信水平 95%)
        # ==========================================
        confidence_level = 0.95
        alpha = 1 - confidence_level  # 尾部概率 0.05

        # 计算 VaR (95% 分位数)
        var_value = norm.ppf(confidence_level, mu, sigma)

        # 计算 ES (Expected Shortfall) - 尾部期望
        # 对于正态分布，ES = mu + sigma * (pdf(VaR) / (1-confidence_level))
        es_value = mu + sigma * (norm.pdf(var_value) / alpha)

        # ==========================================
        # 4. 绘图逻辑
        # ==========================================
        _, ax = plt.subplots(figsize=(8, 5))

        # A. 绘制概率密度曲线
        ax.plot(x, y, color="#2c3e50", lw=2, label="Loss Distribution PDF")

        # B. 绘制尾部阴影 (VaR右侧)
        x_tail = np.linspace(var_value, 4.5, 100)
        y_tail = norm.pdf(x_tail, mu, sigma)
        ax.fill_between(
            x_tail, 0, y_tail, color="#e74c3c", alpha=0.3, label="Tail Risk Area"
        )

        # C. 绘制 VaR 线 (阈值)
        ax.axvline(var_value, color="#e74c3c", linestyle="--", lw=2)
        ax.text(
            var_value,
            max(y) * 0.6,
            f" VaR (95%)\n Threshold: {var_value:.2f}",
            color="#c0392b",
            ha="left",
            va="center",
            fontweight="bold",
        )

        # D. 绘制 ES 线 (尾部均值)
        ax.axvline(es_value, color="#2980b9", linestyle="-", lw=2.5)
        ax.text(
            es_value,
            max(y) * 0.4,
            f" ES (95%)\n Average Loss: {es_value:.2f}",
            color="#2980b9",
            ha="left",
            va="center",
            fontweight="bold",
        )

        # E. 装饰图表
        ax.set_title(
            "Comparison of VaR and ES (Expected Shortfall)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Loss Amount ($)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_xlim(-3, 4.5)
        ax.set_ylim(0, max(y) * 1.1)

        # 去掉上方和右方的边框，更美观
        sns.despine()

        # 标注 Alpha 区域
        ax.annotate(
            r"$\alpha = 5\%$",
            xy=(3, 0.02),
            xytext=(3.5, 0.1),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=12,
        )
        plt.tight_layout()
        plt.show()
