import matplotlib.pyplot as plt

# x축 k 값
ks = [4, 8, 12, 16, 20]

# 표의 수치 정리
data = {
    "Forward": {
        "EXACT@k": {
            "greedy": 0.709,
            "naive": [0.76, 0.779, 0.781, 0.792, 0.798],
            "reflection": [0.792, 0.811, 0.828, 0.824, 0.838],
        },
        "Majority vote": {
            "greedy": 0.709,
            "naive": [0.711, 0.723, 0.73, 0.731, 0.735],
            "reflection": [0.744, 0.761, 0.767, 0.762, 0.772],
        },
    },
    "Retro": {
        "EXACT@k": {
            "greedy": 0.341,
            "naive": [0.406, 0.456, 0.475, 0.495, 0.496],
            "reflection": [0.483, 0.527, 0.55, 0.561, 0.577],
        },
        "Majority vote": {
            "greedy": 0.341,
            "naive": [0.336, 0.347, 0.356, 0.364, 0.362],
            "reflection": [0.397, 0.424, 0.43, 0.438, 0.429],
        },
    },
    "Reagent": {
        "EXACT@k": {
            "greedy": 0.478,
            "naive": [0.585, 0.629, 0.652, 0.68, 0.683],
            "reflection": [0.641, 0.691, 0.712, 0.737, 0.748],
        },
        "Majority vote": {
            "greedy": 0.478,
            "naive": [0.491, 0.493, 0.497, 0.49, 0.502],
            "reflection": [0.529, 0.56, 0.57, 0.56, 0.568],
        },
    },
}

tasks = ["Forward", "Retro", "Reagent"]
metrics = ["EXACT@k", "Majority vote"]

fig, axes = plt.subplots(3, 2, figsize=(10/1.5, 12/1.5), sharex=True)
# set 

for row, task in enumerate(tasks):
    for col, metric in enumerate(metrics):
        ax = axes[row, col]
        vals = data[task][metric]

        # Naive / Reflection曲선
        ax.plot(ks, vals["naive"], marker="o", label="Naive sampling")
        ax.plot(ks, vals["reflection"], marker="s", label="Reflective sampling")

        # Greedy baseline (점선 horizontal line)
        ax.axhline(y=vals["greedy"], linestyle="--", label="Greedy (k=1)")

        # 축/제목 설정
        if row == 2:  # 맨 아래줄만 x축 라벨
            ax.set_xlabel("k")
        if col == 0:
            ax.set_ylabel(f"{task}\n{metric}")
        else:
            ax.set_ylabel(metric)

        ax.grid(True)
        ax.set_xticks(ks)

        # 첫 번째 row의 오른쪽 subplot에만 legend 표시(겹치지 않게)
        if row == 0 and col == 1:
            ax.legend(loc="lower right")

fig.tight_layout()

plt.savefig("Analysis/plots/reflection_strategy_comparison.png")
