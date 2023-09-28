import json
import matplotlib.pyplot as plt


def plot_exp_res():
    with open("reports/method.json", "r", encoding="utf-8") as f:
        method = json.load(f)

    # Recall and F2
    # plt.figure(figsize=(15, 10))
    # plt.subplot(2, 1, 1)
    # plt.xlabel("Top k")
    # plt.ylabel("Recall")
    # for k, v in method.items():
    #     plt.plot(v["recall"], label=k, marker="o",
    #              linestyle="--", markersize=4.5)
    # plt.legend()
    # plt.xticks(range(5), [1, 2, 5, 8, 10])

    # plt.subplot(2, 1, 2)
    # plt.xlabel("Top k")
    # plt.ylabel("F2")
    # for k, v in method.items():
    #     plt.plot(v["f2"], label=k, marker="o", linestyle="--", markersize=4.5)
    # plt.legend()
    # plt.xticks(range(5), [1, 2, 5, 8, 10])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.xlabel("Top k")
    plt.ylabel("Recall")
    for k, v in method.items():
        plt.plot(v["recall"], label=k, marker="o",
                 linestyle="--", markersize=4.5)
    plt.legend()
    plt.xticks(range(5), [1, 2, 5, 8, 10])

    plt.subplot(1, 2, 2)
    plt.xlabel("Top k")
    plt.ylabel("F2")
    for k, v in method.items():
        plt.plot(v["f2"], label=k, marker="o", linestyle="--", markersize=4.5)
    plt.legend()
    plt.xticks(range(5), [1, 2, 5, 8, 10])

    plt.show()
    # Save
    # plt.savefig("reports/method.png")


if __name__ == "__main__":
    plot_exp_res()
