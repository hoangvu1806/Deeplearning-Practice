import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm


def get_wandb_data(project_name="california_housing_regression"):
    try:
        api = wandb.Api()
    except:
        from wandb import Api

        api = Api()

    entity = (
        wandb.api.default_entity
        if hasattr(wandb, "api") and hasattr(wandb.api, "default_entity")
        else None
    )

    runs = api.runs(f"{entity}/{project_name}")
    runs = [run for run in runs if run.name != "summary"]

    results = []

    print("Đang tải dữ liệu từ Weights & Biases...")
    for run in tqdm(runs):
        run_name = run.name
        parts = run_name.split("_")

        if len(parts) >= 2 and parts[0] in ["small", "medium", "large", "learning"]:
            if parts[0] == "learning":
                config = f"{parts[0]}_{parts[1]}"
                run_num = parts[2].replace("run", "")
            else:
                config = f"{parts[0]}_network"
                run_num = parts[1].replace("run", "")

            history = run.history()

            metrics = {
                "test_rmse": run.summary.get("test_rmse", None),
                "test_r2": run.summary.get("test_r2", None),
                "test_mse": run.summary.get("test_mse", None),
                "test_mae": run.summary.get("test_mae", None),
            }

            config_dict = {
                "hidden_layers": run.config.get("hidden_layers", None),
                "batch_size": run.config.get("batch_size", None),
                "learning_rate": run.config.get("learning_rate", None),
                "dropout_rate": run.config.get("dropout_rate", None),
                "epochs": run.config.get("epochs", None),
            }

            results.append(
                {
                    "config": config,
                    "run": run_num,
                    "name": run_name,
                    "history": history,
                    "metrics": metrics,
                    "config_dict": config_dict,
                }
            )

    return results


def plot_loss_curves(results):
    """Vẽ biểu đồ loss của các cấu hình"""
    if not os.path.exists("results"):
        os.makedirs("results")

    configs = set([r["config"] for r in results])

    plt.figure(figsize=(15, 10))

    for i, config in enumerate(configs):
        config_results = [r for r in results if r["config"] == config]

        train_losses = []
        val_losses = []

        for r in config_results:
            if (
                "train_loss" in r["history"].columns
                and "val_loss" in r["history"].columns
            ):
                train_losses.append(r["history"]["train_loss"].values)
                val_losses.append(r["history"]["val_loss"].values)

        if not train_losses or not val_losses:
            print(f"Không tìm thấy dữ liệu loss cho cấu hình {config}")
            continue

        min_length = min([len(loss) for loss in train_losses + val_losses])
        train_losses = [loss[:min_length] for loss in train_losses]
        val_losses = [loss[:min_length] for loss in val_losses]

        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)

        mean_train_loss = np.mean(train_losses, axis=0)
        std_train_loss = np.std(train_losses, axis=0)
        mean_val_loss = np.mean(val_losses, axis=0)
        std_val_loss = np.std(val_losses, axis=0)

        epochs = range(1, min_length + 1)
        plt.subplot(2, 3, i + 1)

        plt.plot(epochs, mean_train_loss, label="Train Loss")
        plt.fill_between(
            epochs,
            mean_train_loss - std_train_loss,
            mean_train_loss + std_train_loss,
            alpha=0.2,
        )

        plt.plot(epochs, mean_val_loss, label="Validation Loss")
        plt.fill_between(
            epochs,
            mean_val_loss - std_val_loss,
            mean_val_loss + std_val_loss,
            alpha=0.2,
        )

        plt.title(f"Loss Curves - {config}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    plt.tight_layout()
    plt.savefig("results/all_loss_curves.png")
    plt.close()
    print("Đã lưu biểu đồ loss tại results/all_loss_curves.png")


def plot_metrics_comparison(results):
    """Vẽ biểu đồ so sánh các metric giữa các cấu hình"""
    if not os.path.exists("results"):
        os.makedirs("results")

    metrics_data = []

    for r in results:
        if "metrics" in r and r["metrics"]:
            metrics_data.append(
                {
                    "config": r["config"],
                    "run": r["run"],
                    "rmse": r["metrics"].get("test_rmse", np.nan),
                    "r2": r["metrics"].get("test_r2", np.nan),
                    "mse": r["metrics"].get("test_mse", np.nan),
                    "mae": r["metrics"].get("test_mae", np.nan),
                }
            )

    if not metrics_data:
        print("Không có dữ liệu metric để hiển thị")
        return

    metrics_df = pd.DataFrame(metrics_data)

    grouped_stats = metrics_df.groupby("config")

    stats_data = []
    for config, group in grouped_stats:
        stats_data.append(
            {
                "config": config,
                "rmse_mean": group["rmse"].mean(),
                "rmse_std": group["rmse"].std(),
                "r2_mean": group["r2"].mean(),
                "r2_std": group["r2"].std(),
                "mse_mean": group["mse"].mean(),
                "mse_std": group["mse"].std(),
                "mae_mean": group["mae"].mean(),
                "mae_std": group["mae"].std(),
            }
        )

    stats = pd.DataFrame(stats_data)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    sns.barplot(x="config", y="rmse", data=metrics_df, errorbar=None)
    sns.stripplot(
        x="config", y="rmse", data=metrics_df, color="black", size=4, jitter=True
    )
    plt.title("RMSE by Configuration")
    plt.ylabel("RMSE (thấp hơn tốt hơn)")
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    sns.barplot(x="config", y="r2", data=metrics_df, errorbar=None)
    sns.stripplot(
        x="config", y="r2", data=metrics_df, color="black", size=4, jitter=True
    )
    plt.title("R² by Configuration")
    plt.ylabel("R² (cao hơn tốt hơn)")
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    sns.barplot(x="config", y="mse", data=metrics_df, errorbar=None)
    sns.stripplot(
        x="config", y="mse", data=metrics_df, color="black", size=4, jitter=True
    )
    plt.title("MSE by Configuration")
    plt.ylabel("MSE (thấp hơn tốt hơn)")
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    sns.barplot(x="config", y="mae", data=metrics_df, errorbar=None)
    sns.stripplot(
        x="config", y="mae", data=metrics_df, color="black", size=4, jitter=True
    )
    plt.title("MAE by Configuration")
    plt.ylabel("MAE (thấp hơn tốt hơn)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("results/metrics_comparison.png")
    plt.close()
    print("Đã lưu biểu đồ so sánh metrics tại results/metrics_comparison.png")

    print("\nThống kê các cấu hình:")
    print("=" * 100)
    print(
        f"{'Configuration':<20} {'Avg RMSE':<15} {'Std RMSE':<15} {'Avg R²':<15} {'Std R²':<15} {'Avg MSE':<15} {'Std MSE':<15}"
    )
    print("-" * 100)

    for _, row in stats.iterrows():
        config = row["config"]
        rmse_mean = row["rmse_mean"]
        rmse_std = row["rmse_std"]
        r2_mean = row["r2_mean"]
        r2_std = row["r2_std"]
        mse_mean = row["mse_mean"]
        mse_std = row["mse_std"]

        print(
            f"{config:<20} {rmse_mean:<15.4f} {rmse_std:<15.4f} {r2_mean:<15.4f} {r2_std:<15.4f} {mse_mean:<15.4f} {mse_std:<15.4f}"
        )


def plot_hyperparameter_effects(results):
    """Phân tích ảnh hưởng của siêu tham số đến hiệu suất"""
    data = []

    for r in results:
        if "metrics" in r and r["metrics"] and "config_dict" in r:
            config = r["config_dict"]
            metrics = r["metrics"]

            if (
                config
                and "hidden_layers" in config
                and metrics
                and "test_rmse" in metrics
            ):
                data.append(
                    {
                        "config_name": r["config"],
                        "hidden_layers": str(config.get("hidden_layers")),
                        "batch_size": config.get("batch_size"),
                        "learning_rate": config.get("learning_rate"),
                        "dropout_rate": config.get("dropout_rate"),
                        "rmse": metrics.get("test_rmse"),
                        "r2": metrics.get("test_r2"),
                    }
                )

    if not data:
        print("Không đủ dữ liệu để phân tích siêu tham số")
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(15, 12))

    # Ảnh hưởng của learning rate
    plt.subplot(2, 2, 1)
    sns.boxplot(x="learning_rate", y="rmse", data=df)
    plt.title("Learning Rate vs RMSE")
    plt.xlabel("Learning Rate")
    plt.ylabel("RMSE")

    # Ảnh hưởng của batch size
    plt.subplot(2, 2, 2)
    sns.boxplot(x="batch_size", y="rmse", data=df)
    plt.title("Batch Size vs RMSE")
    plt.xlabel("Batch Size")
    plt.ylabel("RMSE")

    # Ảnh hưởng của dropout
    plt.subplot(2, 2, 3)
    sns.boxplot(x="dropout_rate", y="rmse", data=df)
    plt.title("Dropout Rate vs RMSE")
    plt.xlabel("Dropout Rate")
    plt.ylabel("RMSE")

    # Ảnh hưởng của kiến trúc mạng
    plt.subplot(2, 2, 4)
    sns.boxplot(x="hidden_layers", y="rmse", data=df)
    plt.title("Network Architecture vs RMSE")
    plt.xlabel("Hidden Layers")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("results/hyperparameter_analysis.png")
    plt.close()
    print(
        "Đã lưu biểu đồ phân tích siêu tham số tại results/hyperparameter_analysis.png"
    )


def main():
    print("Bắt đầu phân tích kết quả từ Weights & Biases...")
    results = get_wandb_data()

    if not results:
        print(
            "Không tìm thấy dữ liệu từ Weights & Biases. Hãy chạy chương trình huấn luyện trước."
        )
        return

    plot_loss_curves(results)

    plot_metrics_comparison(results)

    plot_hyperparameter_effects(results)

    print("\nĐã hoàn thành phân tích và tạo biểu đồ trong thư mục 'results/'")


if __name__ == "__main__":
    main()
