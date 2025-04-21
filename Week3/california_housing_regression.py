import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import wandb

if not os.path.exists('results'):
    os.makedirs('results')

class CaliforniaHousingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Tạo mô hình MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=1, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        
        # Layer đầu vào
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Các hidden layer
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Layer đầu ra
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, run, scheduler=None):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Learning rate scheduler
        if scheduler:
            scheduler.step()
            wandb.log({"learning_rate": scheduler.get_last_lr()[0]}, step=epoch)
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "epoch": epoch
        })
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, criterion, device, scaler_y=None):
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            if scaler_y:
                # Chuyển về giá trị thực
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                outputs_np = scaler_y.inverse_transform(outputs_np)
                targets_np = scaler_y.inverse_transform(targets_np)
                all_predictions.append(outputs_np)
                all_targets.append(targets_np)
            else:
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    return {
        'test_loss': test_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def plot_results(train_losses, val_losses, run_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss - {run_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/{run_name}_loss.png')
    wandb.log({"loss_plot": wandb.Image(plt)})
    plt.close()


def main():
    
    # Các cấu hình siêu tham số
    configs = [
        {
            'name': 'small_network',
            'hidden_layers': [32, 16],
            'batch_size': 64,
            'learning_rate': 0.001,
            'dropout_rate': 0.1,
            'epochs': 100
        },
        {
            'name': 'medium_network',
            'hidden_layers': [64, 32, 16],
            'batch_size': 64,
            'learning_rate': 0.001,
            'dropout_rate': 0.2,
            'epochs': 100
        },
        {
            'name': 'large_network',
            'hidden_layers': [128, 64, 32],
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'epochs': 100
        },
        {
            'name': 'learning_rate_small',
            'hidden_layers': [64, 32],
            'batch_size': 64,
            'learning_rate': 0.0001,
            'dropout_rate': 0.2,
            'epochs': 100
        },
        {
            'name': 'learning_rate_large',
            'hidden_layers': [64, 32],
            'batch_size': 64,
            'learning_rate': 0.01,
            'dropout_rate': 0.2,
            'epochs': 100
        }
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    print(f"Shape of features: {X.shape}")
    print(f"Shape of targets: {y.shape}")
    print(f"Missing values in features: {np.isnan(X).sum()}")
    print(f"Missing values in targets: {np.isnan(y).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    wandb.login()

    config_results = []

    for config in configs:
        print(f"\nRunning configuration: {config['name']}")
        config_metrics = []
        
        for run_idx in range(5):
            run_name = f"{config['name']}_run{run_idx+1}"
            print(f"  Run {run_idx+1}/5")
            
            wandb_run = wandb.init(
                project="california_housing_regression",
                name=run_name,
                config={
                    "hidden_layers": config['hidden_layers'],
                    "batch_size": config['batch_size'],
                    "learning_rate": config['learning_rate'],
                    "dropout_rate": config['dropout_rate'],
                    "epochs": config['epochs'],
                    "architecture": "MLP",
                    "dataset": "California Housing"
                },
                reinit=True
            )
            
            train_dataset = CaliforniaHousingDataset(X_train, y_train)
            val_dataset = CaliforniaHousingDataset(X_val, y_val)
            test_dataset = CaliforniaHousingDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
            
            input_size = X_train.shape[1]
            model = MLP(input_size, config['hidden_layers'], dropout_rate=config['dropout_rate']).to(device)

            wandb.watch(model, log="all")

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            
            # ttrain mô hình
            train_losses, val_losses = train_model(
                model, train_loader, val_loader, criterion, optimizer, device, 
                config['epochs'], wandb_run, scheduler
            )
            
            # Vẽ biểu đồ loss
            plot_results(train_losses, val_losses, run_name)
            
            # Đánh giá mô hình trên tập test
            metrics = evaluate_model(model, test_loader, criterion, device, scaler_y)
            config_metrics.append(metrics)
            
            # Log metrics to wandb
            wandb.log({
                "test_loss": metrics['test_loss'],
                "test_mse": metrics['mse'],
                "test_rmse": metrics['rmse'],
                "test_mae": metrics['mae'],
                "test_r2": metrics['r2']
            })
            
            # Tạo một bảng dữ liệu mẫu trong wandb
            example_data = []
            for i in range(min(5, len(test_dataset))):
                features, target = test_dataset[i]
                prediction = model(features.unsqueeze(0).to(device)).cpu().detach().numpy()[0][0]
                
                # Chuyển về giá trị thực nếu đã chuẩn hóa
                real_target = scaler_y.inverse_transform([[target.item()]])[0][0]
                real_prediction = scaler_y.inverse_transform([[prediction]])[0][0]
                
                example_data.append([i, real_target, real_prediction, abs(real_target - real_prediction)])
            
            example_table = wandb.Table(columns=["id", "actual", "prediction", "error"], data=example_data)
            wandb.log({"predictions_sample": example_table})
            
            # Lưu mô hình vào wandb
            torch.save(model.state_dict(), f"checkpoints/{run_name}_model.pt")
            wandb.save(f"checkpoints/{run_name}_model.pt")
            
            wandb_run.finish()
            
            print(f"  Test RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        # Tính toán trung bình và độ lệch chuẩn cho mỗi cấu hình
        rmse_values = [m['rmse'] for m in config_metrics]
        r2_values = [m['r2'] for m in config_metrics]
        
        avg_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
        avg_r2 = np.mean(r2_values)
        std_r2 = np.std(r2_values)
        
        config_results.append({
            'config': config['name'],
            'avg_rmse': avg_rmse,
            'std_rmse': std_rmse,
            'avg_r2': avg_r2,
            'std_r2': std_r2
        })
        
        print(f"  Average RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
        print(f"  Average R²: {avg_r2:.4f} ± {std_r2:.4f}")
    print("\nSummary of Results:")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Avg RMSE':<15} {'Std RMSE':<15} {'Avg R²':<15} {'Std R²':<15}")
    print("-" * 80)
    
    for result in config_results:
        print(f"{result['config']:<20} {result['avg_rmse']:<15.4f} {result['std_rmse']:<15.4f} {result['avg_r2']:<15.4f} {result['std_r2']:<15.4f}")

    with wandb.init(project="california_housing_regression", name="summary", reinit=True) as summary_run:
        summary_table = wandb.Table(
            columns=["config", "avg_rmse", "std_rmse", "avg_r2", "std_r2"],
            data=[[r['config'], r['avg_rmse'], r['std_rmse'], r['avg_r2'], r['std_r2']] for r in config_results]
        )
        wandb.log({"summary_results": summary_table})
        
        plt.figure(figsize=(10, 6))
        configs = [r['config'] for r in config_results]
        rmse_means = [r['avg_rmse'] for r in config_results]
        rmse_stds = [r['std_rmse'] for r in config_results]
        
        plt.bar(configs, rmse_means, yerr=rmse_stds, capsize=10)
        plt.ylabel('RMSE (thấp hơn tốt hơn)')
        plt.title('Trung bình RMSE theo cấu hình')
        plt.xticks(rotation=45)
        plt.tight_layout()
        wandb.log({"rmse_comparison": wandb.Image(plt)})
        plt.close()
        
        # Tạo biểu đồ tổng hợp R²
        plt.figure(figsize=(10, 6))
        r2_means = [r['avg_r2'] for r in config_results]
        r2_stds = [r['std_r2'] for r in config_results]
        
        plt.bar(configs, r2_means, yerr=r2_stds, capsize=10)
        plt.ylabel('R² (cao hơn tốt hơn)')
        plt.title('Trung bình R² theo cấu hình')
        plt.xticks(rotation=45)
        plt.tight_layout()
        wandb.log({"r2_comparison": wandb.Image(plt)})
        plt.close()

if __name__ == "__main__":
    main() 