"""
Model utility functions for training and evaluation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class PatientDataModel(nn.Module):
    """
    Neural network model for patient demographic and biomarker data
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int = 1, dropout: float = 0.3):
        super(PatientDataModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class GenomicModel(nn.Module):
    """
    CNN-based model for genomic sequence data
    """
    def __init__(self, sequence_length: int, embedding_dim: int, num_filters: int, filter_sizes: List[int], output_size: int = 1):
        super(GenomicModel, self).__init__()
        
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, T, G, C
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, sequence_length)
        
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        concatenated = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)
        return self.sigmoid(output)

class EnsembleModel:
    """
    Ensemble model combining multiple base models
    """
    def __init__(self, models: List, voting_method: str = 'soft'):
        self.models = models
        self.voting_method = voting_method
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        if self.voting_method == 'soft':
            # Average probabilities
            return np.mean(predictions, axis=0)
        else:
            # Majority voting
            binary_preds = [np.round(pred) for pred in predictions]
            return np.mean(binary_preds, axis=0)

def train_patient_model(X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray,
                       config: Dict) -> Tuple[PatientDataModel, Dict]:
    """
    Train the patient data neural network model
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Model configuration
        
    Returns:
        Trained model and training history
    """
    # Convert to PyTorch tensors, handling pandas Series
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
    y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val)
    y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model
    model = PatientDataModel(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        output_size=config['output_size'],
        dropout=config['dropout']
    )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_preds = val_outputs.squeeze().numpy()
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            val_auc = roc_auc_score(y_val_np, val_preds)
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    return model, history

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest model as baseline
    
    Args:
        X_train, y_train: Training data
        
    Returns:
        Trained Random Forest model
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model") -> Dict:
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Handle PyTorch models
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
            outputs = model(X_test_tensor)
            y_pred_proba = torch.sigmoid(outputs).squeeze().numpy()
            y_pred = (y_pred_proba > 0.5).astype(int)
    elif hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = None
    
    # Convert y_test to numpy if it's a pandas Series
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    metrics = {
        'accuracy': accuracy_score(y_test_np, y_pred),
        'precision': precision_score(y_test_np, y_pred, zero_division=0),
        'recall': recall_score(y_test_np, y_pred, zero_division=0),
        'f1': f1_score(y_test_np, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_test_np, y_pred_proba)
    
    print(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return metrics

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib not available. Skipping plot generation.")
        print("Training history summary:")
        print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"  Best validation AUC: {max(history['val_auc']):.4f}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # AUC plot
    ax2.plot(history['val_auc'], label='Validation AUC')
    ax2.set_title('Validation AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_model(model, filepath: str):
    """
    Save a trained model
    
    Args:
        model: Trained model to save
        filepath: Path to save the model
    """
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), filepath)
    else:
        joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, filepath: str, **kwargs):
    """
    Load a saved model
    
    Args:
        model_class: Class of the model to load
        filepath: Path to the saved model
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Loaded model
    """
    if issubclass(model_class, nn.Module):
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(filepath))
        model.eval()
    else:
        model = joblib.load(filepath)
    
    print(f"Model loaded from {filepath}")
    return model
