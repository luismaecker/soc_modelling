import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

########################################################
# Functions for data preprocessing
########################################################

# Apply Savitzky-Golay smoothing filter
def apply_savitzky_golay(X, window_length=11, polyorder=2, deriv=0, delta=2):
    """
    Apply Savitzky-Golay smoothing filter to spectral data.
    
    Parameters:
    -----------
    X : ndarray
        Input data array with shape (n_samples, n_features)
    window_length : int
        Length of the filter window (must be odd)
    polyorder : int
        Order of the polynomial used to fit the samples
    deriv : int
        Order of the derivative to compute
    delta : float
        Sample spacing
        
    Returns:
    --------
    ndarray
        Smoothed data with same shape as input
    """
    smoothed_X = savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, axis=1)
    return smoothed_X

# Apply Standard Normal Variate (SNV) transformation
def standard_normal_variate(X):
    """
    Apply Standard Normal Variate (SNV) transformation to spectral data.
    
    Parameters:
    -----------
    X : ndarray
        Input data array with shape (n_samples, n_features)
        
    Returns:
    --------
    ndarray
        SNV transformed data with same shape as input
    """
    snv_X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    return snv_X

########################################################
# Functions for sampling
########################################################

def kennard_stone(X, n_samples):
    """
    Kennard-Stone algorithm to select a representative subset of samples.
    
    Parameters:
    -----------
    X : ndarray
        Input data array with shape (n_samples, n_features)
    n_samples : int
        Number of samples to select
        
    Returns:
    --------
    ndarray
        Indices of selected samples
    """
    n = X.shape[0]
    selected = np.zeros(n_samples, dtype=int)
    
    # Start with the two most distant samples
    dist_matrix = pairwise_distances(X)
    selected[0] = np.argmax(np.sum(dist_matrix, axis=1))
    selected[1] = np.argmax(dist_matrix[selected[0]])
    
    for i in range(2, n_samples):
        remaining = np.setdiff1d(np.arange(n), selected[:i])
        dist_to_selected = dist_matrix[remaining][:, selected[:i]]
        min_dist = np.min(dist_to_selected, axis=1)
        selected[i] = remaining[np.argmax(min_dist)]
    
    return selected

########################################################
# Functions for model optimization
########################################################
def calculate_rmse(n_components, X_train, y_train):
    """
    Calculate RMSE using cross-validation for PLS Regression.
    
    Parameters:
    -----------
    n_components : int
        Number of PLS components
    X_train : ndarray
        Training feature data
    y_train : ndarray
        Training target data
        
    Returns:
    --------
    float
        Cross-validated RMSE
    """
    pls = PLSRegression(n_components=n_components)
    # Use negative mean squared error as cross-validation scoring
    mse = cross_val_score(pls, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-mse).mean()  # Take square root of MSE and average it across folds
    return rmse

def optimize_pls_components(X_train, y_train, max_components=None, step=1, fine_tune=True, 
                           fine_tune_range=10, show_progress=True, plot_results=False, 
                           figsize=(12, 5)):
    """
    Optimize the number of PLS components using cross-validation.
    
    Parameters:
    -----------
    X_train : ndarray
        Training feature data with shape (n_samples, n_features)
    y_train : ndarray
        Training target data with shape (n_samples,)
    max_components : int, optional
        Maximum number of components to test
    step : int, optional
        Step size for testing components in rough estimation
    fine_tune : bool, optional
        Whether to perform fine-tuning around the optimal rough estimate
    fine_tune_range : int, optional
        Range around optimal rough estimate to fine-tune
    show_progress : bool, optional
        Whether to display progress bars
    plot_results : bool, optional
        Whether to generate plots of RMSE vs components
    figsize : tuple, optional
        Figure size for plots as (width, height) in inches
        
    Returns:
    --------
    dict
        Dictionary containing optimization results:
        - 'optimal_n': Final optimal number of components
        - 'rough_results': Dictionary with rough optimization results
        - 'fine_results': Dictionary with fine-tuning results (if performed)
    """
    from tqdm.notebook import tqdm
    import matplotlib.pyplot as plt
    
    if max_components is None:
        max_components = min(X_train.shape[0], X_train.shape[1], 200)
    
    # First pass: test in larger steps
    rough_components = list(range(1, max_components + 1, step))
    rough_rmse_values = []
    
    # Create iterator with or without progress bar
    if show_progress:
        rough_iterator = tqdm(rough_components, desc="Rough Optimization")
    else:
        rough_iterator = rough_components
    
    # Calculate RMSE for each number of components - rough search
    for n in rough_iterator:

        # Calculate mean RMSE over 5-fold cross-validation
        rmse = calculate_rmse(n, X_train, y_train)
        rough_rmse_values.append(rmse)
        
    # Find optimal number of components from rough search
    optimal_n_rough = rough_components[np.argmin(rough_rmse_values)]
    min_rmse_rough = min(rough_rmse_values)
    
    # Create results dictionary
    results = {
        'optimal_n': optimal_n_rough,
        'rough_results': {
            'components': rough_components,
            'rmse_values': rough_rmse_values,
            'optimal_n': optimal_n_rough,
            'min_rmse': min_rmse_rough
        }
    }
    
    # Plot rough optimization results if requested
    if plot_results:
        if fine_tune and step > 1:
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
        else:
            plt.figure(figsize=(figsize[0]//2, figsize[1]))
            
        plt.plot(rough_components, rough_rmse_values, 'o-')
        plt.axvline(x=optimal_n_rough, color='r', linestyle='--')
        plt.text(optimal_n_rough + 0.5, min_rmse_rough,
                 f'Optimal: {optimal_n_rough}\nRMSE: {min_rmse_rough:.4f}',
                 verticalalignment='bottom')
        plt.xlabel('Number of PLS Components')
        plt.ylabel('RMSE (Cross-Validation)')
        plt.title('Rough Component Optimization')
        plt.grid(True)
    
    # Fine-tune if requested and if step size > 1
    if fine_tune and step > 1:
        
        fine_tune_components = list(range(
            max(1, optimal_n_rough - fine_tune_range),
            min(max_components, optimal_n_rough + fine_tune_range) + 1
        ))
        
        fine_rmse_values = []
        
        # Create iterator with or without progress bar for fine-tuning
        if show_progress:
            fine_iterator = tqdm(fine_tune_components, desc="Fine Tuning")
        else:
            fine_iterator = fine_tune_components
            
        for n in fine_iterator:
            rmse = calculate_rmse(n, X_train, y_train)
            fine_rmse_values.append(rmse)
            
        # Find optimal number of components from fine-tuning
        optimal_n_fine = fine_tune_components[np.argmin(fine_rmse_values)]
        min_rmse_fine = min(fine_rmse_values)
        
        # Update results with fine-tuning information
        results['optimal_n'] = optimal_n_fine
        results['fine_results'] = {
            'components': fine_tune_components,
            'rmse_values': fine_rmse_values,
            'optimal_n': optimal_n_fine,
            'min_rmse': min_rmse_fine
        }
        
        # Plot fine-tuning results if requested
        if plot_results:
            plt.subplot(1, 2, 2)
            plt.plot(fine_tune_components, fine_rmse_values, 'o-', color='green')
            plt.axvline(x=optimal_n_fine, color='r', linestyle='--')
            plt.text(optimal_n_fine + 0.5, min_rmse_fine,
                     f'Optimal: {optimal_n_fine}\nRMSE: {min_rmse_fine:.4f}',
                     verticalalignment='bottom')
            plt.xlabel('Number of PLS Components')
            plt.ylabel('RMSE (Cross-Validation)')
            plt.title('Fine-Tuned Component Optimization')
            plt.grid(True)
    
    # Show the plot if it was created
    if plot_results:
        plt.tight_layout()
        plt.show()

    print(f"Optimal number of components: {results['optimal_n']}")
    
    return results





########################################################
# Functions for visualization
########################################################

def plot_components_vs_rmse(components, rmse_values, model_name="Model", figsize=(10, 5)):
    """
    Plot RMSE vs. number of components.
    
    Parameters:
    -----------
    components : list or array
        Component numbers tested
    rmse_values : list or array
        RMSE values for each component number
    model_name : str, optional
        Name to display in plot title
    figsize : tuple, optional
        Figure size as (width, height) in inches
    """
    plt.figure(figsize=figsize)
    plt.plot(components, rmse_values, marker='o')
    plt.xlabel('Number of PLS Components')
    plt.ylabel('RMSE (Cross-Validation)')
    plt.title(f'RMSE vs. Number of PLS Components ({model_name})')
    plt.grid(True)
    plt.show()
    
def plot_prediction_scatter(y_true, y_pred, model_name="Model", figsize=(7, 5), color ="#1f77b4"):
    """
    Plot actual vs. predicted values scatter plot.
    
    Parameters:
    -----------
    y_true : ndarray
        True target values
    y_pred : ndarray
        Predicted target values
    model_name : str, optional
        Name to display in plot title
    figsize : tuple, optional
        Figure size as (width, height) in inches
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5, color = color)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 
             color='red', linestyle='--')  # Perfect prediction line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Actual vs Predicted Values')
    plt.grid(True)
    plt.show()

def compare_prediction_scatter(true_pred_pairs, model_names=None, figsize=(12, 8), n_cols=2):
    """
    Compare multiple model predictions in subplots.
    
    Parameters:
    -----------
    true_pred_pairs : list of tuples
        List of (y_true, y_pred) pairs for each model
    model_names : list of str, optional
        Names of models to display in subplot titles
    figsize : tuple, optional
        Figure size as (width, height) in inches
    n_cols : int, optional
        Number of columns in subplot grid
    """
    n_models = len(true_pred_pairs)
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]
    
    n_rows = (n_models + n_cols - 1) // n_cols
    
    plt.figure(figsize=figsize)
    for i, ((y_true, y_pred), model_name) in enumerate(zip(true_pred_pairs, model_names)):
        plt.subplot(n_rows, n_cols, i+1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 
                 color='red', linestyle='--')  # Perfect prediction line
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_pca_comparison(X_full, X_train_random, X_test_random, X_train_ks=None, X_test_ks=None, figsize=(12, 8)):
    """
    Plot PCA comparisons between different data splits.
    
    Parameters:
    -----------
    X_full : ndarray
        Full dataset
    X_train : ndarray
        Train set from random sampling
    X_test : ndarray
        Test set from random sampling
    X_train_ks : ndarray, optional
        Train set from Kennard-Stone sampling
    X_test_ks : ndarray, optional
        Test set from Kennard-Stone sampling
    figsize : tuple, optional
        Figure size as (width, height) in inches
    """
    
    # Apply PCA to full dataset
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full)
    
    # Transform subsets
    X_train_pca = pca.transform(X_train_random)
    X_test_pca = pca.transform(X_test_random)
    
    # Print Shapes of the Splits
    print("Random Split:")
    print(f"Train set shape: {X_train_random.shape}")
    print(f"Test set shape: {X_test_random.shape}")

    print("\nKennard-Stone Split:")
    print(f"Train set shape: {X_train_ks.shape}")
    print(f"Test set shape: {X_test_ks.shape}")


    plt.figure(figsize=figsize)
    
    # Plot full dataset
    plt.subplot(2, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', label='Full Data Set')
    plt.title('Full Data Set')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Plot random split
    plt.subplot(2, 2, 2)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='red', label='Random Train Set')
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c='green', label='Random Test Set')
    plt.title('Random Split (70% Train, 30% Test)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    
    # If Kennard-Stone split is provided
    if X_train_ks is not None and X_test_ks is not None:
        X_train_ks_pca = pca.transform(X_train_ks)
        X_test_ks_pca = pca.transform(X_test_ks)
        
        plt.subplot(2, 2, 3)
        plt.scatter(X_train_ks_pca[:, 0], X_train_ks_pca[:, 1], c='red', label='KS Train Set')
        plt.scatter(X_test_ks_pca[:, 0], X_test_ks_pca[:, 1], c='green', label='KS Test Set')
        plt.title('Kennard-Stone Split (70% Train, 30% Test)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
    
    plt.tight_layout()
    plt.show()





########################################################
# Functions for model evaluation
########################################################

def evaluate_model(model, X_test, y_test, print_metrics=True, show_plot=True, plot_kwargs=None):
    """
    Evaluate a fitted model on test data.
    
    Parameters:
    -----------
    model : fitted model object
        Trained model with predict method
    X_test : ndarray
        Test feature data
    y_test : ndarray
        Test target data
    print_metrics : bool, optional
        Whether to print metrics to console
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and predictions
    """
    # Make predictions
    y_pred = model.predict(X_test)
    if hasattr(y_pred, 'squeeze'):
        y_pred = y_pred.squeeze()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    rpd = np.std(y_test) / rmse
    
    if print_metrics:
        print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
        print(f'R²: {r2:.4f}')
        print(f'Bias: {bias:.4f}')
        print(f'RPD: {rpd:.4f}')

    if show_plot:
        plot_prediction_scatter(y_test, y_pred, **(plot_kwargs or {}))
    
    return {
        'y_pred': y_pred,
        'rmse': rmse,
        'r2': r2,
        'bias': bias,
        'rpd': rpd
    }


########################################################
# Modelling
########################################################


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Additional dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Select the last time step
        out = out[:, -1, :]
        # Forward pass through fully connected layer
        out = self.fc(out)
        return out
        
    
    def predict(self, X):
        self.eval()  # Set to evaluation mode
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self(X_tensor.unsqueeze(1)).squeeze()
        return predictions.numpy()

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

def train_lstm(X_train, X_val, X_test, y_train, y_val, y_test, 
                          hidden_size=256, num_layers=5, num_epochs=500, 
                          learning_rate=0.005, patience=7, dropout=0.2,
                          show_training=False, show_history=True):
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Initialize model and training components
    input_size = X_train.shape[1]
    model = LSTMModel(input_size, hidden_size, 1, num_layers, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor.unsqueeze(1))
        train_loss = criterion(train_outputs.squeeze(), y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.unsqueeze(1))
            val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            
        # Store losses
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            history['best_epoch'] = epoch - patience
            break

        if show_training:    
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Train Loss: {train_loss.item():.4f}, '
                    f'Val Loss: {val_loss.item():.4f}')
    
    # Load best model
    model.load_state_dict(early_stopping.best_model)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_test_tensor.unsqueeze(1)).squeeze()
        test_loss = criterion(final_predictions, y_test_tensor)

        print(test_loss)
    
    if show_history:
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


    return model



# Function to compute spectral indices
def compute_indices(df):
    """
    Compute vegetation and soil indices useful for SOC modeling
    using Sentinel-2 bands from the S2-soilsuite dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Sentinel-2 bands (MREF_B* columns)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with computed spectral indices
    """
    # Create a new DataFrame to store indices
    indices = pd.DataFrame(index=df.index)
    
    # Extract bands for easier reference
    B = df['MREF_B02']  # Blue
    G = df['MREF_B03']  # Green
    R = df['MREF_B04']  # Red
    RE1 = df['MREF_B05']  # Red Edge 1
    RE2 = df['MREF_B06']  # Red Edge 2
    RE3 = df['MREF_B07']  # Red Edge 3
    N = df['MREF_B08']  # NIR
    N2 = df['MREF_B8A']  # NIR narrow band
    S1 = df['MREF_B11']  # SWIR 1
    S2 = df['MREF_B12']  # SWIR 2
    
    # Constants used in some indices
    L = 0.5  # Soil adjustment factor

    # ----- Vegetation Indices -----
    
    # NDVI - Normalized Difference Vegetation Index
    indices['NDVI'] = (N - R) / (N + R)
    
    # GNDVI - Green Normalized Difference Vegetation Index
    indices['GNDVI'] = (N - G) / (N + G)
    
    # EVI - Enhanced Vegetation Index
    indices['EVI'] = 2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)
    
    # SAVI - Soil-Adjusted Vegetation Index
    indices['SAVI'] = (1 + L) * (N - R) / (N + R + L)
    
    
    # OSAVI - Optimized Soil-Adjusted Vegetation Index
    indices['OSAVI'] = (N - R) / (N + R + 0.16)
    
    # NDMI - Normalized Difference Moisture Index
    indices['NDMI'] = (N - S1) / (N + S1)
    
    # CIG - Chlorophyll Index Green
    indices['CIG'] = (N / G) - 1
    
    # CIRE - Chlorophyll Index Red Edge
    indices['CIRE'] = (N / RE1) - 1
    
    # MTCI - MERIS Terrestrial Chlorophyll Index
    indices['MTCI'] = (RE3 - RE1) / (RE1 - R)
    
    # NDRE - Normalized Difference Red Edge
    indices['NDRE'] = (N - RE1) / (N + RE1)
    
    # IRECI - Inverted Red Edge Chlorophyll Index
    indices['IRECI'] = (RE3 - R) / (RE1 / RE2)
    
    # S2REP - Sentinel-2 Red Edge Position
    indices['S2REP'] = 705 + 35 * ((RE3 + R) / 2 - RE1) / (RE2 - RE1)
    
    # ----- Soil Indices -----
    
    # BI - Bare Soil Index
    indices['BI'] = ((S1 + R) - (N + B)) / ((S1 + R) + (N + B))
    
    # NDSoI - Normalized Difference Soil Index
    indices['NDSoI'] = (S1 - G) / (S1 + G)
    
    # MBI - Modified Bare Soil Index
    indices['MBI'] = ((S1 - S2 - N) / (S1 + S2 + N)) + 0.5
    
    # NSDS - Normalized Shortwave Infrared Difference Soil-Moisture
    indices['NSDS'] = (S1 - S2) / (S1 + S2)
    
    # NSDSI1 - Normalized Shortwave-Infrared Difference Bare Soil Moisture Index 1
    indices['NSDSI1'] = (S1 - N) / (S1 + N)
    
    # NSDSI2 - Normalized Shortwave-Infrared Difference Bare Soil Moisture Index 2
    indices['NSDSI2'] = (S1 - S2) / (S1 + S2)
    
    return indices

def plot_spectra_comparison(*spectra, wavelengths=None, labels=None, title="Spectral Comparison"):
    """
    Create a comparison plot of multiple spectra.
    
    Args:
        *spectra: Variable number of spectrum data arrays
        wavelengths: X-axis values (optional)
        labels: List/tuple of labels for legend matching number of spectra
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Generate x-axis values if not provided
    if wavelengths is None:
        wavelengths = range(len(spectra[0]))
    
    # Set default labels if not provided
    if labels is None:
        labels = [f'Spectrum {i+1}' for i in range(len(spectra))]
    
    # Ensure number of labels matches number of spectra
    if len(labels) != len(spectra):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of spectra ({len(spectra)})")
    
    # Plot each spectrum
    for spectrum, label in zip(spectra, labels):
        plt.plot(wavelengths, spectrum, label=label, linewidth=2)
    
    # Customize plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add a subtle background color
    plt.gca().set_facecolor('#f8f9fa')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()