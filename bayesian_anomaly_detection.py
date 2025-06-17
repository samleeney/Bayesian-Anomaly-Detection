#!/usr/bin/env python3
"""
Bayesian Anomaly Detection - Python Script Version

This script demonstrates Bayesian anomaly detection for robust parameter inference
in the presence of outliers. It implements the methodology from:

- Leeney et al. (2022): "Bayesian approach to radio frequency interference mitigation"
  https://arxiv.org/abs/2211.15448
- Anstey and Leeney (2023): "Enhanced Bayesian RFI Mitigation and Transient Flagging 
  Using Likelihood Reweighting" https://arxiv.org/abs/2310.02146

The method introduces a binary anomaly mask with Bernoulli priors and uses a 
piecewise likelihood that combines normal and anomalous cases. This allows for
automatic outlier detection whilst preserving principled Bayesian inference.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """Run the complete Bayesian anomaly detection demonstration."""
    
    print("=== Bayesian Anomaly Detection Demonstration ===\n")
    
    # Generate mock data
    print("1. Generating mock data...")
    x, y, m_true, c_true, sig_true = generate_data()
    
    # Add anomalies
    print("2. Adding anomalies to data...")
    y = add_anomalies(y)
    
    # Plot data
    print("3. Plotting data...")
    plot_data(x, y, m_true, c_true)
    
    # Set up priors and parameters
    print("4. Setting up priors and parameters...")
    delta = np.max(y)
    prior_func = create_prior_function()
    
    # Define likelihood functions
    likelihood_standard = create_standard_likelihood(x, y)
    likelihood_anomaly = create_anomaly_likelihood(x, y, delta)
    
    # Test likelihood functions
    print("5. Testing likelihood functions...")
    test_likelihoods(likelihood_standard, likelihood_anomaly)
    
    # Fit with standard likelihood
    print("6. Fitting with standard likelihood...")
    samples_standard = run_mcmc(likelihood_standard, prior_func, "standard")
    
    # Fit with anomaly-corrected likelihood  
    print("7. Fitting with anomaly-corrected likelihood...")
    samples_anomaly = run_mcmc(likelihood_anomaly, prior_func, "anomaly-corrected")
    
    # Compare results
    print("8. Comparing results...")
    compare_results(samples_standard, samples_anomaly, m_true, c_true, sig_true)
    
    # Create visualisations
    print("9. Creating visualisations...")
    create_visualisations(x, y, samples_standard, samples_anomaly, m_true, c_true)
    
    print("\n=== Analysis Complete ===")
    print("Check the generated plots to see the dramatic improvement in parameter")
    print("recovery when using Bayesian anomaly detection!")


def generate_data():
    """Generate clean linear data with Gaussian noise."""
    np.random.seed(42)  # For reproducibility
    
    N = 25
    x = np.linspace(0, 25, N)
    m_true = 1
    c_true = 1  
    sig_true = 5
    
    y = m_true * x + c_true + np.random.randn(N) * sig_true
    
    return x, y, m_true, c_true, sig_true


def add_anomalies(y):
    """Add two large anomalies to the data."""
    y_anomalous = y.copy()
    y_anomalous[10] += 100
    y_anomalous[15] += 100
    return y_anomalous


def plot_data(x, y, m_true, c_true):
    """Plot the original and anomalous data."""
    # Create clean version for comparison
    y_clean = y.copy()
    y_clean[10] -= 100  # Remove anomalies for clean plot
    y_clean[15] -= 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data
    ax1.plot(x, y_clean, "o", markersize=8, alpha=0.7, label='Clean data')
    ax1.plot(x, m_true * x + c_true, 'k--', alpha=0.7, label=f'True model: y = {m_true}x + {c_true}')
    ax1.set_title("Original Data", fontsize=16)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Data with anomalies
    ax2.plot(x, y, "ro", markersize=8, alpha=0.7, label='Data with anomalies')
    ax2.plot(x, m_true * x + c_true, 'k--', alpha=0.7, label=f'True model: y = {m_true}x + {c_true}')
    
    # Highlight anomalous points
    ax2.plot(x[10], y[10], 'ro', markersize=12, markeredgecolor='black', 
             markeredgewidth=2, label='Anomalous points')
    ax2.plot(x[15], y[15], 'ro', markersize=12, markeredgecolor='black', 
             markeredgewidth=2)
    
    ax2.set_title("Data with Anomalies", fontsize=16)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_prior_function():
    """Create the prior transformation function."""
    def uniform_prior(a, b):
        """Simple uniform prior function."""
        def prior_func(u):
            return a + (b - a) * u
        return prior_func

    def prior(hypercube):
        """Transform from unit hypercube to parameter space."""
        theta = np.zeros_like(hypercube)
        # More reasonable prior ranges
        theta[0] = uniform_prior(-2, 4)(hypercube[0])      # m: slope
        theta[1] = uniform_prior(-20, 20)(hypercube[1])    # c: intercept  
        theta[2] = uniform_prior(0.1, 50)(hypercube[2])    # sig: noise
        theta[3] = uniform_prior(-10, -0.1)(hypercube[3])  # logp: anomaly threshold
        return theta
    
    return prior


def create_standard_likelihood(x, y):
    """Create standard Gaussian likelihood function."""
    def likelihood(theta):
        m = theta[0]
        c = theta[1] 
        sig = theta[2]
        
        # Add regularisation to avoid numerical issues
        sig = np.maximum(sig, 0.1)
        
        y_pred = m * x + c
        logL = -0.5 * ((y_pred - y) ** 2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)
        
        # Handle potential numerical issues
        if np.any(~np.isfinite(logL)):
            return -np.inf
        
        return logL.sum()
    
    return likelihood


def create_anomaly_likelihood(x, y, delta):
    """Create anomaly-corrected likelihood function."""
    def likelihood(theta):
        m = theta[0]
        c = theta[1]
        sig = theta[2] 
        logp = theta[3]  # logp is a free parameter
        
        # Add regularisation to avoid numerical issues
        sig = np.maximum(sig, 0.1)
        p = np.exp(np.clip(logp, -20, -0.1))
        
        y_pred = m * x + c
        
        # Standard log-likelihood for each point
        logL_normal = -0.5 * ((y_pred - y) ** 2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)
        
        # Add log(1-p) for the probability that each point is not anomalous
        logL_with_prior = logL_normal + np.log(1 - p)
        
        # Anomaly threshold: log(p/Delta)
        anomaly_threshold = logp - np.log(delta)
        
        # For each point, take the maximum of normal likelihood and anomaly threshold
        logL_corrected = np.maximum(logL_with_prior, anomaly_threshold)
        
        return logL_corrected.sum()
    
    return likelihood


def test_likelihoods(likelihood_standard, likelihood_anomaly):
    """Test likelihood functions with known parameter values."""
    # Test parameters
    true_params_standard = np.array([1.0, 1.0, 5.0])
    true_params_anomaly = np.array([1.0, 1.0, 5.0, -3.0])
    
    logL_standard = likelihood_standard(true_params_standard)
    logL_anomaly = likelihood_anomaly(true_params_anomaly)
    
    print(f"   Standard likelihood at true values: {logL_standard:.2f}")
    print(f"   Anomaly-corrected likelihood at true values: {logL_anomaly:.2f}")


def run_mcmc(log_likelihood, prior_func, method_name, n_samples=5000, n_burn=1000):
    """Run MCMC sampling using Metropolis-Hastings algorithm."""
    print(f"   Running MCMC for {method_name} method...")
    
    n_dims = 4
    samples = np.zeros((n_samples, n_dims))
    
    # Start from reasonable point
    current_hypercube = np.array([0.5, 0.5, 0.3, 0.3])
    current_params = prior_func(current_hypercube)
    current_logL = log_likelihood(current_params)
    
    # Find good starting point if needed
    if not np.isfinite(current_logL):
        for _ in range(100):
            test_hypercube = np.random.uniform(0.1, 0.9, n_dims)
            test_params = prior_func(test_hypercube)
            test_logL = log_likelihood(test_params)
            if np.isfinite(test_logL):
                current_hypercube = test_hypercube
                current_params = test_params
                current_logL = test_logL
                break
    
    n_accepted = 0
    step_size = 0.01  # Conservative step size
    
    for i in range(n_samples + n_burn):
        # Propose new state
        proposal_hypercube = current_hypercube + np.random.normal(0, step_size, n_dims)
        proposal_hypercube = np.clip(proposal_hypercube, 0.001, 0.999)
        
        proposal_params = prior_func(proposal_hypercube)
        proposal_logL = log_likelihood(proposal_params)
        
        # Accept/reject
        if np.isfinite(proposal_logL):
            if not np.isfinite(current_logL) or np.random.rand() < np.exp(min(0, proposal_logL - current_logL)):
                current_hypercube = proposal_hypercube
                current_params = proposal_params
                current_logL = proposal_logL
                n_accepted += 1
        
        # Store sample after burn-in
        if i >= n_burn:
            samples[i - n_burn] = current_params
    
    acceptance_rate = n_accepted / (n_samples + n_burn)
    print(f"   Acceptance rate: {acceptance_rate:.3f}")
    
    return samples


def compare_results(samples_standard, samples_anomaly, m_true, c_true, sig_true):
    """Print quantitative comparison of parameter estimates."""
    param_names = ['m', 'c', 'σ', 'log p']
    true_values = [m_true, c_true, sig_true, None]
    
    print("\n   Parameter Estimation Summary:")
    print("   " + "=" * 60)
    print(f"   {'Parameter':<10} {'True':<8} {'Standard':<15} {'Anomaly Corr':<15}")
    print("   " + "-" * 60)
    
    for i, (name, true_val) in enumerate(zip(param_names, true_values)):
        std_mean = np.mean(samples_standard[:, i])
        std_std = np.std(samples_standard[:, i])
        anom_mean = np.mean(samples_anomaly[:, i])
        anom_std = np.std(samples_anomaly[:, i])
        
        if i < 3:  # For original parameters, show comparison to true value
            print(f"   {name:<10} {true_val:<8.1f} {std_mean:<7.2f}±{std_std:<6.2f} {anom_mean:<7.2f}±{anom_std:<6.2f}")
        else:  # For logp, don't show "true" value since it's now free
            print(f"   {name:<10} {'free':<8} {std_mean:<7.2f}±{std_std:<6.2f} {anom_mean:<7.2f}±{anom_std:<6.2f}")


def create_visualisations(x, y, samples_standard, samples_anomaly, m_true, c_true):
    """Create comprehensive visualisations of the results."""
    param_names = ['m', 'c', 'σ', 'log p']
    true_values = [m_true, c_true, 5, -2.5]
    
    # 1. Parameter comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(4):
        ax = axes[i]
        
        # Plot histograms
        ax.hist(samples_standard[:, i], bins=50, alpha=0.6, 
               label='Standard', density=True, color='red')
        ax.hist(samples_anomaly[:, i], bins=50, alpha=0.6, 
               label='Anomaly Corrected', density=True, color='blue')
        
        # Add true value line for first 3 parameters
        if i < 3:
            ax.axvline(true_values[i], color='black', linestyle='--', 
                      linewidth=2, label='True Value')
        
        ax.set_xlabel(param_names[i], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Parameter: {param_names[i]}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.suptitle('Parameter Estimation Comparison\n(logp as free parameter)', 
                 fontsize=16, y=1.02)
    plt.savefig('parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Model fitting comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get parameter estimates
    m_std = np.mean(samples_standard[:, 0])
    c_std = np.mean(samples_standard[:, 1])
    m_anom = np.mean(samples_anomaly[:, 0])
    c_anom = np.mean(samples_anomaly[:, 1])
    
    x_model = np.linspace(0, 25, 100)
    
    # Plot 1: Standard method
    ax1.plot(x, y, "ro", markersize=8, alpha=0.7, label='Data with anomalies')
    ax1.plot(x[10], y[10], 'ro', markersize=12, markeredgecolor='black', 
             markeredgewidth=2)
    ax1.plot(x[15], y[15], 'ro', markersize=12, markeredgecolor='black', 
             markeredgewidth=2)
    ax1.plot(x_model, m_true * x_model + c_true, 'k--', alpha=0.7, linewidth=2, label='True model')
    ax1.plot(x_model, m_std * x_model + c_std, 'r-', linewidth=3, 
             label=f'Fitted: y = {m_std:.2f}x + {c_std:.1f}')
    ax1.set_title("Standard Likelihood", fontsize=14)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anomaly corrected
    ax2.plot(x, y, "ro", markersize=8, alpha=0.7, label='Data with anomalies')
    ax2.plot(x[10], y[10], 'ro', markersize=12, markeredgecolor='black', 
             markeredgewidth=2, label='Anomalous points')
    ax2.plot(x[15], y[15], 'ro', markersize=12, markeredgecolor='black', 
             markeredgewidth=2)
    ax2.plot(x_model, m_true * x_model + c_true, 'k--', alpha=0.7, linewidth=2, label='True model')
    ax2.plot(x_model, m_anom * x_model + c_anom, 'b-', linewidth=3, 
             label=f'Fitted: y = {m_anom:.2f}x + {c_anom:.1f}')
    ax2.set_title("Anomaly-Corrected Likelihood", fontsize=14)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Model Fitting Comparison', fontsize=16, y=1.02)
    plt.savefig('model_fitting_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Full corner plot
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: 1D histograms
                ax.hist(samples_standard[:, i], bins=50, alpha=0.6, 
                       label='Standard', density=True, color='red')
                ax.hist(samples_anomaly[:, i], bins=50, alpha=0.6, 
                       label='Anomaly Corrected', density=True, color='blue')
                if i < 3:
                    ax.axvline(true_values[i], color='black', linestyle='--', label='True Value')
                ax.set_xlabel(param_names[i])
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
                    
            elif i > j:
                # Lower triangle: 2D scatter plots
                ax.scatter(samples_standard[:, j], samples_standard[:, i], 
                          alpha=0.3, s=0.5, color='red', label='Standard')
                ax.scatter(samples_anomaly[:, j], samples_anomaly[:, i], 
                          alpha=0.4, s=0.5, color='blue', label='Anomaly Corrected')
                if i < 3 and j < 3:
                    ax.axhline(true_values[i], color='black', linestyle='--', alpha=0.7)
                    ax.axvline(true_values[j], color='black', linestyle='--', alpha=0.7)
                ax.set_xlabel(param_names[j])
                ax.set_ylabel(param_names[i])
                ax.grid(True, alpha=0.3)
                
            else:
                # Upper triangle: remove
                ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle('MCMC Samples: Parameter Correlations and Marginals\n(logp as free parameter)', 
                 fontsize=16, y=1.02)
    plt.savefig('corner_plot.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()