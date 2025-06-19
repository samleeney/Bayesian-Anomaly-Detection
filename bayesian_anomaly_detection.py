#!/usr/bin/env python3
"""
Bayesian Anomaly Detection

Demonstrates Bayesian anomaly detection for robust parameter inference
in the presence of outliers. Implements methodology from:

- Leeney et al. (2022): "Bayesian approach to radio frequency interference mitigation"
- Anstey and Leeney (2023): "Enhanced Bayesian RFI Mitigation and Transient Flagging"

Binary anomaly mask with Bernoulli priors and piecewise likelihood for
automatic outlier detection with principled Bayesian inference.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    print("=== Bayesian Anomaly Detection Demonstration ===\n")
    
    print("1. Generating mock data...")
    x, y, m_true, c_true, sig_true = generate_data()
    
    print("2. Adding anomalies to data...")
    y = add_anomalies(y)
    
    print("3. Plotting data...")
    plot_data(x, y, m_true, c_true)
    
    print("4. Setting up priors and parameters...")
    delta = np.max(y)
    prior_func = create_prior_function()
    
    likelihood_standard = create_standard_likelihood(x, y)
    likelihood_anomaly = create_anomaly_likelihood(x, y, delta)
    likelihood_tracking = create_anomaly_likelihood_with_tracking(x, y, delta)
    
    print("5. Testing likelihood functions...")
    test_likelihoods(likelihood_standard, likelihood_anomaly)
    
    print("6. Fitting with standard likelihood...")
    samples_standard = run_mcmc(likelihood_standard, prior_func, "standard")
    
    print("7. Fitting with anomaly-corrected likelihood...")
    samples_anomaly = run_mcmc(likelihood_anomaly, prior_func, "anomaly-corrected")
    
    print("8. Running MCMC with epsilon mask tracking...")
    samples_tracking, epsilon_masks = run_mcmc_with_tracking(
        likelihood_tracking, prior_func, "anomaly-corrected with tracking", x, n_samples=20000, n_burn=5000
    )
    
    print("9. Comparing results...")
    compare_results(samples_standard, samples_anomaly, m_true, c_true, sig_true)
    
    print("10. Creating visualisations...")
    create_visualisations(x, y, samples_standard, samples_anomaly, m_true, c_true)
    
    print("11. Visualizing posterior fraction of epsilon masks...")
    visualize_posterior_fraction(x, y, epsilon_masks, m_true, c_true)
    
    print("12. Comparison to optimal parameters from clean data...")
    y_clean = y.copy()
    y_clean[10] -= 100
    y_clean[15] -= 100
    A = np.vstack([x, np.ones(len(x))]).T
    m_opt, c_opt = np.linalg.lstsq(A, y_clean, rcond=None)[0]
    residuals = y_clean - (m_opt * x + c_opt)
    sig_opt = np.std(residuals)
    
    print(f"   Optimal parameters from clean data:")
    print(f"   m = {m_opt:.3f}, c = {c_opt:.3f}, σ = {sig_opt:.3f}")
    
    anom_mean = np.mean(samples_tracking, axis=0)
    print(f"   MCMC anomaly-corrected results:")
    print(f"   m = {anom_mean[0]:.3f}, c = {anom_mean[1]:.3f}, σ = {anom_mean[2]:.3f}")
    
    print("\n=== Analysis Complete ===")
    print("Parameter recovery dramatically improved with Bayesian anomaly detection.")
    print("Posterior fraction plot shows frequency of anomaly flagging per data point.")


def generate_data():
    np.random.seed(123)
    N = 25
    x = np.linspace(0, 25, N)
    m_true = 1
    c_true = 1  
    sig_true = 2
    y = m_true * x + c_true + np.random.randn(N) * sig_true
    return x, y, m_true, c_true, sig_true


def add_anomalies(y):
    y_anomalous = y.copy()
    y_anomalous[10] += 100
    y_anomalous[15] += 100
    return y_anomalous


def plot_data(x, y, m_true, c_true):
    y_clean = y.copy()
    y_clean[10] -= 100
    y_clean[15] -= 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(x, y_clean, "o", markersize=8, alpha=0.7, label='Clean data')
    ax1.plot(x, m_true * x + c_true, 'k--', alpha=0.7, label=f'True model: y = {m_true}x + {c_true}')
    ax1.set_title("Original Data", fontsize=16)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x, y, "ro", markersize=8, alpha=0.7, label='Data with anomalies')
    ax2.plot(x, m_true * x + c_true, 'k--', alpha=0.7, label=f'True model: y = {m_true}x + {c_true}')
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
    def uniform_prior(a, b):
        def prior_func(u):
            return a + (b - a) * u
        return prior_func

    def prior(hypercube):
        theta = np.zeros_like(hypercube)
        theta[0] = uniform_prior(-2, 4)(hypercube[0])
        theta[1] = uniform_prior(-20, 20)(hypercube[1])
        theta[2] = uniform_prior(0.1, 20)(hypercube[2])
        theta[3] = uniform_prior(-10, -0.1)(hypercube[3])
        return theta
    
    return prior


def create_standard_likelihood(x, y):
    def likelihood(theta):
        m = theta[0]
        c = theta[1] 
        sig = theta[2]
        sig = np.maximum(sig, 0.1)
        y_pred = m * x + c
        logL = -0.5 * ((y_pred - y) ** 2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)
        return logL.sum()
    return likelihood


def create_anomaly_likelihood(x, y, delta):
    def likelihood(theta):
        m = theta[0]
        c = theta[1]
        sig = theta[2] 
        logp = theta[3]
        sig = np.maximum(sig, 0.1)
        p = np.exp(np.clip(logp, -20, -0.1))
        y_pred = m * x + c
        logL_normal = -0.5 * ((y_pred - y) ** 2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)
        logL_with_prior = logL_normal + np.log(1 - p)
        anomaly_threshold = logp - np.log(delta)
        logL_corrected = np.maximum(logL_with_prior, anomaly_threshold)
        return logL_corrected.sum()
    return likelihood


def create_anomaly_likelihood_with_tracking(x, y, delta):
    def likelihood(theta, return_mask=False):
        m = theta[0]
        c = theta[1]
        sig = theta[2] 
        logp = theta[3]
        sig = np.maximum(sig, 0.1)
        p = np.exp(np.clip(logp, -20, -0.1))
        y_pred = m * x + c
        logL_normal = -0.5 * ((y_pred - y) ** 2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)
        logL_with_prior = logL_normal + np.log(1 - p)
        anomaly_threshold = logp - np.log(delta)
        logL_corrected = np.maximum(logL_with_prior, anomaly_threshold)
        
        epsilon_mask = (logL_with_prior < anomaly_threshold).astype(int) * return_mask
        return (logL_corrected.sum(), epsilon_mask) if return_mask else logL_corrected.sum()
    return likelihood


def test_likelihoods(likelihood_standard, likelihood_anomaly):
    true_params_standard = np.array([1.0, 1.0, 5.0])
    true_params_anomaly = np.array([1.0, 1.0, 5.0, -3.0])
    logL_standard = likelihood_standard(true_params_standard)
    logL_anomaly = likelihood_anomaly(true_params_anomaly)
    print(f"   Standard likelihood at true values: {logL_standard:.2f}")
    print(f"   Anomaly-corrected likelihood at true values: {logL_anomaly:.2f}")


def run_mcmc(log_likelihood, prior_func, method_name, n_samples=5000, n_burn=1000):
    print(f"   Running MCMC for {method_name} method...")
    samples = np.zeros((n_samples, 4))
    current_params = np.array([1.0, 1.0, 2.0, -3.0])
    current_logL = log_likelihood(current_params)
    n_accepted = 0
    step_size = 0.1
    
    for i in range(n_samples + n_burn):
        proposal_params = current_params + np.random.normal(0, step_size, 4)
        proposal_params = np.clip(proposal_params, [-2, -20, 0.1, -10], [4, 20, 20, -0.1])
        proposal_logL = log_likelihood(proposal_params)
        
        accept = np.random.rand() < np.exp(min(0, proposal_logL - current_logL))
        current_params = proposal_params * accept + current_params * (1 - accept)
        current_logL = proposal_logL * accept + current_logL * (1 - accept)
        n_accepted += accept
        
        samples[i - n_burn] = current_params * (i >= n_burn)
    
    print(f"   Acceptance rate: {n_accepted / (n_samples + n_burn):.3f}")
    return samples[samples.any(axis=1)]


def run_mcmc_with_tracking(log_likelihood, prior_func, method_name, x, n_samples=5000, n_burn=1000):
    print(f"   Running MCMC with tracking for {method_name} method...")
    N = len(x)
    samples = np.zeros((n_samples, 4))
    epsilon_masks = np.zeros((n_samples, N))
    current_params = np.array([1.0, 1.0, 2.0, -3.0])
    current_logL, current_mask = log_likelihood(current_params, return_mask=True)
    n_accepted = 0
    step_size = 0.1
    
    for i in range(n_samples + n_burn):
        proposal_params = current_params + np.random.normal(0, step_size, 4)
        proposal_params = np.clip(proposal_params, [-2, -20, 0.1, -10], [4, 20, 20, -0.1])
        proposal_logL, proposal_mask = log_likelihood(proposal_params, return_mask=True)
        
        accept = np.random.rand() < np.exp(min(0, proposal_logL - current_logL))
        current_params = proposal_params * accept + current_params * (1 - accept)
        current_logL = proposal_logL * accept + current_logL * (1 - accept)
        current_mask = proposal_mask * accept + current_mask * (1 - accept)
        n_accepted += accept
        
        samples[i - n_burn] = current_params * (i >= n_burn)
        epsilon_masks[i - n_burn] = current_mask * (i >= n_burn)
    
    print(f"   Acceptance rate: {n_accepted / (n_samples + n_burn):.3f}")
    return samples[samples.any(axis=1)], epsilon_masks[epsilon_masks.any(axis=1)]


def compare_results(samples_standard, samples_anomaly, m_true, c_true, sig_true):
    param_names = ['m', 'c', 'σ', 'log p']
    true_values = [m_true, c_true, sig_true, 'free']
    
    print("\n   Parameter Estimation Summary:")
    print("   " + "=" * 60)
    print(f"   {'Parameter':<10} {'True':<8} {'Standard':<15} {'Anomaly Corr':<15}")
    print("   " + "-" * 60)
    
    for i, (name, true_val) in enumerate(zip(param_names, true_values)):
        std_mean = np.mean(samples_standard[:, i])
        std_std = np.std(samples_standard[:, i])
        anom_mean = np.mean(samples_anomaly[:, i])
        anom_std = np.std(samples_anomaly[:, i])
        print(f"   {name:<10} {true_val:<8} {std_mean:<7.2f}±{std_std:<6.2f} {anom_mean:<7.2f}±{anom_std:<6.2f}")


def visualize_posterior_fraction(x, y, epsilon_masks, m_true, c_true):
    posterior_fraction = np.mean(epsilon_masks, axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    scatter = ax1.scatter(x, y, c=posterior_fraction, s=100, cmap='RdYlBu_r', 
                         edgecolors='black', linewidth=1, vmin=0, vmax=1)
    ax1.plot(x, m_true * x + c_true, 'k--', alpha=0.7, linewidth=2, label='True model')
    ax1.scatter(x[10], y[10], s=200, facecolors='none', edgecolors='red', 
               linewidth=3, label='True anomalies')
    ax1.scatter(x[15], y[15], s=200, facecolors='none', edgecolors='red', 
               linewidth=3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Posterior Mask on Epsilon', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Posterior Anomaly Fraction', fontsize=12)
    
    scatter2 = ax2.scatter(x, posterior_fraction, c=posterior_fraction, s=100, 
                          cmap='RdYlBu_r', edgecolors='black', linewidth=1, vmin=0, vmax=1)
    ax2.scatter(x[10], posterior_fraction[10], s=200, facecolors='none', 
               edgecolors='red', linewidth=3)
    ax2.scatter(x[15], posterior_fraction[15], s=200, facecolors='none', 
               edgecolors='red', linewidth=3)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Posterior Fraction', fontsize=12)
    ax2.set_title('Posterior Mask on Epsilon', fontsize=14)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('posterior_fraction_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n   Posterior Anomaly Fractions:")
    print("   " + "=" * 50)
    print(f"   {'Data Point':<15} {'x value':<10} {'y value':<10} {'Posterior Fraction':<20}")
    print("   " + "-" * 50)
    
    sorted_indices = np.argsort(posterior_fraction)[::-1]
    for idx in sorted_indices[:10]:
        print(f"   Point {idx:<13} {x[idx]:<10.1f} {y[idx]:<10.1f} {posterior_fraction[idx]:<20.3f}")
    
    print(f"\n   True anomalies:")
    print(f"   Point 10: posterior fraction = {posterior_fraction[10]:.3f}")
    print(f"   Point 15: posterior fraction = {posterior_fraction[15]:.3f}")


def create_visualisations(x, y, samples_standard, samples_anomaly, m_true, c_true):
    param_names = ['m', 'c', 'σ', 'log p']
    true_values = [m_true, c_true, 2, -2.5]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(4):
        ax = axes[i]
        ax.hist(samples_standard[:, i], bins=50, alpha=0.6, 
               label='Standard', density=True, color='red')
        ax.hist(samples_anomaly[:, i], bins=50, alpha=0.6, 
               label='Anomaly Corrected', density=True, color='blue')
        ax.set_xlabel(param_names[i], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Parameter: {param_names[i]}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Parameter Estimation Comparison', fontsize=16, y=1.02)
    plt.savefig('parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    m_std = np.mean(samples_standard[:, 0])
    c_std = np.mean(samples_standard[:, 1])
    m_anom = np.mean(samples_anomaly[:, 0])
    c_anom = np.mean(samples_anomaly[:, 1])
    
    x_model = np.linspace(0, 25, 100)
    
    ax1.plot(x, y, "ro", markersize=8, alpha=0.7, label='Data with anomalies')
    ax1.plot(x[10], y[10], 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
    ax1.plot(x[15], y[15], 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
    ax1.plot(x_model, m_true * x_model + c_true, 'k--', alpha=0.7, linewidth=2, label='True model')
    ax1.plot(x_model, m_std * x_model + c_std, 'r-', linewidth=3, 
             label=f'Fitted: y = {m_std:.2f}x + {c_std:.1f}')
    ax1.set_title("Standard Likelihood", fontsize=14)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x, y, "ro", markersize=8, alpha=0.7, label='Data with anomalies')
    ax2.plot(x[10], y[10], 'ro', markersize=12, markeredgecolor='black', 
             markeredgewidth=2, label='Anomalous points')
    ax2.plot(x[15], y[15], 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
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
    


if __name__ == "__main__":
    main()