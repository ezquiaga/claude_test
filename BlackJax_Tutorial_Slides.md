# BlackJax Tutorial
## Bayesian Inference in JAX

---

## What is BlackJax?

**BlackJax** is a library for sampling and inference in JAX

- 🚀 High-performance Bayesian inference
- 🔧 Built on JAX (GPU/TPU support)
- 📊 Multiple sampling algorithms (NUTS, HMC, MALA, etc.)
- 🎯 Functional and composable design
- 🔄 Automatic differentiation

**Installation:**
```bash
pip install blackjax
```

---

## Why BlackJax?

### Advantages
- **Fast**: JIT compilation with JAX
- **Flexible**: Easy to customize algorithms
- **Scalable**: GPU/TPU acceleration
- **Modular**: Mix and match components
- **Interoperable**: Works with NumPyro, Flax, Haiku

### Use Cases
- Bayesian parameter estimation
- Probabilistic programming
- Scientific inference
- Machine learning with uncertainty

---

## Quick Start: Basic Example

```python
import jax
import jax.numpy as jnp
import blackjax

# Define your log probability function
def logprob_fn(x):
    # Example: 1D Gaussian
    return -0.5 * jnp.sum(x**2)

# Initial position
initial_position = jnp.array([1.0])

# Random key
rng_key = jax.random.PRNGKey(0)
```

---

## Step 1: Choose an Algorithm

BlackJax offers multiple samplers:

```python
# NUTS (No-U-Turn Sampler) - Recommended for most cases
nuts = blackjax.nuts(logprob_fn, step_size=0.1)

# Or other algorithms:
# hmc = blackjax.hmc(logprob_fn, step_size=0.1, num_integration_steps=10)
# mala = blackjax.mala(logprob_fn, step_size=0.1)
# rwm = blackjax.random_walk_metropolis(logprob_fn)
```

**Popular Algorithms:**
- `nuts`: Adaptive HMC (best for complex posteriors)
- `hmc`: Hamiltonian Monte Carlo
- `mala`: Metropolis-Adjusted Langevin Algorithm
- `rwm`: Random Walk Metropolis

---

## Step 2: Initialize the Sampler

```python
# Initialize the sampler state
rng_key, init_key = jax.random.split(rng_key)

initial_state = nuts.init(initial_position)

print(f"Initial position: {initial_state.position}")
```

**State contains:**
- Current position
- Momentum (for HMC/NUTS)
- Acceptance rate
- Other algorithm-specific info

---

## Step 3: Run the Sampler

```python
# Single step
rng_key, step_key = jax.random.split(rng_key)
new_state, info = nuts.step(step_key, initial_state)

# Multiple steps with a loop
def inference_loop(rng_key, initial_state, num_samples):
    keys = jax.random.split(rng_key, num_samples)
    
    def one_step(state, key):
        new_state, info = nuts.step(key, state)
        return new_state, (new_state.position, info)
    
    final_state, (positions, infos) = jax.lax.scan(
        one_step, initial_state, keys
    )
    
    return positions, infos

# Run inference
num_samples = 1000
positions, infos = inference_loop(rng_key, initial_state, num_samples)
```

---

## Step 4: Analyze Results

```python
import matplotlib.pyplot as plt

# Plot trace
plt.figure(figsize=(10, 4))
plt.plot(positions)
plt.xlabel('Iteration')
plt.ylabel('Parameter value')
plt.title('MCMC Trace')
plt.show()

# Compute statistics
mean_estimate = jnp.mean(positions[500:])  # Discard burn-in
std_estimate = jnp.std(positions[500:])

print(f"Posterior mean: {mean_estimate:.3f}")
print(f"Posterior std: {std_estimate:.3f}")
```

---

## Advanced: Window Adaptation

Automatically tune step size and mass matrix:

```python
from blackjax.adaptation import window_adaptation

# Setup window adaptation for NUTS
adapt = window_adaptation(blackjax.nuts, logprob_fn)

# Run adaptation
rng_key, adapt_key = jax.random.split(rng_key)
(final_state, parameters), _ = adapt.run(
    adapt_key,
    initial_position,
    num_steps=1000
)

# Use adapted parameters
adapted_nuts = blackjax.nuts(
    logprob_fn,
    step_size=parameters['step_size'],
    inverse_mass_matrix=parameters['inverse_mass_matrix']
)
```

---

## Multi-Dimensional Problems

```python
# Define a 2D Gaussian
def logprob_2d(x):
    mu = jnp.array([0.0, 0.0])
    sigma = jnp.array([[1.0, 0.5], 
                       [0.5, 1.0]])
    diff = x - mu
    return -0.5 * diff @ jnp.linalg.inv(sigma) @ diff

# Initial position (2D)
initial_position = jnp.array([1.0, 1.0])

# Same workflow as before
nuts = blackjax.nuts(logprob_2d, step_size=0.1)
initial_state = nuts.init(initial_position)

# Run sampling...
```

---

## Real Example: Linear Regression

```python
# Generate synthetic data
true_slope = 2.0
true_intercept = 1.0
x_data = jnp.linspace(0, 10, 50)
y_data = true_slope * x_data + true_intercept + \
         jax.random.normal(jax.random.PRNGKey(0), (50,)) * 0.5

# Define log probability
def logprob_regression(params):
    slope, intercept, log_sigma = params
    sigma = jnp.exp(log_sigma)
    
    # Likelihood
    y_pred = slope * x_data + intercept
    log_likelihood = jnp.sum(
        -0.5 * ((y_data - y_pred) / sigma)**2 
        - jnp.log(sigma)
    )
    
    # Priors (Gaussian)
    log_prior = (-0.5 * slope**2 - 0.5 * intercept**2 
                 - 0.5 * log_sigma**2)
    
    return log_likelihood + log_prior
```

---

## Linear Regression: Inference

```python
# Initial parameters [slope, intercept, log_sigma]
initial_params = jnp.array([0.0, 0.0, 0.0])

# Setup and run NUTS with adaptation
adapt = window_adaptation(blackjax.nuts, logprob_regression)

rng_key = jax.random.PRNGKey(42)
(final_state, parameters), _ = adapt.run(
    rng_key, initial_params, num_steps=1000
)

# Sample from posterior
nuts = blackjax.nuts(
    logprob_regression,
    step_size=parameters['step_size'],
    inverse_mass_matrix=parameters['inverse_mass_matrix']
)

rng_key, sample_key = jax.random.split(rng_key)
samples, _ = inference_loop(sample_key, final_state, 2000)

# Extract parameters
slope_samples = samples[:, 0]
intercept_samples = samples[:, 1]
```

---

## Vectorized Sampling (Multiple Chains)

```python
# Run multiple chains in parallel
num_chains = 4

# Vectorized initialization
rng_keys = jax.random.split(rng_key, num_chains)
initial_positions = jax.random.normal(rng_keys[0], (num_chains, 3))

# Vectorized step function
vmapped_step = jax.vmap(
    lambda key, state: nuts.step(key, state),
    in_axes=(0, 0)
)

# Initialize all chains
initial_states = jax.vmap(nuts.init)(initial_positions)

# Run chains in parallel
def multi_chain_step(states, key):
    keys = jax.random.split(key, num_chains)
    new_states, infos = vmapped_step(keys, states)
    return new_states, (new_states.position, infos)

keys = jax.random.split(rng_key, 1000)
_, (all_samples, _) = jax.lax.scan(
    multi_chain_step, initial_states, keys
)
```

---

## Integration with NumPyro

BlackJax works well with NumPyro models:

```python
import numpyro
import numpyro.distributions as dist

# Define NumPyro model
def numpyro_model(x, y=None):
    slope = numpyro.sample('slope', dist.Normal(0, 10))
    intercept = numpyro.sample('intercept', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    
    mu = slope * x + intercept
    numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

# Convert to BlackJax
from numpyro.infer.util import initialize_model

init_params, potential_fn, *_ = initialize_model(
    jax.random.PRNGKey(0),
    numpyro_model,
    model_args=(x_data,),
    model_kwargs={'y': y_data}
)

# Use potential_fn as logprob for BlackJax
logprob_fn = lambda x: -potential_fn(x)
```

---

## Performance Tips

### 1. JIT Compilation
```python
# JIT the inference loop for speed
inference_loop_jit = jax.jit(inference_loop, static_argnums=(2,))
```

### 2. GPU Acceleration
```python
# Automatically uses GPU if available
# Check with:
print(jax.devices())
```

### 3. Batching
```python
# Use vmap for multiple chains
# Process data in batches for large datasets
```

### 4. Step Size Tuning
```python
# Use window_adaptation for automatic tuning
# Or manually adjust based on acceptance rate
# Target ~65% for NUTS, ~23% for HMC
```

---

## Common Algorithms Summary

| Algorithm | When to Use | Key Parameters |
|-----------|------------|----------------|
| **NUTS** | Default choice, complex posteriors | `step_size` |
| **HMC** | Known structure, tuning control | `step_size`, `num_integration_steps` |
| **MALA** | High dimensions, gradient available | `step_size` |
| **RWM** | Simple problems, no gradients | `step_size` |

---

## Diagnostics

```python
# Check convergence
def compute_rhat(chains):
    """Gelman-Rubin statistic"""
    # Implementation...
    pass

# Effective sample size
def compute_ess(samples):
    """Effective sample size"""
    # Implementation...
    pass

# Acceptance rate
acceptance_rate = jnp.mean(infos.acceptance_probability)
print(f"Acceptance rate: {acceptance_rate:.2%}")

# Trace plots, autocorrelation, etc.
```

---

## Best Practices

1. **Start simple**: Test on small problems first
2. **Use adaptation**: Let BlackJax tune parameters
3. **Multiple chains**: Run 4+ chains to check convergence
4. **Burn-in**: Discard initial samples (e.g., first 50%)
5. **Check diagnostics**: R-hat, ESS, trace plots
6. **Profile code**: Use JAX profiler for large problems
7. **Batch data**: For big datasets, use mini-batching

---

## Resources

**Documentation:**
- [Official Docs](https://blackjax-devs.github.io/blackjax/)
- [GitHub](https://github.com/blackjax-devs/blackjax)
- [Examples](https://blackjax-devs.github.io/blackjax/examples/)

**Related Libraries:**
- NumPyro: Probabilistic programming
- JAX: Autodiff and JIT
- Optax: Optimization
- TensorFlow Probability: Alternative

**Papers:**
- Hoffman & Gelman (2014): The No-U-Turn Sampler
- Neal (2011): MCMC using Hamiltonian dynamics

---

## Quick Reference Card

```python
# Standard workflow
import blackjax
import jax

# 1. Define log probability
def logprob(x):
    return -0.5 * jnp.sum(x**2)

# 2. Setup sampler with adaptation
adapt = blackjax.window_adaptation(blackjax.nuts, logprob)
(state, params), _ = adapt.run(rng_key, init_pos, num_steps=1000)

# 3. Create adapted sampler
sampler = blackjax.nuts(logprob, **params)

# 4. Run inference
def step(state, key):
    return sampler.step(key, state)

keys = jax.random.split(rng_key, 1000)
_, samples = jax.lax.scan(step, state, keys)

# 5. Analyze
posterior_mean = jnp.mean(samples.position, axis=0)
```

---

## Questions?

**Thank you!**

For more examples and advanced usage:
- Check the official documentation
- Explore the examples gallery
- Join the community discussions

**Key Takeaway**: BlackJax makes Bayesian inference fast, flexible, and fun! 🚀

---
