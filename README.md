

# Experiments with Diffusion Models on MNIST Dataset

## Introduction

This project explores the application of diffusion models on the MNIST dataset. We conducted a series of experiments to evaluate the impact of various parameters, including noise schedules, diffusion time steps, and sampling methods, on the model's performance. This README provides an overview of the experimental setup, methodology, and key findings.

## Methodology

### Key Variables

1. **Noise Schedules**: Tested four types - `linear`, `cosine`, `quadratic`, and `sigmoid`.
2. **Diffusion Time Steps & beta_end**:
   - 200 steps with a beta_end of 0.08
   - 1000 steps with a beta_end of 0.02
3. **Sampling Methods**: Evaluated both DDPM and DDIM methods.

### Additional Fixed Parameters

- **Batch Size**: 128
- **Number of Epochs**: 20
- **Beta Start**: 0.0001

### Performance Metric

- **Fréchet Inception Distance (FID)**: Used to assess model performance, with lower scores indicating better performance.

### Experimental Setup

In total, 16 experiments were conducted. The best FID score over the 20 epochs for each configuration was selected to represent the outcome of that setup.

## Results

### Noise Schedule Variations

- **Quadratic**: Lowest average FID score.
- **Cosine**: Best single FID score.
- **Linear** and **Sigmoid**: Higher FID scores compared to `quadratic` and `cosine`.

### Diffusion Time Steps & beta_end

- Models with 200 diffusion time steps (beta_end = 0.08) outperformed those with 1000 steps (beta_end = 0.02).
- DDPM performed better with the (time step = 200, beta_end = 0.08) configuration, showing significant improvement.

### Sampling Method Comparison

- **DDPM**: Outperformed DDIM in terms of average and best FID scores, despite being slower.

### Best Sampling Result Configuration

- **Time Steps**: 200
- **Beta End**: 0.08
- **Beta Start**: 0.0001
- **Noise Schedule**: Cosine
- **Sampling Method**: DDPM

## Conclusion

The choice of noise schedule and diffusion steps significantly impacts image quality in diffusion models. The `cosine` noise schedule and 200 diffusion steps were found to be the most effective, with the DDPM sampling method outperforming DDIM. Careful tuning of these parameters is crucial for optimizing diffusion models for high-fidelity image generation tasks.

## Usage

### Prerequisites

- Python 3.8+
- Dependencies specified in `environment.yml`

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/diffusion-models-mnist.git
   cd diffusion-models-mnist
   ```

2. Create and activate the conda environment:
   ```sh
   conda env create -f environment.yml
   conda activate diffusion-env
   ```

### Running the Experiments

To run the experiments, execute the `main.py` script:
```sh
python main.py
```

The script will log the training process and save the results in `training.log` and `grid_search_results.json`.

## File Structure

- `main.py`: Script to run the diffusion model experiments.
- `environment.yml`: Conda environment file with dependencies.
- `training.log`: Log file containing training details.
- `grid_search_results.json`: JSON file with the results of the grid search experiments.

