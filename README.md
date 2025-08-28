# Deep-SSVFEP-BCI

A Python implementation of the deep neural network for SSVEP-based brain-computer interfaces, based on the paper "A Deep Neural Network for SSVEP-Based Brain-Computer Interfaces" by O. B. Guney, M. Oblokulov and H. Ozkan.

## Project Structure

- `data/`: Data preprocessing and loading modules
- `models/`: Deep learning model implementations
- `utils/`: Utility functions and helpers
- `results/`: Training results and evaluation metrics
- `tests/`: Unit tests for various components

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- SciPy
- Matplotlib
- Scikit-learn

## Usage

1. Install requirements: `pip install -r requirements.txt`
2. Prepare your SSVEP dataset
3. Run the main script: `python main.py`

## References

1. O. B. Guney, M. Oblokulov and H. Ozkan, "A Deep Neural Network for SSVEP-Based Brain-Computer Interfaces," IEEE Transactions on Biomedical Engineering, vol. 69, no. 2, pp. 932-944, 2022.