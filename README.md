# pyrec
Library to train and serve recommendation models in Python

## Installation
```bash
pip install git+https://github.com/redbubble/pyrec.git
```

## Usage
### Implicit Library Wrapper
```python
from pyrec import load_recommender

# Load a recommender from file
recommender = load_recommender(als_model_file='als_model.npz', index_file='annoy_index.ann')
recommender.recommend(item_ids=[28418326, 15779237, 11422387], item_weights=[10,20,3], number_of_results=5)
```