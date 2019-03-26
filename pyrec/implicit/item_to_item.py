import logging

import numpy as np
from implicit.nearest_neighbours import BM25Recommender
from scipy import sparse

log = logging.getLogger("rb.recommendation")


class RecommenderException(Exception):
    pass

class BM25Transformer:
    def __init__(self, N=0, length_norm=None, k1=100, b=0.8):
        self.k1 = k1
        self.b = b
        self.N = N
        self.length_norm = length_norm

    def fit_transform(self, X):
        """ Weighs each row of a sparse matrix X  by BM25 weighting """
        # calculate idf per term (user)
        X = sparse.coo_matrix(X)

        N = float(X.shape[0])
        self.N = N
        idf = np.log(N) - np.log1p(np.bincount(X.col))

         # calculate length_norm per document (artist)
        row_sums = np.ravel(X.sum(axis=1))
        average_length = row_sums.mean()
        length_norm = (1.0 - self.b) + self.b * row_sums / average_length
        self.length_norm = length_norm

         # weight matrix rows by bm25
        X.data = X.data * (self.k1 + 1.0) / (self.k1 * length_norm[X.row] + X.data) * idf[X.col]
        return X

    def transform(self, user_items):
        idf = np.log(self.N) - np.log1p(1)
        return user_items.data * (self.k1 + 1.0) / (self.k1 * self.length_norm[user_items.col] + user_items.data) * idf

class ItemToItemRecommender:

    def __init__(self, i2i_model: BM25Recommender, user_labels: np.ndarray, item_labels: np.ndarray, transformer: BM25Transformer=None):
        self.i2i_model = i2i_model
        self.transformer = transformer
        self.user_labels = user_labels
        self.item_labels = item_labels
        self.user_labels_idx = {idx: label for label, idx in enumerate(user_labels)}
        self.item_labels_idx = {idx: label for label, idx in enumerate(item_labels)}

    def get_item_label(self, item_id):
        return self.item_labels_idx.get(item_id)

    def get_item_id(self, item_label):
        return self.item_labels[item_label]

    def get_user_label(self, user_id):
        return self.user_labels_idx.get(user_id)

    def get_user_id(self, user_label):
        return self.user_labels[user_label]

    def __recommend_internal__(self, user_label, user_items, N=10, filter_already_liked_items=True):
        if self.transformer is not None:
            user_items = self.transformer.transform(user_items)
        return self.i2i_model.recommend(user_label, user_items=user_items, N=N, recalculate_user=True,
                                        filter_already_liked_items=filter_already_liked_items)

    def recommend(self, item_ids, item_weights=None, number_of_results=50, filter_already_liked_items=True):
        """
        Recommend items from a list of items and weights
        :param item_ids:
        :param item_weights:
        :param number_of_results:
        :param filter_already_liked_items:
        :return: a list of tuples (item_id, weight)
        """
        item_lb = [self.get_item_label(i) for i in item_ids]
        user_ll = [0] * len(item_ids)
        confidence = [10] * len(item_ids) if item_weights is None else item_weights
        user_items = sparse.csr_matrix((confidence, (user_ll, item_lb)))
        user_label = 0

        recommendations = self.__recommend_internal__(user_label, user_items=user_items, N=number_of_results,
                                                      filter_already_liked_items=filter_already_liked_items)

        recommendations = [(self.get_item_id(x[0]), x[1]) for x in recommendations]

        return recommendations

    def save(self, base_name, compress=False):
        als_file = base_name + ".npz"
        log.info("Saving item to item bm25 model to %s", als_file)
        data = {
            'model.K': self.i2i_model.K,
            'model.bm25.K1': self.i2i_model.K1,
            'model.bm25.B': self.i2i_model.B,
            'model.similarity': self.i2i_model.similarity,
            'user_labels': self.user_labels,
            'item_labels': self.item_labels,
        }
        if self.transformer is not None:
            data['model.bm25.item_length_norm']=self.transformer.length_norm
        if compress:
            np.savez_compressed(als_file, **data)
        else:
            np.savez(als_file, **data)


def load_recommender(item_to_item_model_file: str) -> ItemToItemRecommender:
    log.info("Loading item to item bm25 model")
    data = np.load(item_to_item_model_file)
    k=data['model.K']
    k1=data['model.bm25.K1']
    b=data['model.bm25.B']
    model = BM25Recommender(K=k, K1=k1, B=b)
    model.similarity = data['model.similarity']
    user_labels = data['user_labels']
    item_labels = data['item_labels']
    bm25_transformer = BM25Transformer(N=item_labels.shape[0], length_norm= data['model.bm25.item_length_norm'], k1=k1, b=b)
    return ItemToItemRecommender(model, user_labels, item_labels, bm25_transformer)
