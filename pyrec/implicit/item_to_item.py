import logging

import numpy as np
from implicit.nearest_neighbours import BM25Recommender, NearestNeighboursScorer
from scipy import sparse

log = logging.getLogger("rb.recommendation")


class RecommenderException(Exception):
    pass

class ItemToItemRecommender:

    def __init__(self, i2i_model: BM25Recommender, user_labels: np.ndarray, item_labels: np.ndarray):
        self.i2i_model = i2i_model
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
        i2i_file = base_name + ".npz"
        log.info("Saving item to item bm25 model to %s", i2i_file)
        data = {
            'model.K': np.array([self.i2i_model.K]),
            'model.bm25.K1': np.array([self.i2i_model.K1]),
            'model.bm25.B': np.array([self.i2i_model.B]),
            'model.similarity.data': self.i2i_model.similarity.data,
            'model.similarity.indices': self.i2i_model.similarity.indices,
            'model.similarity.indptr': self.i2i_model.similarity.indptr,
            'model.similarity.shape': self.i2i_model.similarity.shape,
            'user_labels': self.user_labels,
            'item_labels': self.item_labels,
        }
        if compress:
            np.savez_compressed(i2i_file, **data)
        else:
            np.savez(i2i_file, **data)


def load_recommender(item_to_item_model_file: str) -> ItemToItemRecommender:
    log.info("Loading item to item bm25 model")
    data = np.load(item_to_item_model_file)
    k = data['model.K'][0]
    k1 = data['model.bm25.K1'][0]
    b = data['model.bm25.B'][0]
    model = BM25Recommender(K=k, K1=k1, B=b)
    model.similarity = sparse.csr_matrix(
        (data['model.similarity.data'], data['model.similarity.indices'], data['model.similarity.indptr']),
        shape=data['model.similarity.shape'])
    model.scorer = NearestNeighboursScorer(model.similarity)
    user_labels = data['user_labels']
    item_labels = data['item_labels']
    return ItemToItemRecommender(model, user_labels, item_labels)
