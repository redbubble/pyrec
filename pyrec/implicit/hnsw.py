import itertools
import logging

import hnswlib
import numpy as np
from implicit.als import AlternatingLeastSquares

from pyrec import ImplicitRecommender

log = logging.getLogger("rb.recommendation")


class ImplicitHNSWRecommender(ImplicitRecommender):

    @staticmethod
    def build_hnsw_recommender(als_model: AlternatingLeastSquares,
                               user_labels: np.ndarray, item_labels: np.ndarray,
                               m: int = 96, ef_construction: int = 1024, space: str = 'l2'):
        log.info("Building hnsw items index")
        num_elements, dim = als_model.item_factors.shape
        data_labels = np.arange(num_elements)
        recommend_index = hnswlib.Index(space=space, dim=dim)
        recommend_index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=m)
        recommend_index.add_items(als_model.item_factors, data_labels)
        return ImplicitHNSWRecommender(als_model, recommend_index=recommend_index, user_labels=user_labels,
                                       item_labels=item_labels)

    def __init__(self, als_model: AlternatingLeastSquares,
                 recommend_index: hnswlib.Index, user_labels: np.ndarray, item_labels: np.ndarray):
        super(ImplicitHNSWRecommender, self).__init__(als_model=als_model, user_labels=user_labels,
                                                      item_labels=item_labels)
        self.recommend_index = recommend_index

    def __recommend_internal__(self, user_id, user_items, N=10, filter_items=None, recalculate_user=True,
                               filter_already_liked_items=True, **kwargs):
        user = self.als_model._user_factor(user_id, user_items, recalculate_user)

        # calculate the top N items, removing the users own liked items from
        # the results
        item_filter = set(filter_items) if filter_items else set()
        if filter_already_liked_items:
            item_filter.update(user_items[user_id].indices)
        count = N + len(item_filter)

        ids, dist = self.recommend_index.index.knn_query(user, k=count)
        ids, dist = ids[0], dist[0]
        return list(itertools.islice((rec for rec in zip(ids, dist) if rec[0] not in item_filter), N))

    def save(self, base_name, user_factors=False, compress=False):
        super(ImplicitHNSWRecommender, self).save(base_name=base_name, user_factors=user_factors, compress=compress)

        annoy_file = base_name + '.hnsw'
        log.info("Saving hnsw index to %s", annoy_file)

        self.recommend_index.save(annoy_file)
