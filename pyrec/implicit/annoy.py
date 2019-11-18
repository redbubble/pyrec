import itertools
import logging

import annoy
import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import augment_inner_product_matrix

from pyrec import ImplicitRecommender

log = logging.getLogger("rb.recommendation")


class ImplicitAnnoyRecommender(ImplicitRecommender):

    @staticmethod
    def build_annoy_recommender(als_model: AlternatingLeastSquares,
                                user_labels: np.ndarray, item_labels: np.ndarray,
                                approximate_similar_items=True, approximate_recommend=True,
                                n_trees: int = 1000):
        # build up an Annoy Index with all the item_factors (for calculating similar items)
        if approximate_similar_items:
            log.info("Building annoy similar items index")

            similar_items_index = annoy.AnnoyIndex(
                als_model.item_factors.shape[1], 'angular')
            for i, row in enumerate(als_model.item_factors):
                similar_items_index.add_item(i, row)
            similar_items_index.build(n_trees)

        # build up a separate index for the inner product (for recommend methods)
        if approximate_recommend:
            log.info("Building annoy recommendation index")
            max_norm, extra = augment_inner_product_matrix(als_model.item_factors)
            recommend_index = annoy.AnnoyIndex(extra.shape[1], 'angular')
            for i, row in enumerate(extra):
                recommend_index.add_item(i, row)
            recommend_index.build(n_trees)

        return ImplicitAnnoyRecommender(als_model, recommend_index=recommend_index, max_norm=max_norm,
                                        user_labels=user_labels, item_labels=item_labels)

    def __init__(self, als_model: AlternatingLeastSquares,
                 recommend_index: annoy.AnnoyIndex, max_norm: float, user_labels: np.ndarray, item_labels: np.ndarray):
        super(ImplicitAnnoyRecommender, self).__init__(als_model=als_model, user_labels=user_labels,
                                                       item_labels=item_labels)
        self.recommend_index = recommend_index
        self.max_norm = max_norm

    def __recommend_internal__(self, user_id, user_items, N=10, filter_items=None, recalculate_user=True,
                               filter_already_liked_items=True, search_k=2000000, **kwargs):
        user = self.als_model._user_factor(user_id, user_items, recalculate_user)

        # calculate the top N items, removing the users own liked items from
        # the results
        item_filter = set(filter_items) if filter_items else set()
        if filter_already_liked_items:
            item_filter.update(user_items[user_id].indices)
        count = N + len(item_filter)

        query = np.append(user, 0)
        ids, dist = self.recommend_index.get_nns_by_vector(query, count, include_distances=True,
                                                           search_k=search_k)

        # convert the distances from euclidean to cosine distance,
        # and then rescale the cosine distance to go back to inner product
        scaling = self.max_norm * np.linalg.norm(query)
        dist = scaling * (1 - (np.array(dist) ** 2) / 2)
        return list(itertools.islice((rec for rec in zip(ids, dist) if rec[0] not in item_filter), N))

    def save(self, base_name, user_factors=False, compress=False):
        super(ImplicitAnnoyRecommender, self).save(base_name=base_name, user_factors=user_factors, compress=compress)

        annoy_file = base_name + '.ann'
        log.info("Saving annoy index to %s", annoy_file)

        self.recommend_index.save(annoy_file)
        return annoy_file
