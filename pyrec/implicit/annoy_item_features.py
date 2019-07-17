import itertools
import annoy
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

from pyrec import ImplicitAnnoyRecommender


class ImplicitAnnoyItemFeatureRecommender(ImplicitAnnoyRecommender):
    def __init__(self, als_model: AlternatingLeastSquares,
                 recommend_index: annoy.AnnoyIndex, max_norm: float, user_labels: np.ndarray, item_labels: np.ndarray,
                 tag_tfidf_transformer: TfidfTransformer, tag_lookup: dict, item_embedding_weight: np.ndarray):
        super(ImplicitAnnoyItemFeatureRecommender, self).__init__(als_model=als_model, recommend_index=recommend_index,
                                                                  max_norm=max_norm, user_labels=user_labels,
                                                                  item_labels=item_labels)
        self.tag_tfidf_transformer = tag_tfidf_transformer
        self.tag_lookup = tag_lookup
        self.tag_count = len(tag_lookup)
        self.item_count = len(item_labels)
        self.item_embedding_weight = item_embedding_weight

    def __recommend_internal__(self, user_id, user_items, N=10, filter_items=None, recalculate_user=True,
                               filter_already_liked_items=True, search_k=2000000, user_tags=None):
        if user_items is not None and len(user_items) > 0:
            return super(ImplicitAnnoyItemFeatureRecommender, self).__recommend_internal__(user_id, user_items, N,
                                                                                           filter_items,
                                                                                           recalculate_user,
                                                                                           filter_already_liked_items,
                                                                                           search_k)
        else:
            user = self.__represent_user_by_tags(user_tags)

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

    def __represent_user_by_tags(self, user_tags: dict):
        tag_count_vec = np.zeros(self.tag_count)
        for q, v in user_tags:
            if q in self.tag_lookup:
                tag_label = self.tag_lookup[q]
                tag_count_vec[tag_label] += v
        if tag_count_vec.sum() == 0:
            return None
        tag_tfidf_vec = self.tag_tfidf_transformer.transform(tag_count_vec.reshape(1, -1))
        item_vec = sparse.hstack((sparse.csr_matrix(np.zeros(self.item_count)), tag_tfidf_vec), format='csr', dtype=np.float32)
        return item_vec*self.item_embedding_weight

