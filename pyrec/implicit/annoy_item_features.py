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
                               filter_already_liked_items=True, search_k=2000000, item_feature_vec=None):
        if user_items is not None and user_items.shape[0] > 0:
            return super(ImplicitAnnoyItemFeatureRecommender, self).__recommend_internal__(user_id, user_items, N,
                                                                                           filter_items,
                                                                                           recalculate_user,
                                                                                           filter_already_liked_items,
                                                                                           search_k)
        else:
            user = self.__represent_user_by_tags__(item_feature_vec)
            count = N
            query = np.append(user, 0)
            ids, dist = self.recommend_index.get_nns_by_vector(query, count, include_distances=True,
                                                               search_k=search_k)

            return list(itertools.islice((rec for rec in zip(ids, dist)), N))

    def __represent_user_by_tags__(self, item_feature_vec: np.array):
        return item_feature_vec*self.item_embedding_weight

