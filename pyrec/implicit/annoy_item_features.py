import annoy
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

from pyrec import ImplicitAnnoyRecommender


class ImplicitAnnoyItemFeatureRecommender(ImplicitAnnoyRecommender):
    def __init__(self, als_model: AlternatingLeastSquares,
                 recommend_index: annoy.AnnoyIndex, max_norm: float, user_labels: np.ndarray, item_labels: np.ndarray,
                 tag_tfidf_transformer: TfidfTransformer, tag_lookup: dict, top_sales_work_labels: np.ndarray):
        super(ImplicitAnnoyItemFeatureRecommender, self).__init__(als_model=als_model, recommend_index=recommend_index,
                                                                  max_norm=max_norm, user_labels=user_labels,
                                                                  item_labels=item_labels)
        self.tag_tfidf_transformer = tag_tfidf_transformer
        self.tag_lookup = tag_lookup
        self.tag_count = len(tag_lookup)
        self.item_count = len(item_labels)
        self.top_sales_work_labels = top_sales_work_labels

    def __recommend_internal__(self, user_label, user_items, N=10, filter_items=None, recalculate_user=True,
                               filter_already_liked_items=True, search_k=2000000, user_tags=None):
        if len(user_items) > 0:
            return super(ImplicitAnnoyItemFeatureRecommender, self).__recommend_internal__(user_label, user_items, N,
                                                                                           filter_items,
                                                                                           recalculate_user,
                                                                                           filter_already_liked_items,
                                                                                           search_k)
        else:
            return self.__recommend_by_item_tags__(user_tags)

    def __recommend_by_item_tags__(self, user_tags: dict, N=10):
        als_model = self.als_model
        tag_count_vec = np.zeros(self.tag_count)
        for q, v in user_tags:
            if q in self.tag_lookup:
                tag_label = self.tag_lookup[q]
                tag_count_vec[tag_label] += v
        if tag_count_vec.sum() == 0:
            return []
        tag_tfidf_vec = self.tag_tfidf_transformer.transform(tag_count_vec.reshape(1, -1))
        item_vec = sparse.hstack((sparse.csr_matrix(np.zeros(self.item_count)), tag_tfidf_vec), format='csr')
        item_embedding = self.als_model.item_factors * item_vec
        return als_model._get_similarity_score(item_embedding, als_model.item_factors[self.top_sales_work_labels],
                                               als_model.item_norms[self.top_sales_work_labels], N)
