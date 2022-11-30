import logging

import numpy as np
import pickle
from implicit.als import AlternatingLeastSquares
from implicit.utils import augment_inner_product_matrix
from scipy import sparse

log = logging.getLogger("rb.recommendation")


class RecommenderException(Exception):
    pass


class ImplicitRecommender:

    def __init__(self, als_model: AlternatingLeastSquares, user_labels: np.ndarray, item_labels: np.ndarray):
        self.als_model = als_model
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

    def __recommend_internal__(self, user_label, user_items, N=10, filter_items=None, recalculate_user=True,
                               filter_already_liked_items=True, **kwargs):
        recommendations = self.als_model.recommend(user_label, user_items=user_items, N=N, recalculate_user=True,
                                                   filter_already_liked_items=filter_already_liked_items)
        return [(self.get_item_id(item), score) for (item, score) in zip(*recommendations)]

    def recommend(self, item_ids, item_weights=None, number_of_results=50, filter_already_liked_items=True,
                  tag_count_vec: np.array=None):
        """
        Recommend items from a list of items and weights
        :param item_ids:
        :param item_weights:
        :param number_of_results:
        :param filter_already_liked_items:
        :param tag_count_vec: counts of tags of user engaged work
        :return: a list of tuples (item_id, weight)
        """
        user_label = 0
        user_items = None
        if item_ids is not None and len(item_ids)>0:
            item_lb = [self.get_item_label(i) for i in item_ids]
            user_ll = [0] * len(item_ids)
            confidence = [10] * len(item_ids) if item_weights is None else item_weights
            user_items = sparse.csr_matrix((confidence, (user_ll, item_lb)))

        recommendations = self.__recommend_internal__(user_label, user_items=user_items, N=number_of_results,
                                                      recalculate_user=True,
                                                      filter_already_liked_items=filter_already_liked_items,
                                                      tag_count_vec=tag_count_vec)

        recommendations = [(self.get_item_id(x[0]), x[1]) for x in recommendations]

        return recommendations

    def build_index(self, index_type: str, approximate_similar_items=True, approximate_recommend=True, **kwargs):
        """
        Builds an index from this model and returns a new ImplicitRecommender
        :param index_type:
        :param approximate_similar_items:
        :param approximate_recommend:
        :param kwargs:
        :return:
        """
        if index_type == 'annoy':
            from .annoy import ImplicitAnnoyRecommender
            recommender = ImplicitAnnoyRecommender.build_annoy_recommender(
                als_model=self.als_model,
                user_labels=self.user_labels, item_labels=self.item_labels,
                approximate_similar_items=approximate_similar_items, approximate_recommend=approximate_recommend,
                **kwargs
            )
            return recommender
        elif index_type == 'hnsw':
            from .hnsw import ImplicitHNSWRecommender
            recommender = ImplicitHNSWRecommender.build_hnsw_recommender(
                als_model=self.als_model,
                user_labels=self.user_labels, item_labels=self.item_labels,
                **kwargs
            )
            return recommender
        elif index_type is None:
            self.recommender = self.model
        else:
            raise RecommenderException("Unsupported optimization " + index_type)

    def save(self, base_name, user_factors=False, compress=False):
        als_file = base_name + ".npz"
        log.info("Saving ALS model to %s", als_file)
        data = {
            'model.item_factors': self.als_model.item_factors,
            'user_labels': self.user_labels,
            'item_labels': self.item_labels,
        }
        if user_factors:
            data.update({'model.user_factors': self.als_model.user_factors})
        if compress:
            np.savez_compressed(als_file, **data)
        else:
            np.savez(als_file, **data)


def load_recommender(als_model_file: str, index_file: str, item_feature_file: str = None, **kwargs) -> ImplicitRecommender:
    log.info("Loading als model")
    data = np.load(als_model_file, allow_pickle=True)
    model = AlternatingLeastSquares(factors=data['model.item_factors'].shape[1])
    model.item_factors = data['model.item_factors']
    model.YtY  # This will initialize the _YtY instance variable which is used directly in internal methods
    if 'user_factors' in data:
        model.user_factors = data['model.user_factors']

    user_labels = data['user_labels']
    item_labels = data['item_labels']

    if index_file is None:
        return ImplicitRecommender(model, user_labels, item_labels)

    elif index_file.endswith('.ann'):
        import annoy
        log.info("Loading annoy recommendation index")
        max_norm, extra = augment_inner_product_matrix(model.item_factors)
        recommend_index = annoy.AnnoyIndex(extra.shape[1], 'angular')
        recommend_index.load(index_file)  # prefault=load_to_memory does not seem to work

        if item_feature_file is None:
            from .annoy import ImplicitAnnoyRecommender
            return ImplicitAnnoyRecommender(model, recommend_index, max_norm, user_labels, item_labels)
        else:
            log.info("Loading item features for recommendation")
            item_feature_data = pickle.load(open(item_feature_file, "rb"))
            tag_tfidf_transformer = item_feature_data['tag_tfidf_transformer']
            tag_lookup = item_feature_data['tag_lookup']
            item_embedding_weight = item_feature_data['item_embedding_weight']
            from .annoy_item_features import ImplicitAnnoyItemFeatureRecommender
            return ImplicitAnnoyItemFeatureRecommender(model, recommend_index, max_norm, user_labels,item_labels,
                                                       tag_tfidf_transformer, tag_lookup, item_embedding_weight)
    elif index_file.endswith('.hnsw'):
        import hnswlib
        from .hnsw import ImplicitHNSWRecommender
        log.info("Loading hnsw recommendation index")
        # we build the index in l2 space and load it in inner product space on purpose.
        # This space change gives us 0.96 recall
        l2_recommend_index = hnswlib.Index(space='ip', dim=model.item_factors.shape[1])
        l2_recommend_index.load_index(index_file)
        l2_recommend_index.set_ef(kwargs.get('ef', 2000))
        return ImplicitHNSWRecommender(model, l2_recommend_index,user_labels, item_labels)
    else:
        raise RecommenderException("Unsupported file type" + index_file)
