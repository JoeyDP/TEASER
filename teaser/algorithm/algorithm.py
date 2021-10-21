from typing import List

import numpy as np
import scipy.sparse

import teaser.util as util

# explanations with absolute value scores belows this are not shown
MIN_EXPL_SCORE = 0.05

# Amount by which value should rise when a user clicks plus (respectively minus)
FEEDBACK_INCREMENT = 0.2
MAX_FEEDBACK_MOD = np.ceil(1. / FEEDBACK_INCREMENT)


class Algorithm:
    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags: List[str] = None):
        return self

    def predict_all(self, X: scipy.sparse.csr_matrix, retarget: bool = False):
        raise NotImplementedError()


class AspectAlgorithm(Algorithm):
    DT_: np.array

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags: List[str] = None):
        super().fit(X)

        t = S.shape[1]
        if tags is None:
            tags = list(map(str, list(range(t))))
        else:
            tags = tags.copy()

        assert len(tags) == t
        tags.append("popularity_popularity")
        t += 1
        self.tags = tags

        features = [tag.split('_')[0] for tag in tags]
        self.features = list(set(features))  # unique features with same order as in the index
        feature_id_map = {feature: i for i, feature in enumerate(self.features)}
        feature_ids = [feature_id_map[feature] for feature in features]
        # tag by feature matrix to indicate feature membership
        self.features_index = scipy.sparse.csr_matrix((np.ones(t), (feature_ids, np.arange(t))), dtype=np.int8).T

    def _user_vector(self, history):
        raise NotImplementedError()

    def certainty(self, history, user_vec):
        """ A score in ]0, 1] to indicate how certain the system is of the user vector. """
        # Default implementation scales certainty from 0.2 to 0.8 for histories between 0 and 3.
        # 0.8 further on
        return min(0.2 + len(history) * 0.2, 0.8)

    def user_vector(self, history, modifications):
        """
        Compute the user vector based on history of items and given feedback.

        modifications is a dictionary (tag -> int) with possible values from -10 to 10.
        Each step represents one press of the plus or minus buttons where 5 times plus would
        bring a tag of score 0 to a represented score of 1 (increments of 0.2).

        :param history: list of item ids user interacted with
        :param modifications: dictionary of user feedback on tags
        :return: user vector
        """
        user_vec = self._user_vector(history)

        certainty = self.certainty(history, user_vec)

        if len(history) == 0:
            user_vec[-1] = 3 * FEEDBACK_INCREMENT
            certainty = user_vec[-1]

        bound = np.abs(user_vec).max() / certainty

        incr = bound * FEEDBACK_INCREMENT
        for tag, mod in modifications.items():
            if abs(mod) > MAX_FEEDBACK_MOD:
                print(f"invalid modifier for tag {tag}: {mod}.")
                continue

            try:
                tag_index = self.tags.index(tag)
            except ValueError as e:
                print(f"Couldn't find tag: {tag}")
                continue

            user_vec[tag_index] += incr * mod
            user_vec[tag_index] = max(-bound, user_vec[tag_index])
            user_vec[tag_index] = min(bound, user_vec[tag_index])

        return user_vec

    def explain_profile(self, history, modifications=dict()):
        """
        Give scores for each tag normalized in [-1, 1] and for each feature normalized in [0, 1].
        :param history: list of item ids user interacted with
        :param modifications: dictionary of user feedback on tags
        :return:
        """
        history = np.array(history, dtype=np.int32)

        base_user_vec = self.user_vector(history, dict())
        user_vec = self.user_vector(history, modifications)

        certainty = self.certainty(history, user_vec)

        if len(history) == 0:
            certainty = base_user_vec[-1]

        tag_scores = user_vec / np.abs(base_user_vec).max() * certainty
        tag_scores = {tag: score for tag, score in zip(self.tags, tag_scores)}

        # feature scores is sum of absolute values times tag values
        tag_values = np.asarray(self.DT_.T.sum(axis=0)).flatten()
        feature_scores = (tag_values * np.abs(user_vec)) @ self.features_index
        feature_importances = util.normalize(feature_scores)
        feature_importances = {feature: score for feature, score in zip(self.features, feature_importances)}

        return tag_scores, feature_importances

    def predict(self, history, modifications=dict(), retarget: bool = False):
        """
        Compute scores for one user as fast as possible.
        :param history: list of item ids the user interacted with
        :return:
        """
        user_vec = self.user_vector(history, modifications)
        scores = np.asarray(user_vec @ self.DT_).flatten()

        if not retarget:
            scores[history] = -1e10

        return scores

    def predict_explain(self, history, modifications=dict(), top_k=20, top_expl=5, retarget: bool = False):
        # first get top-k best items
        user_vec = self.user_vector(history, modifications)
        # user_vec = util.rescale0(user_vec, -1, 1)
        prediction = np.asarray(user_vec @ self.DT_).flatten()

        max_score = np.max(prediction)
        if max_score == 0:
            max_score = 1

        if not retarget:
            prediction[history] = -1e10

        top_items, scores = util.prediction_to_recommendations(prediction, top_k=top_k)

        # scores /= max_score
        scores[scores < 0] = 0

        # Compute % match score as cosine similarity between user vector and item vector scaled to 0-1
        scores /= np.linalg.norm(user_vec)
        scores /= np.linalg.norm(self.DT_[:, top_items].toarray(), axis=0)
        scores = (scores + 1) / 2

        top_items = np.array(top_items, dtype=np.int32)

        # generate explanations for only those to save time
        aspect_scores = scipy.sparse.csc_matrix(self.DT_[:, top_items].multiply(user_vec[:, np.newaxis]))
        abs_sum = np.abs(aspect_scores).sum(axis=0).flatten()
        abs_sum[abs_sum == 0] = 1
        aspect_scores = np.abs(aspect_scores) / abs_sum

        def top_explanations(aspect_array):
            # TODO: can speed this method up using sparsity
            aspect_array = np.asarray(aspect_array).flatten()
            expl_count = min(top_expl, (np.abs(aspect_array) >= MIN_EXPL_SCORE).sum())

            top_tags, _ = util.prediction_to_recommendations(np.abs(aspect_array), top_k=expl_count)
            return [
                {'tag': self.tags[tag_idx],
                 'score': aspect_array[tag_idx]} for tag_idx in top_tags]

        return [
            {'item': item,
             'score': scores[idx],
             'aspects': top_explanations(aspect_scores[:, idx])
             } for idx, item in enumerate(top_items)
        ]