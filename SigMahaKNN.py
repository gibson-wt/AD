from joblib import Parallel, delayed

from sklearn.neighbors import NearestNeighbors
from sktime.transformations.panel.signature_based import SignatureTransformer
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

from Distance import Mahalanobis

__all__ = "SignatureMahalanobisKNN"


class SignatureMahalanobisKNN:
    def __init__(self,
                 n_jobs=-2,
                 ):
        """
        :param n_jobs: parameter for joblib, number of parallel processors to use.
        :param augmentation_list: Possible augmentation strings are
        ['leadlag', 'ir', 'addtime', 'cumsum', 'basepoint']
        :param window_name: str, String from ['global', 'sliding', 'expanding', 'dyadic']
        :param window_depth: int, The depth of the dyadic window. (Active only if
        `window_name == 'dyadic']`.
        :param window_length: int, The length of the sliding/expanding window. (Active
        only if `window_name in ['sliding, 'expanding'].
        :param window_step: int, The step of the sliding/expanding window. (Active
        only if `window_name in ['sliding, 'expanding'].
        :param rescaling: "pre" or "post",
                    "pre": rescale the path last signature term should be roughly O(1)
                    "post": Rescals the output signature by multiplying the depth-d term by d!.
                            Aim is that every term become ~O(1).
        :param sig_tfm: One of:.
        :param depth: int, Signature truncation depth.
        """
        self.n_jobs = n_jobs
        self.knn = None

    def fit(self, sigs):
        """

        :param X: Must support index operation X[i] where each X[i] returns a data point in the corpus
        :return:
        """

        mahal_distance = Mahalanobis()
        mahal_distance.fit(sigs)

        knn = NearestNeighbors(
            metric=mahal_distance.distance,
            n_jobs=self.n_jobs,
            algorithm='auto')
        knn.fit(sigs)
        self.knn = knn

    def conformance(self, sigs):
        distances, _ = self.knn.kneighbors(sigs, n_neighbors=1)

        return distances

    def compute_auc(self, test_in, test_out):
        distances_in = self.conformance(test_in)
        distances_out = self.conformance(test_out)
        return self.compute_auc_given_dists(distances_in, distances_out)

    def compute_auc_given_dists(self, distances_in, distances_out):
        # replace infinity with twice of the maximum value, hacky, may need more thoughts
        distances_in[distances_in == np.inf] = np.nan
        distances_out[distances_out == np.inf] = np.nan
        max_val = max(np.nanmax(distances_in), np.nanmax(distances_out))
        distances_in = np.nan_to_num(distances_in, max_val * 2)
        distances_out = np.nan_to_num(distances_out, max_val * 2)

        auc = roc_auc_score([False] * len(distances_in) + [True] * len(distances_out),
                            np.concatenate([distances_in,
                                            distances_out]
                                           )
                            )
        return auc
