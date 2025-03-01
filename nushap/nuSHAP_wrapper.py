#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   nuSHAP wrapper
#
#
################################################################################
import sys, time
import pandas as pd
import numpy as np
import shap
import warnings

import copy
import gc
import itertools
import logging
import time
import warnings
import subprocess

import scipy.sparse
import sklearn
from packaging import version
from scipy.special import binom
from sklearn.linear_model import Lasso, LassoLarsIC, lars_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from shap._explanation import Explanation
from shap.utils import safe_isinstance
from shap.utils._exceptions import DimensionError
from shap.utils._legacy import (
    DenseData,
    SparseData,
    convert_to_data,
    convert_to_instance,
    convert_to_instance_with_index,
    convert_to_link,
    convert_to_model,
    match_instance_to_data,
    match_model_to_data,
)
from shap.explainers._explainer import Explainer

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger('nu_shap')
np.random.seed(119)
################################################################################


class NuSHAP_tabular:
    """
    Some line of codes are copied from the Kernel SHAP, input model can be KNN, XGBoost
    """
    def __init__(self, model, data, feature_names=None, link="identity", **kwargs):

        self.collected_X = np.empty((0, 0))
        self.collected_y = np.empty((0,))

        if feature_names is not None:
            self.data_feature_names = feature_names
        elif isinstance(data, pd.DataFrame):
            self.data_feature_names = list(data.columns)

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.keep_index = kwargs.get("keep_index", False)
        self.model = convert_to_model(model, keep_index=self.keep_index)
        self.data = convert_to_data(data, keep_index=self.keep_index)
        model_null = match_model_to_data(self.model, self.data)

        if not isinstance(self.data, (DenseData, SparseData)):
            emsg = "NuShap explainer only supports the DenseData and SparseData input currently."
            raise TypeError(emsg)
        if self.data.transposed:
            emsg = "NuShap explainer does not support transposed DenseData or SparseData currently."
            raise DimensionError(emsg)

    def predict_and_collect_samples(self, X):
        """
        Predict the given samples X, and collect them for NuSHAP.

        :param X: Given samples
        :return:
        """
        y_prob = self.model.f(X)

        # Check if y_prob is a probability array or class labels
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:  # Probability array
            y = np.argmax(y_prob, axis=1)
        else:  # Predicted class labels
            y = y_prob

        # If `collected_X` is still empty, initialize it with the shape of the input X
        if self.collected_X.size == 0:
            self.collected_X = np.array(X)
        else:
            self.collected_X = np.vstack((self.collected_X, X))

        # Ensure `y` is a 2D column vector before concatenating
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Initialize `collected_y` with the correct dtype based on `y`
        if self.collected_y.size == 0:
            self.collected_y = y.astype(y.dtype)  # Ensure dtype matches y's dtype
        else:
            self.collected_y = np.vstack((self.collected_y.astype(y.dtype), y))

        # Combine `collected_X` and `collected_y` to remove duplicates
        combined_Xy = np.hstack((self.collected_X, self.collected_y))
        unique_Xy = np.unique(combined_Xy, axis=0)

        # Update `collected_X` and `collected_y` with unique values only, ensuring dtype consistency with y
        self.collected_X = unique_Xy[:, :-1]  # All columns except the last (features)
        self.collected_y = unique_Xy[:, -1:].astype(y.dtype)  # Ensure dtype matches y

        return y_prob

    def comp_nushap(self, inst_x, inst_y, save_sample, save_inst, error=0.0015, alpha=0.015):
        """
        Compute the NuSHAP values for the given instance, with the collected samples.
        :param inst_x: input x
        :param inst_y: output y of this x
        :param save_sample: save sample
        :param save_inst: save inst
        :return:
        """

        # Check if self.collected_X is flattened; if not, flatten it
        if self.collected_X.ndim > 2:
            # Flatten each sample in collected_X along the last axis
            self.collected_X = np.reshape(self.collected_X, (self.collected_X.shape[0], -1))

        # Check if inst_x is flattened; if not, flatten it
        if inst_x.ndim > 2:
            # Flatten inst_x along the last axis
            inst_x = np.reshape(inst_x, (inst_x.shape[0], -1))

        X_df = pd.DataFrame(self.collected_X, columns=self.data_feature_names)
        y_df = pd.DataFrame(self.collected_y, columns=["target"])
        samples_df = pd.concat([X_df, y_df], axis=1)
        samples_df.to_csv(save_sample, index=False)

        inst_X_df = pd.DataFrame(inst_x, columns=self.data_feature_names)
        inst_y_df = pd.DataFrame(inst_y, columns=["target"])
        inst_df = pd.concat([inst_X_df, inst_y_df], axis=1)
        inst_df.to_csv(save_inst, index=False)

        # reseting the collected samples
        self.collected_X = np.empty((0, 0))
        self.collected_y = np.empty((0,))

        # Define the command to run the script
        command = [
            "python3", "tools/nushap/nushap.py",
            "--dataset", save_sample,
            "--instfile", save_inst,
            "--error", f"{error}",
            "--alpha", f"{alpha}"
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in result.stdout.splitlines():
            if line.startswith("### Svs:"):
                print(f"#Samples: {samples_df.shape[0]}, ", line)
                return samples_df.shape[0], line
