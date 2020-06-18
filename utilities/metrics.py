"""One stop shop for implementation of all metrics
"""
import sys
import torch
import numpy as np
import sklearn.metrics as mt


def align_by_pelvis(
    joints: np.ndarray, get_pelvis: bool = False
) -> (np.ndarray, np.ndarray):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.

    Args:
        joints (np.ndarray): 3D Joints
        get_pelvis (bool, optional): return pelvis joints (True) or not (False)

    Returns:
        np.ndarray, np.ndarray: Description
    """
    left_id = 3
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    a = joints - np.expand_dims(pelvis, axis=0)
    b = pelvis if get_pelvis else None

    return (a, b)


def compute_iou(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Computes IOU between targets and predictions

    Args:
        targets (np.ndarray): Description
        predictions (np.ndarray): Description

    Returns:
        np.ndarray: IOU per image
    """
    predictions = predictions.squeeze(1)
    SMOOTH = 1e-6
    intersection = (predictions & targets).sum((1, 2))
    union = (predictions | targets).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou


def compute_mpjpe(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Calculates MPJPE

    Args:
        targets (np.ndarray): Target 3D joints (LSP Format)
        predictions (np.ndarray): Predicted 3D joints (LSP Format)

    Returns:
        np.ndarray: MPJPE calculated per subject
    """
    errors = []
    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        target = target.reshape(-1, 3)

        # Root align.
        target, _ = align_by_pelvis(target)
        prediction, _ = align_by_pelvis(prediction)
        joint_error = np.sqrt(np.sum(((target - prediction)) ** 2, axis=1))
        errors.append(np.mean(joint_error))

    error = np.array(errors)

    return error


def compute_s2s(
    targets: np.ndarray, predictions: np.ndarray, align: bool = True
) -> np.ndarray:
    """Calculates surface to surface error

    Args:
        targets (np.ndarray): Target 3D vertices
        predictions (np.ndarray): Prediction 3D vertices
        align (bool, optional): Align by pelvis (True) or not (False)

    Returns:
        np.ndarray: Surface 2 Surface error per subject
    """
    errors = []
    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        target = target.reshape(-1, 3)
        if align:
            target, _ = align_by_pelvis(target)
            prediction, _ = align_by_pelvis(prediction)

        diff = target - prediction
        l2_norm = np.linalg.norm(diff, axis=1)
        surface_errors = np.mean(l2_norm)
        errors.append(surface_errors)

    errors = np.array(errors)
    return errors


def compute_mae(targets: np.ndarray, predictions: np.ndarray,) -> np.ndarray:
    """Calculates mean absolute error

    Args:
        targets (np.ndarray):
        predictions (np.ndarray):

    Returns:
        np.ndarray: MAE per subject
    """
    return np.mean(np.abs(targets - predictions))


def compute_rel_vol_error(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Calculates relative volume error

    Args:
        targets (np.ndarray): Target volumes
        predictions (np.ndarray): Predicted volumes

    Returns:
        np.ndarray: Description
    """
    return np.mean(np.vstack(np.abs(targets - predictions) / targets))


def compute_pck(
    targets: np.ndarray, predictions: np.ndarray, threshold: int = 150
) -> np.ndarray:
    """Percentage of correct keypoints

    Definition: https://github.com/cbsudux/Human-Pose-Estimation-101#percentage-of-correct-key-points---pck

    Args:
        targets (np.ndarray): Target 3D Joints (in metres)
        predictions (np.ndarray): Predicted 3D Joints (in metres)
        threshold (int, optional): Units in milli metre

    Returns:
        np.ndarray: Percentage of correct keypoints (%)
    """
    threshold = threshold * 1e-3
    full_pck = list()
    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        target = target.reshape(-1, 3)
        # Root align.
        target, _ = align_by_pelvis(target)
        prediction, _ = align_by_pelvis(predictions)
        diff = target - prediction

        l2_norm = np.linalg.norm(diff, axis=1)

        pred_joints_within_bounds = l2_norm < threshold
        pck = np.sum(pred_joints_within_bounds) / pred_joints_within_bounds.shape[0]
        full_pck.append(pck)

    pck = np.vstack(full_pck)
    return pck


def compute_fgbg_f1(
    targets: np.ndarray, predictions: np.ndarray,
) -> (float, float, float, float):
    """Segmentation precision, recall, F1 and accuracy

    Args:
        targets (np.ndarray): Target segmentation masks
        predictions (np.ndarray): Predicted segmentation masks

    Returns:
        float, float, float, float: Precision, recall, F1-score, accuracy
    """
    predictions = predictions.reshape(-1, 1)
    targets = targets.reshape(-1, 1)

    f1 = mt.f1_score(targets, predictions)
    precision = mt.precision_score(targets, predictions)
    recall = mt.recall_score(targets, predictions)
    accuracy = mt.accuracy_score(targets, predictions)
    return precision, recall, f1, accuracy


def compute_similarity_transform(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Computes a similarity transform (sR, t) that takes a set of 3D points S1 (3 x N) closest to a set of 3D points S2, where R is an 3x3 rotation matrix, t 3x1 translation, s scale. i.e. solves the orthogonal Procrutes prob≈≈≈¨lem.

    Args:
        S1 (np.ndarray): 3D Joints of subject 1
        S2 (np.ndarray): 3D Joints of subject 2

    Returns:
    # TODO: What does this function return?
        np.ndarray: Description
    """
    assert any(
        [S1.shape[0] == x for x in (2, 3)]
    ), "S1 does not have the correct dimensions ({shape})".format(shape=S1.shape[0])

    S1 = S1.T
    S2 = S2.T

    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    #    singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))

    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    return S1_hat.T


def multi_class_matrices(
    targets: np.ndarray, predictions: np.ndarray, average: str = "binary"
) -> (float, float, float, float):
    """Segmentation precision, recall, F1 and accuracy

        Args:
            targets (np.ndarray): Target segmentation masks
            predictions (np.ndarray): Predicted segmentation masks

        Returns:
            float, float, float, float: Precision, recall, F1-score, accuracy
        """
    f1 = mt.f1_score(targets, predictions, average=average)
    precision = mt.precision_score(targets, predictions, average=average)
    recall = mt.recall_score(targets, predictions, average=average)
    accuracy = mt.accuracy_score(targets, predictions)
    return precision, recall, f1, accuracy
