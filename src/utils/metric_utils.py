import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class CustomCOCOeval(COCOeval):
    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None

        Some variable definitions are as follows (defaults in brackets):
        - iouThrs: [.5:.05:.95] T=10 IoU thresholds for evaluation
        - recThrs: [0:.01:1] R=101 recall thresholds for evaluation
        - catIds : [all] K cat ids to use for evaluation
        - areaRng: [...] A=4 object area ranges for evaluation
        - maxDets: [1 10 100] M=3 thresholds on max detections per image
        - useCats: [1] if true use category labels for evaluation
        """
        print("Accumulating evaluation results...")
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)

        # -1 for the precision of absent categories
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        pre = {}
        rec = {}
        fp_per_image = {}

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e["dtMatches"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    dtIg = np.concatenate(
                        [e["dtIgnore"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)

                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

                    # we only record pre, rec and fppi(for curve plotting)
                    # when area range: "all", maxDet: 100 and iou thres: 0.5
                    if a0 == 0 and m == len(m_list) - 1:
                        fp_per_image[p.catIds[k0]] = fp_sum[0] / len(i_list)
                        if npig == 0:
                            pre[p.catIds[k0]] = np.zeros_like(tp_sum[0])
                            rec[p.catIds[k0]] = np.zeros_like(tp_sum[0])
                        else:
                            pre[p.catIds[k0]] = tp_sum[0] / (
                                np.arange(len(tp_sum[0])) + 1
                            )
                            rec[p.catIds[k0]] = tp_sum[0] / npig

                        if len(fp_sum[0]):
                            pre[p.catIds[k0]] = np.hstack((1, pre[p.catIds[k0]]))
                            rec[p.catIds[k0]] = np.hstack((0, rec[p.catIds[k0]]))
                            fp_per_image[p.catIds[k0]] = np.hstack(
                                (0, fp_per_image[p.catIds[k0]])
                            )

                    if npig == 0:
                        continue

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        for ri, pi in enumerate(inds):
                            if pi < len(pr) and pi < len(dtScoresSorted):
                                q[ri] = pr[pi]

                        precision[t, :, k, a, m] = np.array(q)
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "precision": precision,
            "recall": recall,
            "pre": pre,
            "rec": rec,
            "fp_per_image": fp_per_image,
        }

    def summarize(self):
        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = {}
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(
                1, iouThr=0.5, areaRng="small", maxDets=self.params.maxDets[2]
            )
            stats[5] = _summarize(
                1, iouThr=0.75, areaRng="small", maxDets=self.params.maxDets[2]
            )
            stats[6] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[7] = _summarize(
                1, iouThr=0.5, areaRng="medium", maxDets=self.params.maxDets[2]
            )
            stats[8] = _summarize(
                1, iouThr=0.75, areaRng="medium", maxDets=self.params.maxDets[2]
            )
            stats[9] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(
                1, iouThr=0.5, areaRng="large", maxDets=self.params.maxDets[2]
            )
            stats[11] = _summarize(
                1, iouThr=0.75, areaRng="large", maxDets=self.params.maxDets[2]
            )
            stats[12] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[13] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[14] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[15] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[16] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[17] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            stats[18] = self.eval["pre"]
            stats[19] = self.eval["rec"]
            stats[20] = self.eval["fp_per_image"]
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")

        self.stats = _summarizeDets()


class CustomMeanAveragePrecision(MeanAveragePrecision):
    @property
    def cocoeval(self):
        return CustomCOCOeval

    @staticmethod
    def _coco_stats_to_tensor_dict(stats, prefix):
        """Converts the output of COCOeval.stats to a dict of tensors."""
        return {
            f"{prefix}map": torch.tensor([stats[0]], dtype=torch.float32),
            f"{prefix}map_50": torch.tensor([stats[1]], dtype=torch.float32),
            f"{prefix}map_75": torch.tensor([stats[2]], dtype=torch.float32),
            f"{prefix}map_small": torch.tensor([stats[3]], dtype=torch.float32),
            f"{prefix}map_small_50": torch.tensor([stats[4]], dtype=torch.float32),
            f"{prefix}map_small_75": torch.tensor([stats[5]], dtype=torch.float32),
            f"{prefix}map_medium": torch.tensor([stats[6]], dtype=torch.float32),
            f"{prefix}map_medium_50": torch.tensor([stats[7]], dtype=torch.float32),
            f"{prefix}map_medium_75": torch.tensor([stats[8]], dtype=torch.float32),
            f"{prefix}map_large": torch.tensor([stats[9]], dtype=torch.float32),
            f"{prefix}map_large_50": torch.tensor([stats[10]], dtype=torch.float32),
            f"{prefix}map_large_75": torch.tensor([stats[11]], dtype=torch.float32),
            f"{prefix}mar_1": torch.tensor([stats[12]], dtype=torch.float32),
            f"{prefix}mar_10": torch.tensor([stats[13]], dtype=torch.float32),
            f"{prefix}mar_100": torch.tensor([stats[14]], dtype=torch.float32),
            f"{prefix}mar_small": torch.tensor([stats[15]], dtype=torch.float32),
            f"{prefix}mar_medium": torch.tensor([stats[16]], dtype=torch.float32),
            f"{prefix}mar_large": torch.tensor([stats[17]], dtype=torch.float32),
            f"{prefix}pre": {
                k: torch.tensor(v, dtype=torch.float32) for k, v in stats[18].items()
            },
            f"{prefix}rec": {
                k: torch.tensor(v, dtype=torch.float32) for k, v in stats[19].items()
            },
            f"{prefix}fp_per_image": {
                k: torch.tensor(v, dtype=torch.float32) for k, v in stats[20].items()
            },
        }


def perpendicula_distance(x, y, mode="max"):
    """
    compute the perpendicular distance between points and line. The line is pass through first and latest point.

    :param x: (np.array) A numpy array representing the x coordiate.
    :param y: (np.array) A numpy array representing the y coordiate.
    :param mode: (str) The mode to calculate the distance. Available options:
            - "max": Returns the maximum perpendicular distance between the points and the line.
            - "mid": Returns the perpendicular distance between the midpoint on curve and the line.
            Defaults is "max".

    :Retrun: (flaot) The perpendicular distance between point and the line.
    """
    if len(x) <= 2:
        return 0

    x1, x2 = x[0], x[-1]
    y1, y2 = y[0], y[-1]

    if x1 == x2 and y1 == y2:
        return 0

    # ax + by + c = 0, The equation representing the line passing through the first and last points.
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    if mode == "max":
        distance = (
            np.abs(a * x[1:-1] + b * y[1:-1] + c).max() / (a**2 + b**2) ** 0.5
        )
    elif mode == "mid":
        m_idx = len(x) // 2
        distance = np.abs(a * x[m_idx] + b * y[m_idx] + c) / (a**2 + b**2) ** 0.5
    else:
        raise ValueError(
            "`{mode}` is not valid mode. Supported modes are 'max' and 'mid'."
        )

    return distance


def curve_approximation(x, y, threshold=1e-4):
    """
    Downsamples a 2D curve by removing points that have a small perpendicular distance to the line passing through their neighboring points.
    Reference: https://github.com/kghose/groho/blob/stable/docs/dev/adaptive-display-points.ipynb (cell 6: Saturating maximum deviation)

    :param x: (np.array) A numpy array representing the x coordiate. The shape of the array is (n,), where n is the number of points
    :param y: (np.array) A numpy array representing the y coordiate. The shape of the array is (n,), where n is the number of points
    :param threshold: (flaot) A threshold value that determines the minimum perpendicular distance allowed for a point to be retained.

    :Return: (Tuple) numpy array containing the downsampled curve.
    """
    assert x.ndim > 0, "The dimension of `x` should be greater than zero."
    assert y.ndim > 0, "The dimension of `y` should be greater than zero."

    num_x, num_y = len(x), len(y)
    assert (
        num_x == num_y
    ), f"Expected `point_x` and `point_y` to be equal lengths, but `point_x` has {num_x} elements while `point_y` has {num_y}."

    if num_x <= 2:
        return x, y

    sample_idx = [0]

    for end_idx in range(2, num_x):
        start_idx = sample_idx[-1]
        if (
            perpendicula_distance(
                x[start_idx : end_idx + 1], y[start_idx : end_idx + 1]
            )
            > threshold
        ):
            sample_idx.append(end_idx - 1)

    sample_idx.append(num_x - 1)

    x, y = x[sample_idx], y[sample_idx]

    return x, y


def drop_intermediate(x, y):
    """
    Remove intermediate points in vertical/horizontal lines from the roc/pr curve.
    :param x: (np.array) Input array representing the x-axis values.
    :param y: (np.array) Input array representing the y-axis values.

    :Returns: (Tuple) x and y arrays with intermediate points dropped.
    """
    assert x.ndim > 0, "The dimension of `x` should be greater than zero."
    assert y.ndim > 0, "The dimension of `y` should be greater than zero."

    num_x, num_y = len(x), len(y)
    assert (
        num_x == num_y
    ), f"Expected `x` and `y` to be equal lengths, but `x` has {num_x} elements while `y` has {num_y}."

    if num_x <= 2:
        return x, y

    index = np.r_[True, np.logical_and(np.diff(x, 2), np.diff(y, 2)), True]
    return x[index], y[index]
