import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from statistics import NormalDist


def _pearson(x, y):
    return stats.pearsonr(x, y).statistic


def _spearman(x, y):
    return stats.spearmanr(x, y).statistic


# measure the log likelihood of the data given the model
def _loglikelihood(model):
    return model.llf


def _aic(x, y, n, k):
    rss = np.sum((x - y) ** 2)
    return n * np.log(rss / n) + 2 * k


def _bic(x, y, n, k):
    rss = np.sum((x - y) ** 2)
    return n * np.log(rss / n) + k * np.log(n)


def _r2(x, y):
    return r2_score(x, y)


def _rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))


def confidence_interval(data, confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.0)
    h = dist.stdev * z / ((len(data) - 1) ** 0.5)
    return dist.mean - h, dist.mean + h


def results_to_mean_confidence_interval(results):
    return {
        k1: {
            k2: {k3: [np.mean(v3), confidence_interval(v3)] for k3, v3 in v2.items()}
            for k2, v2 in v1.items()
        }
        for k1, v1 in results.items()
    }
