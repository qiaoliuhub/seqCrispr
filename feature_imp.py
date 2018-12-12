from sklearn import metrics
import numpy as np
from scipy.stats import spearmanr

class Ranking(object):
    def __init__(self, names):
        self.names = names

    def _normalize(self, impt, fea_num):
        impt = impt / sum(impt)
        impt = list(zip(impt, self.names, range(fea_num)))
        impt.sort(key=lambda x: -x[0])
        return impt


class InputPerturbationRank(Ranking):
    def __init__(self, names):
        super(InputPerturbationRank, self).__init__(names)

    def _raw_rank(self, rep, y, network, x):

        fea_num = 0
        for fea in x:
            fea_num += int(fea.shape[1])
        impt = np.zeros(fea_num)

        fea_index = 0
        for fea_dfs in x:
            for i in range(fea_dfs.shape[1]):
                hold = np.array(fea_dfs[:, i])
                for j in range(rep):
                    np.random.shuffle(fea_dfs[:, i])

                    # Handle both TensorFlow and SK-Learn models.
                    if 'tensorflow' in str(type(network)).lower():
                        pred = list(network.predict(x))
                    else:
                        pred = network.predict(x)

                    rmse = metrics.mean_squared_error(y, pred)
                    spearman_correlation = spearmanr(y, pred)[0]
                    impt[fea_index] += (spearman_correlation - impt[fea_index]) / (j + 1)

                fea_index += 1
                fea_dfs[:, i] = hold

        return impt, fea_num

    def rank(self, rep, y, network, x):
        impt, fea_num = self._raw_rank(rep, y, network, x)
        return self._normalize(impt, fea_num)
