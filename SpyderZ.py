import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mpl_scatter_density

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, _predict_binary
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class SpyderZ:
    def __init__(self, n_features, bin_size=0.1, max_z=4, C=1, run_parallel=True):
        """
        SpyderZ algorithm.

        Parameters
        ----------
        n_features: 
        bin_size: bin size for photo z prediction
        """
        
        self.n_features = n_features
        self.bin_size = bin_size
        self.max_z = max_z
        self.C = C
        self.run_parallel = run_parallel
        maxLabel = np.ceil(max_z/bin_size)*bin_size
        self.labels = np.arange(0, maxLabel + bin_size, bin_size)

        self._fitted = False

        self.model = OneVsOneClassifier(SVC(decision_function_shape='ovo',
                                            C=C,
                                            probability=False),
                                            n_jobs=(-1 if self.run_parallel else 1))

    def convert_z_to_bin(self, z):
        return SpyderZ.round_to_nearest_integer(z/self.bin_size).astype(np.int16)

    def convert_bin_to_z(self, y):
        return y*self.bin_size #(y+1)*bin_size - bin_size/2

    def train(self, X, Z):
        """
        """
        
        assert Z.max()<=self.max_z, "maximum redshift value should be < max_z"
        assert Z.min()>=0, "negative redshift values not allowed"
        
        Y = self.convert_z_to_bin(Z)
        
        self.model.fit(X, Y)

        self.n_classes = len(self.model.classes_)
        
        self._fitted = True
        
    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        assert (len(X.shape)==2 and X.shape[1]==self.n_features), \
                "X must be a 2-d array of shape (n_test_samples, n_features)"

        # vik - make batch-wise splits according to RAM available

        X = self.model._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )

        # confidences is of shape (n_samples, n_classes)
        # represents decision function values of each classifier for each test sample
        confidences = np.vstack([_predict_binary(est, X) for est in
                                 self.model.estimators_]).T
        
        EPDFs = self._convert_confidences_to_EPDFs(confidences)
        y_pred = EPDFs.argmax(axis=1)
        z_pred = self.convert_bin_to_z(y_pred)
        
        return z_pred, EPDFs

    def _convert_confidences_to_EPDFs(self, confidences):
        n_samples, n_pairs = confidences.shape
        votes = np.zeros((n_samples, len(self.labels)))
        pairs = np.array(list(combinations(self.model.classes_, 2))) # does not allow repeats (i,i)
        
        for sample_idx in range(n_samples):
            for pair_idx in range(n_pairs):
                ### negative decision value means the first (0th) class got the vote, else second (1th) class did
                if confidences[sample_idx,pair_idx]<0:
                    votes[sample_idx, pairs[pair_idx,0]] += 1
                else:
                    votes[sample_idx, pairs[pair_idx,1]] += 1
        votes /= np.array(votes.max(axis=1), ndmin=2).T
        
        return votes

    def get_metrics(self, z_test, z_pred, print_metrics=True):
        y_pred = self.convert_z_to_bin(z_pred)
        size = len(y_pred)

        error = np.abs(z_pred-z_test)/(1+z_test)
        outlier_pct = (error>0.15).sum() / size
        cat_outlier_pct = (np.abs(z_pred-z_test)>1).sum() / size
        y_test = self.convert_z_to_bin(z_test)
        num_correct = (y_pred==y_test).sum()

        metrics={'rmse':np.sqrt(np.square(error).mean()),
                 'r-rmse':np.sqrt(np.square(error[np.where(error<0.15)]).mean()),
                 'outlier_pct':100*outlier_pct,
                 'cat_outlier_pct': 100*cat_outlier_pct,
                 'bin_accuracy': 100*num_correct/size}

        if print_metrics:
            print(f"RMSE: {round(metrics['rmse'],5)}")
            print(f"R-RMSE: {round(metrics['r-rmse'],5)}")
            print(f"\nOutlier Percent:, {round(metrics['outlier_pct'],3)}")
            print(f"Catastrophic Outlier Percent: {round(metrics['cat_outlier_pct'],3)}")
            print(f"\nBin Match Accuracy: {round(metrics['bin_accuracy'],3)}")

        return metrics

    def plot_actual_vs_predicted_z(self, z_test, z_pred, save_plot_as=None, mode='size'):
        assert (mode in ['color', 'size']), "'mode' parameter should be either 'color' or 'size'"
        
        fig, ax = plt.subplots(figsize=(5,5),dpi=100)

        maxSize = 30
        if mode=='size':
            H, X, Y = np.histogram2d(z_test, z_pred, bins=[self.labels, self.labels])
            dotX, dotY = np.meshgrid(X[:-1],Y[:-1])
            scatter = ax.scatter(dotY, dotX, marker='s', s=H/H.max()*maxSize)
        else:
            H, X, Y, hist = ax.hist2d(z_test, z_pred, bins=[self.labels, self.labels], cmap='plasma', norm=colors.LogNorm())

        upper_error_line = 0.15+1.15*self.labels
        lower_error_line = 0.85*self.labels-0.15
        
        ax.plot(self.labels, upper_error_line,'r', linestyle='dashed', label='Outlier Boundary')
        ax.plot(self.labels, lower_error_line,'r', linestyle='dashed')
        ax.plot(self.labels, 1+self.labels,'r', label='Cat. Outlier Boundary')
        ax.plot(self.labels, -1+self.labels,'r')
        
        ax.set_xlabel('Actual Z')
        ax.set_ylabel('Predicted Z')
        ax.set_title('Actual vs Predicted Photo-Z')
        
        legend1 = ax.legend(fontsize=7, title_fontsize=12, loc="upper right")
        ax.add_artist(legend1)
        ax.set_aspect('equal')

        # produce legend of densities for size/color
        if mode=='size':
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6,c='tab:blue')
            labels = np.int16(0.1*np.arange(1,maxSize+1)*H.max())
            legend2 = ax.legend(handles, labels, fontsize=7, title_fontsize=7, loc="lower right", title="# of Samples")

            ax.set_xlim([-self.bin_size/2,self.max_z+self.bin_size/2])
            ax.set_ylim([-self.bin_size/2,self.max_z+self.bin_size/2])
        else:
            cbar = plt.colorbar(hist)
            cbar.set_label('# of test samples')

        if save_plot_as is not None:
            plt.savefig(save_plot_as, dpi=300)
            
        plt.show()

    def plot_ePDF(self, epdf):
        fig, ax = plt.subplots()
        ax.bar(self.labels, epdf, width=self.bin_size, align='center')
        
        ax.set_xlabel('Photo-Z')
        ax.set_ylabel('Effective probability')
        ax.set_title('Effective probability distribution')
        ax.set_xlim([-self.bin_size/2,self.max_z+self.bin_size/2])
        
        plt.show()
    
    ########## Static methods
    def round_to_nearest_integer(a):
        # rounds to nearest integer
        return np.rint(np.nextafter(a, a+np.sign(a)))

