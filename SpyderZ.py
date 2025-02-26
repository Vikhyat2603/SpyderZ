import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, _predict_binary
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class SpyderZ:
    def __init__(self, n_features, bin_size=0.1, max_z=4, C=1, run_parallel=True):
        """
        SpyderZ model constructor.

        Parameters
        ----------
        n_features: number of features being used (e.g.: number of photometric bands)
        bin_size: bin size for photo z prediction
        max_z: maximum value for redshift that model will consider
        C: hyperparameter for SVM training
        run_parallel: whether to use parallelization for training & prediction
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
                                            verbose=True,
                                            probability=False),
                                            n_jobs=(-1 if self.run_parallel else 1))

    def _convert_z_to_bin(self, z):
        """
        Convert z value to the discrete bin it belongs to
        """
        return SpyderZ._round_to_nearest_integer(z/self.bin_size).astype(np.int16)

    def _convert_bin_to_z(self, bin_idx):
        """
        Convert discrete bin index to the z value it represents
        """
        return bin_idx*self.bin_size

    def train(self, X, Z):
        """
        Train the model on the features and redshift data

        Parameters
        ----------
        X: input data, of shape (n_train_samples, n_features)
        Z: spectroscopic redshift data, of shape (n_train_samples, )
        """
        # validate parameter X
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        assert (len(X.shape)==2 and X.shape[1]==self.n_features), \
                "X must be a 2-d array of shape (n_train_samples, n_features)"

        # validate parameter Z

        if not isinstance(Z, np.ndarray):
            Z = np.array(Z)
            
        assert Z.max()<=self.max_z, "maximum redshift value should be < max_z"
        assert Z.min()>=0, "negative redshift values not allowed"

        # train model on (X,Z)
        
        Y = self._convert_z_to_bin(Z)
        
        self.model.fit(X, Y)

        self.n_classes = len(self.model.classes_)
        self._fitted = True
        
    def predict(self, X):
        """
        Predicts the redshift and ePDFs for the given input data

        Parameters
        ----------
        X: input data of shape (n_test_samples, n_features)

        Returns
        ----------
        (Z, ePDFs) where
        Z: photometric redshift estimate for each galaxy, of shape (n_test_samples, )
        ePDFs: effective PDFs for each galaxy, of shape (n_test_samples, n_bins)
        """
        
        # validate parameter X
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        assert (len(X.shape)==2 and X.shape[1]==self.n_features), \
                "X must be a 2-d array of shape (n_test_samples, n_features)"

        assert self._fitted, "model has not been fitted to training data yet"

        # predict Z and generate ePDFs for input data X
        
        X = self.model._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )

        # confidences represents decision function values of each classifier
        # is of shape (n_test_samples, n_classes(n_classes-1)/2)
        confidences = np.vstack([_predict_binary(est, X) for est in
                                 self.model.estimators_]).T

        ePDFs = self._convert_confidences_to_ePDFs(confidences)
        bin_pred = ePDFs.argmax(axis=1) # predict the photo-z as the most voted bin class
        z_pred = self._convert_bin_to_z(bin_pred)
        
        return z_pred, ePDFs

    def _convert_confidences_to_ePDFs(self, confidences):
        """ converts given confidence matrix to ePDFs """
        n_samples, n_pairs = confidences.shape
        votes = np.zeros((n_samples, len(self.labels)))
        pairs = np.array(list(combinations(self.model.classes_, 2))) # does not allow repeats (i,i)
        
        for sample_idx in range(n_samples):
            for pair_idx in range(n_pairs):
                # negative decision value means the first class got the vote
                if confidences[sample_idx,pair_idx]<0:
                    votes[sample_idx, pairs[pair_idx,0]] += 1
                else:
                    votes[sample_idx, pairs[pair_idx,1]] += 1

        # normalise votes for each galaxy so that highest voted bin class has value=1
        ePDFs = votes / np.array(votes.max(axis=1), ndmin=2).T
        
        return ePDFs

    def get_metrics(self, z_test, z_pred, print_metrics=True):
        """
        Calculate and return relevant photo-z estimation metrics 

        Parameters
        ----------
        z_test: spectroscopic redshifts for each galaxy, of shape (n_test_samples, )
        z_pred: photometric redshift estimtates for each galaxy, of shape (n_test_samples, )
        print_metrics: whether to print the calculated metrics

        Returns
        ----------
        metrics: a dictionary containing the given metrics as values, with the keys being:
        'rmse', 'r-rmse', 'nmad', 'outlier_frac', 'cat_outlier_frac', 'bin_accuracy'
        """
        
        metrics = dict()

        bin_pred = self._convert_z_to_bin(z_pred)
        size = len(bin_pred)

        error = (z_pred-z_test)/(1+z_test)
        non_outlier_error = error[np.where(np.abs(error)<0.15)]
        bin_test = self._convert_z_to_bin(z_test)

        metrics['rmse'] = np.sqrt(np.square(error).mean())
        metrics['r-rmse'] = np.sqrt(np.square(non_outlier_error).mean())
        metrics['nmad'] = 1.4826*np.median(np.abs(error - np.median(error)))
        metrics['outlier_frac'] = (np.abs(error)>0.15).sum() / size
        metrics['cat_outlier_frac'] = (np.abs(z_pred-z_test)>1).sum() / size
        metrics['bin_accuracy'] = (bin_pred==bin_test).sum() / size

        if print_metrics:
            print(f"RMSE: {round(metrics['rmse'],3)}")
            print(f"R-RMSE: {round(metrics['r-rmse'],3)}")
            print(f"NMAD: {round(metrics['nmad'],3)}")
            print(f"Outlier Fraction: {round(metrics['outlier_frac'],3)}")
            print(f"Catastrophic Outlier Fraction: {round(metrics['cat_outlier_frac'],3)}")
            print(f"Bin Match Accuracy: {round(metrics['bin_accuracy'],3)}")

        return metrics
    
    def plot_actual_vs_predicted_z(self, z_test, z_pred, save_filename=None, mode='color'):
        """
        Plots the photo-z vs spec-z density scatter graph

        Parameters
        ----------
        z_test: spectroscopic redshifts for each galaxy, of shape (n_test_samples, )
        z_pred: photometric redshift estimtates for each galaxy, of shape (n_test_samples, )
        save_filename: file name to save plot (default: None, does not save plot)
        mode: desired visual representative of galaxy density ('size' or 'color')'
        """
        
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
        
        ax.set_xlabel('$z_{spec}$')
        ax.set_ylabel('$z_{phot}$')
        ax.set_title('Spec-Z vs Photo-Zs')
        
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

        if save_filename is not None:
            plt.savefig(save_filename, dpi=300)
            
        plt.show()

    def plot_bias(self, z_test, z_pred, save_filename=None):
        """
        Plots the bias (=z_pred - z_test) vs spec-z density scatter graph

        Parameters
        ----------
        z_test: spectroscopic redshifts for each galaxy, of shape (n_test_samples, )
        z_pred: photometric redshift estimtates for each galaxy, of shape (n_test_samples, )
        save_filename: file name to save plot (default: None, does not save plot)
        """
        
        delta_bins = np.arange(-self.max_z, self.max_z+self.bin_size, self.bin_size)
        H, X, Y, hist = plt.hist2d(z_test, (z_pred-z_test),
                                   bins=(self.labels, delta_bins),
                                   cmap='plasma', norm='log')
        cbar = plt.colorbar(hist)
        cbar.set_label('# of test samples')
        plt.xlabel('$z_{spec}$')
        plt.ylabel('$z_{phot}-z_{spec}$')
        plt.title('Bias in photo-z estimation by spec-z bin')

        if save_filename is not None:
            plt.savefig(save_filename, dpi=300)
            
        plt.show()

    def plot_ePDF(self, epdf, save_filename=None):
        """
        Plots a given effective probability density function

        Parameters
        ----------
        epdf: effective probability density function for a galaxy, of shape (n_bins, )
        save_filename: file name to save plot (default: None, does not save plot)
        """
        
        fig, ax = plt.subplots()
        ax.bar(self.labels, epdf, width=self.bin_size, align='center')
        
        ax.set_xlabel('Photo-Z')
        ax.set_ylabel('Effective probability density')
        ax.set_title('Effective PDF for galaxy redshift')
        ax.set_xlim([-self.bin_size/2,self.max_z+self.bin_size/2])

        if save_filename is not None:
            plt.savefig(save_filename, dpi=300)
        
        plt.show()
    
    ########## Static methods
    def _round_to_nearest_integer(a):
        "rounds number to nearest integer using the 'round half away from zero' rule"
        # rounds to nearest integer
        return np.rint(np.nextafter(a, a+np.sign(a)))

