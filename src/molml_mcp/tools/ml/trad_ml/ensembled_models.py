

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
import numpy as np

class BayesianEnsemble(BaseEstimator):
    """
    Ensemble wrapper for uncertainty estimation.
    
    Wraps any sklearn estimator and trains n independent models
    with bootstrap sampling. Provides uncertainty estimates while
    maintaining sklearn API compatibility.
    """

    _ensemble_params = {'n_estimators'}

    
    def __init__(self, base_estimator, n_estimators=10, **kwargs):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_scikit_base_params(self):
        """Extract only base estimator params"""
        all_params = self.get_params(deep=False)
        
        # Remove ensemble-specific and base_estimator itself
        base_params = {
            k: v for k, v in all_params.items() 
            if k not in self._ensemble_params and k != 'base_estimator'
        }
        
        return base_params
        
    def fit(self, X, y):
        base_params = self._get_scikit_base_params()
        self.estimators_ = []
        
        # Handle random_state: generate unique seeds for each model if provided
        random_states = None
        if 'random_state' in base_params and base_params['random_state'] is not None:
            rng = np.random.RandomState(base_params['random_state'])
            random_states = rng.randint(0, 100000, size=self.n_estimators)
        
        for i in range(self.n_estimators):
            estimator = clone(self.base_estimator)

            # Only set params that the estimator actually accepts
            valid_params = estimator.get_params().keys()
            filtered_params = {k: v for k, v in base_params.items() if k in valid_params}
            
            # Override random_state with unique seed for this model
            if random_states is not None and 'random_state' in valid_params:
                filtered_params['random_state'] = random_states[i]
            
            if filtered_params:
                estimator.set_params(**filtered_params)

            X_sample, y_sample = X, y
                
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)
            
        return self
    
    def predict(self, X):
        """Standard sklearn predict - returns mean"""
        predictions = self._get_predictions(X)
        return predictions.mean(axis=0)
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty estimates.
        
        Returns:
            tuple: (mean, std, predictions)
                - mean: Average predictions across all models
                - std: Standard deviation (uncertainty)
                - predictions: Full array of shape (n_models, n_samples)
        """
        predictions = self._get_predictions(X)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std, predictions
    
    def _get_predictions(self, X):
        """Get predictions from all estimators"""
        return np.array([est.predict(X) for est in self.estimators_])
    
    # Optional: Add predict_proba support for classifiers
    def predict_proba(self, X):
        """Average probability predictions (classifiers only)"""
        if not hasattr(self.estimators_[0], 'predict_proba'):
            raise AttributeError("Base estimator doesn't support predict_proba")
        
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return probas.mean(axis=0)
    
    def predict_proba_with_uncertainty(self, X):
        """
        Probability predictions with uncertainty.
        
        Returns:
            tuple: (mean, std, probas)
                - mean: Average probabilities across all models
                - std: Standard deviation (uncertainty) of probabilities
                - probas: Full array of shape (n_models, n_samples, n_classes)
        """
        if not hasattr(self.estimators_[0], 'predict_proba'):
            raise AttributeError("Base estimator doesn't support predict_proba")
            
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        mean = probas.mean(axis=0)
        std = probas.std(axis=0)
        
        return mean, std, probas

    def __len__(self):
        return len(self.estimators_)

    def __repr__(self):
        random_state = getattr(self, 'random_state', None)
        return f"BayesianEnsemble(base_estimator={self.base_estimator}, n_estimators={self.n_estimators}, random_state={random_state})"



# Random_Forest
# Extra Trees
# Gradient Boosting
# AdaBoost
# Logistic Regression
# Ridge/Lasso/ElasticNet