"""
Topological Data Analysis using Ripser for persistent homology.
"""

import numpy as np
import pandas as pd
from ripser import ripser
from persim import PersistenceImager
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

class TDAHomologyAnalyzer:
    def __init__(self, max_dim: int = 2, n_landscapes: int = 5):
        self.max_dim = max_dim
        self.n_landscapes = n_landscapes
        self.pimgr = PersistenceImager(pixel_size=0.01, birth_range=(0, 1), pers_range=(0, 1))
        
    def compute_point_cloud(self, returns: pd.DataFrame, method: str = 'returns') -> np.ndarray:
        """
        Create point cloud from returns or correlation distances.
        """
        if method == 'returns':
            X = returns.values.T  # assets as points in time
        elif method == 'correlation':
            corr = returns.corr().values
            # Use correlation distance: sqrt(2*(1-corr))
            dist = np.sqrt(2 * (1 - corr))
            X = dist
        else:
            X = returns.values.T
        # Standardize
        X = StandardScaler().fit_transform(X)
        return X
    
    def compute_persistence(self, point_cloud: np.ndarray) -> Dict:
        """
        Compute persistent homology diagrams.
        """
        if point_cloud.shape[0] < 10:
            return {'betti_numbers': [0]* (self.max_dim+1), 'max_persistence': 0, 'diagrams': None}
        
        try:
            dgms = ripser(point_cloud, maxdim=self.max_dim, thresh=1.0)['dgms']
        except Exception:
            return {'betti_numbers': [0]* (self.max_dim+1), 'max_persistence': 0, 'diagrams': None}
        
        betti = []
        max_pers = 0
        for dim in range(self.max_dim+1):
            if dim < len(dgms):
                pers = dgms[dim][:, 1] - dgms[dim][:, 0]
                finite = pers[np.isfinite(pers)]
                betti.append(len(finite))
                if len(finite) > 0:
                    max_pers = max(max_pers, np.max(finite))
            else:
                betti.append(0)
        
        return {
            'betti_numbers': betti,
            'max_persistence': float(max_pers),
            'diagrams': [d.tolist() for d in dgms] if dgms else None
        }
    
    def compute_landscapes(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Convert persistence diagrams to persistence landscapes for time-series analysis.
        """
        landscapes = []
        for dim, dgm in enumerate(diagrams):
            if dim > self.max_dim:
                break
            if len(dgm) == 0:
                landscapes.append(np.zeros(self.n_landscapes))
                continue
            # Use PersistenceImager as proxy for landscape features
            img = self.pimgr.transform(dgm)
            landscapes.append(img.flatten()[:self.n_landscapes])
        return np.concatenate(landscapes)
    
    def rolling_tda(self, returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Compute TDA features over rolling windows.
        Returns DataFrame with Betti numbers and max persistence over time.
        """
        results = []
        for end in range(window, len(returns) + 1, 5):  # every 5 days for efficiency
            start = end - window
            window_returns = returns.iloc[start:end]
            point_cloud = self.compute_point_cloud(window_returns, method='correlation')
            pers = self.compute_persistence(point_cloud)
            results.append({
                'date': returns.index[end-1],
                'betti_0': pers['betti_numbers'][0],
                'betti_1': pers['betti_numbers'][1],
                'betti_2': pers['betti_numbers'][2] if len(pers['betti_numbers'])>2 else 0,
                'max_persistence': pers['max_persistence']
            })
        df = pd.DataFrame(results).set_index('date')
        return df
    
    def compute_early_warning(self, tda_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate early-warning signals from TDA features.
        """
        df = tda_df.copy()
        # Normalized change in Betti-1 (holes)
        df['betti_1_change'] = df['betti_1'].diff(5).abs() / (df['betti_1'].shift(5) + 1)
        # Max persistence z-score
        pers_mean = df['max_persistence'].rolling(252, min_periods=50).mean()
        pers_std = df['max_persistence'].rolling(252, min_periods=50).std().replace(0, 1e-6)
        df['persistence_z'] = (df['max_persistence'] - pers_mean) / pers_std
        return df
