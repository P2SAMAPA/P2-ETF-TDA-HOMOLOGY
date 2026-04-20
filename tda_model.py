"""
Topological Data Analysis using Ripser for persistent homology.
"""

import numpy as np
import pandas as pd
from ripser import ripser
from persim import PersistenceImager
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

class TDAHomologyAnalyzer:
    def __init__(self, max_dim: int = 2, n_landscapes: int = 5):
        self.max_dim = max_dim
        self.n_landscapes = n_landscapes
        self.pimgr = PersistenceImager(pixel_size=0.01, birth_range=(0, 1), pers_range=(0, 1))
        
    def compute_point_cloud(self, returns: pd.DataFrame, method: str = 'correlation') -> np.ndarray:
        """
        Create point cloud from returns or correlation distances.
        """
        if method == 'returns':
            X = returns.values.T  # assets as points in time
            X = StandardScaler().fit_transform(X)
        elif method == 'correlation':
            corr = returns.corr().values
            # Correlation distance matrix: sqrt(2*(1-corr))
            X = np.sqrt(2 * (1 - corr))
        else:
            X = returns.values.T
            X = StandardScaler().fit_transform(X)
        return X
    
    def compute_persistence(self, point_cloud: np.ndarray, is_distance: bool = True) -> Dict:
        """
        Compute persistent homology diagrams.
        """
        if point_cloud.shape[0] < 10:
            return {'betti_numbers': [0]* (self.max_dim+1), 'max_persistence': 0, 'diagrams': None}
        
        try:
            dgms = ripser(point_cloud, maxdim=self.max_dim, distance_matrix=is_distance, thresh=1.0)['dgms']
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
    
    def rolling_tda(self, returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Compute TDA features over rolling windows.
        """
        results = []
        step = max(5, window // 20)  # adaptive step
        for end in range(window, len(returns) + 1, step):
            start = end - window
            window_returns = returns.iloc[start:end]
            point_cloud = self.compute_point_cloud(window_returns, method='correlation')
            pers = self.compute_persistence(point_cloud, is_distance=True)
            results.append({
                'date': returns.index[end-1],
                'betti_0': pers['betti_numbers'][0],
                'betti_1': pers['betti_numbers'][1],
                'betti_2': pers['betti_numbers'][2] if len(pers['betti_numbers'])>2 else 0,
                'max_persistence': pers['max_persistence']
            })
        return pd.DataFrame(results).set_index('date')
    
    def compute_early_warning(self, tda_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate early-warning signals from TDA features.
        """
        df = tda_df.copy()
        df['betti_1_change'] = df['betti_1'].diff(5).abs() / (df['betti_1'].shift(5) + 1)
        pers_mean = df['max_persistence'].rolling(252, min_periods=50).mean()
        pers_std = df['max_persistence'].rolling(252, min_periods=50).std().replace(0, 1e-6)
        df['persistence_z'] = (df['max_persistence'] - pers_mean) / pers_std
        return df
