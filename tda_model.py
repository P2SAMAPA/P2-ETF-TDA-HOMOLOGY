"""
Topological Data Analysis using Ripser for persistent homology.
Includes regime classification for ETF selection.
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
        self.history = None
        
    def compute_point_cloud(self, returns: pd.DataFrame, method: str = 'correlation') -> np.ndarray:
        if method == 'returns':
            X = returns.values.T
            X = StandardScaler().fit_transform(X)
        elif method == 'correlation':
            corr = returns.corr().values
            X = np.sqrt(2 * (1 - corr))
        else:
            X = returns.values.T
            X = StandardScaler().fit_transform(X)
        return X
    
    def compute_persistence(self, point_cloud: np.ndarray, is_distance: bool = True) -> Dict:
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
        results = []
        step = max(5, window // 20)
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
        self.history = pd.DataFrame(results).set_index('date')
        return self.history
    
    def compute_regime_signal(self) -> Dict:
        """
        Determine market regime from recent TDA history and output ETF selection signal.
        Returns dict with 'regime', 'confidence', 'recommended_style'.
        """
        if self.history is None or len(self.history) < 20:
            return {'regime': 'unknown', 'confidence': 0.0, 'recommended_style': 'neutral'}
        
        recent = self.history.iloc[-20:]
        betti1_trend = np.polyfit(range(20), recent['betti_1'].values, 1)[0]
        betti1_mean = recent['betti_1'].mean()
        betti1_norm = betti1_trend / (betti1_mean + 1e-6)
        
        pers_current = recent['max_persistence'].iloc[-1]
        pers_mean = recent['max_persistence'].mean()
        pers_std = recent['max_persistence'].std()
        pers_z = (pers_current - pers_mean) / (pers_std + 1e-6)
        
        # Classification logic
        if pers_z > 2.5:
            regime = 'regime_break'
            style = 'safe_haven'
            confidence = min(pers_z / 5.0, 1.0)
        elif betti1_norm > 0.2:
            regime = 'fragmentation'
            style = 'defensive'
            confidence = min(betti1_norm / 0.5, 1.0)
        elif betti1_norm < -0.2:
            regime = 'simplification'
            style = 'momentum'
            confidence = min(abs(betti1_norm) / 0.5, 1.0)
        else:
            regime = 'neutral'
            style = 'neutral'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': float(confidence),
            'recommended_style': style,
            'betti1_trend': float(betti1_norm),
            'persistence_z': float(pers_z)
        }
