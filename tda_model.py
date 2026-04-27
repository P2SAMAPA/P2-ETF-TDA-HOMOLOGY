"""
Topological Data Analysis using Ripser.
Return‑seeking regime classifier.
"""

import numpy as np
import pandas as pd
from ripser import ripser
from sklearn.preprocessing import StandardScaler

class TDAHomologyAnalyzer:
    def __init__(self, max_dim=2):
        self.max_dim = max_dim
        self.history = None

    def compute_point_cloud(self, returns: pd.DataFrame, method='correlation') -> np.ndarray:
        if method == 'correlation':
            corr = returns.corr().values
            X = np.sqrt(2 * (1 - corr))
        else:
            X = returns.values.T
            X = StandardScaler().fit_transform(X)
        return X

    def compute_persistence(self, point_cloud: np.ndarray, is_distance=True) -> dict:
        if point_cloud.shape[0] < 10:
            return {'betti_numbers': [0,0,0], 'max_persistence': 0}
        try:
            dgms = ripser(point_cloud, maxdim=self.max_dim, distance_matrix=is_distance, thresh=1.0)['dgms']
        except:
            return {'betti_numbers': [0,0,0], 'max_persistence': 0}
        betti = []
        max_pers = 0
        for dim in range(self.max_dim+1):
            if dim < len(dgms):
                pers = dgms[dim][:,1] - dgms[dim][:,0]
                finite = pers[np.isfinite(pers)]
                betti.append(len(finite))
                if len(finite) > 0:
                    max_pers = max(max_pers, np.max(finite))
            else:
                betti.append(0)
        return {'betti_numbers': betti, 'max_persistence': float(max_pers)}

    def rolling_tda(self, returns: pd.DataFrame, window=252) -> pd.DataFrame:
        results = []
        step = max(5, window // 20)
        for end in range(window, len(returns)+1, step):
            start = end - window
            win_ret = returns.iloc[start:end]
            pc = self.compute_point_cloud(win_ret, 'correlation')
            pers = self.compute_persistence(pc, is_distance=True)
            results.append({
                'date': returns.index[end-1],
                'betti_0': pers['betti_numbers'][0],
                'betti_1': pers['betti_numbers'][1],
                'betti_2': pers['betti_numbers'][2] if len(pers['betti_numbers'])>2 else 0,
                'max_persistence': pers['max_persistence']
            })
        self.history = pd.DataFrame(results).set_index('date')
        return self.history

    def compute_regime(self, history=None) -> dict:
        """Determine regime from recent history. Returns regime string and amplification factor."""
        if history is None:
            history = self.history
        if history is None or len(history) < 20:
            return {'regime': 'neutral', 'confidence': 0.5, 'boost_factor': 1.0}

        recent = history.iloc[-20:]
        betti1_trend = np.polyfit(range(20), recent['betti_1'].values, 1)[0]
        betti1_mean = recent['betti_1'].mean()
        betti1_norm = betti1_trend / (betti1_mean + 1e-6)

        pers_current = recent['max_persistence'].iloc[-1]
        pers_mean = recent['max_persistence'].mean()
        pers_std = recent['max_persistence'].std()
        pers_z = (pers_current - pers_mean) / (pers_std + 1e-6)

        if pers_z > 2.5:
            regime = 'regime_break'
            confidence = min(pers_z / 5.0, 1.0)
        elif betti1_norm > 0.2:
            regime = 'fragmentation'
            confidence = min(betti1_norm / 0.5, 1.0)
        elif betti1_norm < -0.2:
            regime = 'simplification'
            confidence = min(abs(betti1_norm) / 0.5, 1.0)
        else:
            regime = 'neutral'
            confidence = 0.5

        boost_factor = config.REGIME_BOOST.get(regime, 1.0)
        # Apply macro conditioning: during high VIX, reduce boost slightly
        # This is optional – we can add it later.
        return {'regime': regime, 'confidence': confidence, 'boost_factor': boost_factor}
