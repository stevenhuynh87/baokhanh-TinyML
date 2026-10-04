import numpy as np
import os

class EnhantsLoader:
    @staticmethod
    def load(location_id: str) -> np.ndarray:
        """
        Tải trace năng lượng ánh sáng.
        Đầu ra: np.ndarray có shape [N, 2] gồm (t, µW/cm²)
        """
        if not os.path.exists(f"./data/traces_{location_id}.csv"):
            t = np.arange(0, 1000, 30)
            power = np.abs(np.sin(t / 100.0) * 50 + 10)
            trace = np.stack([t, power], axis=-1)
            return trace
        else:
            data = np.loadtxt(f"./data/traces_{location_id}.csv", delimiter=",")
            return data
