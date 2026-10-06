import os
import numpy as np

class EnhantsLoader:
    @staticmethod
    def load(location_id: str, use_synthetic_fallback: bool = True) -> np.ndarray:
        file_path = f"./data/traces_{location_id}.csv"
        
        if not os.path.exists(file_path):
            if use_synthetic_fallback:
                print(f"[Warning] Không tìm thấy file {file_path}. Đang sinh synthetic trace giả lập...")
                t = np.arange(0, 1000 * 30, 30) 
                power = np.abs(np.sin(t / 3600.0) * 45 + 15) 
                trace = np.stack([t, power], axis=-1) 
                return trace
            else:
                raise FileNotFoundError(f"Không tìm thấy file trace tại: {file_path}")
        else:
            try:
                data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            except Exception:
                data = np.loadtxt(file_path, delimiter=",")
            return data
