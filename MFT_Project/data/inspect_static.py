import pickle
from pathlib import Path

path = Path('MFT_Project/data/static_market_data.pkl')
df = pickle.loads(path.read_bytes())
feature_cols = [c for c in df.columns if c[0] == 'feature']
label_cols = [c for c in df.columns if c[0] == 'label']
print('feature count:', len(feature_cols))
print('label columns:', label_cols)
print('first feature columns:', feature_cols[:10])
