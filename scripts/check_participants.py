import pandas as pd
import json
from pathlib import Path

human_dir = Path(r'C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data')
files = list(human_dir.glob('*.json'))
data = []
for f in files:
    with open(f, encoding='utf-8') as fp:
        data.extend(json.load(fp))

df = pd.DataFrame(data)
print(f'Total records: {len(df)}')
print(f'Unique participants: {df["participant_id"].nunique()}')
print(f'Participants: {sorted(df["participant_id"].unique())}')
print(f'\nRecords per participant:')
print(df['participant_id'].value_counts().sort_index())
