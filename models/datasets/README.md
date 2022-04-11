# 데이터 읽기
```python
import os
import pandas as pd

ds_path = os.path.join(os.getcwd(), "models/datasets")
data = pd.read_csv(f"{ds_path}/030350.csv")
data = data.iloc[:,1:]
data["date"] = pd.to_datetime(data["date"])
data
```

