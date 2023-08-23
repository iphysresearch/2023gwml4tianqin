# 引力波暑期学校 Summer School on Gravitational Waves

Time: 8月22日（星期二）/22th August (Tuesday)

Tutorial: 机器学习和引力波数据处理 Machine learning and GW data analysis

Lecturer: 曹周键 (Zhoujian Cao)/王赫 (He Wang)

# Acknowledge

中山大学天琴中心 Tian Qin Center for Gravitational Physics, Sun Yat-sen University


# Code

All files except for the `test.npy` can be found at the Kaggle ([https://www.kaggle.com/competitions/can-you-find-the-gw-signals/data](https://www.kaggle.com/competitions/can-you-find-the-gw-signals/data))

## Files

*   **data_prep_bbh.py** - script for data generation (credit: [Dr. Hunter Gabbard](https://github.com/hagabbar/cnn_matchfiltering/))
*   **utils.py** - supplemental script containing some useful functions
*   **main.py** - main script for training / evaluation / submission
*   **test.npy** - test data for submission

You can load the test data in the Kaggle notebook as follows

```python
import numpy as np
test_dataset = np.load('/kaggle/input/can-you-find-the-gw-signals/test.npy')
```

## Hints

Some useful information about `test.npy` dataset

- SNR: 2,4,5,6,7,8,9,10
- num of injections: 800
- `astro_metric`

---

Anyway, just check the [tutorial notebook](https://www.kaggle.com/code/herbwang/tutorial-notebook) for everything!
