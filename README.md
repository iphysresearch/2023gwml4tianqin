<img width="672" alt="image" src="https://github.com/iphysresearch/2023gwml4tianqin/assets/14322948/0020ec47-ed87-41c0-9301-8f391071cf82">

# 引力波暑期学校 Summer School on Gravitational Waves

Time: 8月22日（星期二）/22th August (Tuesday)

Tutorial: 机器学习和引力波数据处理 Machine learning and GW data analysis

Lecturer: 曹周键 (Zhoujian Cao)/王赫 (He Wang)

FYI: 
> Based on [Gabbard, Hunter, Michael Williams, Fergus Hayes, and Chris Messenger. “Matching Matched Filtering with Deep Networks for Gravitational-Wave Astronomy.” Physical Review Letters 120, no. 14 (December 17, 2017): 141103.](https://doi.org/10.1103/PhysRevLett.120.141103), this is a full reproduction of the code that's both simple and concise. Utilizing PyTorch, it maintains the same network structure as the original code and is also based on Gaussian noise. Furthermore, it has been extended to two detectors.

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
