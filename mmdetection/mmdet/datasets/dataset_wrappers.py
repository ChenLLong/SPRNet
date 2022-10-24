import bisect
import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .registry import DATASETS


@DATASETS.register_module
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

@DATASETS.register_module
class RatioConcatDataset(_ConcatDataset):

    def __init__(self, datasets, ratios):
        super(RatioConcatDataset, self).__init__(datasets)
        self.datasets = datasets
        self.ratios = ratios / np.array(ratios).sum()
        self.lengths = []
        for dataset in self.datasets:
            self.lengths.append(len(dataset))
        self.lengths = np.array(self.lengths)
        self.cumulative_ratios = self.cumsum_ratio()

        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def __len__(self):
        return self.lengths.sum()

    def __getitem__(self, item):
        i = np.random.rand()
        ind = bisect.bisect_right(self.cumulative_ratios, i)
        b_ind = np.random.randint(self.lengths[ind])
        return self.datasets[ind][b_ind]

    def cumsum_ratio(self):
        r, s = [], 0
        for e in self.ratios[:-1]:
            r.append(e + s)
            s += e
        return r



@DATASETS.register_module
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len
