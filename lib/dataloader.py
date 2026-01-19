import os
import torch 
import numpy as np

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:

    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


def STDataloader_T(X, X_m, Y, time_label, c, batch_size, device,shuffle=True, drop_last=True,train_flag=True):

    TensorFloat = torch.FloatTensor
    TensorInt = torch.LongTensor

    X, X_m, Y, time_label,c= TensorFloat(X).to(device), TensorFloat(X_m).to(device), TensorFloat(Y).to(device), TensorInt(time_label).to(device), TensorFloat(c).to(device)
    data = torch.utils.data.TensorDataset(X, X_m, Y, time_label, c)

    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader

def normalize_data(data, axis, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'MinMax01':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        scalar = MinMax01Scaler(min=min_val, max=max_val)
    elif scalar_type == 'MinMax11':
        scalar = MinMax11Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'Standard':
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    return scalar

def get_dataloader(data_dir, dataset, batch_size, test_batch_size, device, scalar_type='MinMax01'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['x_m_' + category] = cat_data['x_m']
        data['y_' + category] = cat_data['y']
        data['time_'+category] = cat_data['time_label'] 
        data['c_'+category] = cat_data['c']
    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), axis = (0,1,3), scalar_type = scalar_type)
    scaler_m = normalize_data(np.concatenate([data['x_m_train'], data['x_m_val']], axis=0), axis=(0,1), scalar_type = scalar_type)
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['x_m_' + category] = scaler_m.transform(data['x_m_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
    dataloader = {}
    dataloader['train'] = STDataloader_T(
        data['x_train'],
        data["x_m_train"],
        data['y_train'],
        data['time_train'],
        data['c_train'],
        batch_size,
        device=device,
        shuffle=True
    )

    dataloader['val'] = STDataloader_T(
        data['x_val'],
        data["x_m_val"],
        data['y_val'],
        data['time_val'],
        data['c_val'],
        test_batch_size,
        device=device, 
        shuffle=False
    )
    dataloader['test'] = STDataloader_T(
        data['x_test'],
        data["x_m_test"],
        data['y_test'],
        data['time_val'],
        data['c_test'], 
        test_batch_size,
        device=device, 
        shuffle=False, 
        drop_last=False,
        train_flag=False
    )
    dataloader['scaler'] = scaler
    return dataloader
