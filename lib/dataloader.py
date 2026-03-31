import os
import torch
import numpy as np


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.mean, np.ndarray):
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


class MinMax01Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.min, np.ndarray):
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return data * (self.max - self.min + 1e-8) + self.min


class MinMax11Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min + 1e-8)) * 2.0 - 1.0

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.min, np.ndarray):
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.0) / 2.0) * (self.max - self.min + 1e-8) + self.min


def normalize_data(data, axis, scalar_type='Standard'):
    if scalar_type == 'MinMax01':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        return MinMax01Scaler(min=min_val, max=max_val)
    if scalar_type == 'MinMax11':
        return MinMax11Scaler(min=data.min(), max=data.max())
    if scalar_type == 'Standard':
        return StandardScaler(mean=data.mean(), std=data.std() + 1e-8)
    raise ValueError('scalar_type is not supported in data_normalization.')


def build_pgr_labels(y_data, train_y, num_levels=6):
    train_power = np.asarray(train_y)[..., 0] if np.asarray(train_y).ndim == 4 else np.asarray(train_y)
    y_power = np.asarray(y_data)[..., 0] if np.asarray(y_data).ndim == 4 else np.asarray(y_data)

    z_max = np.max(train_power, axis=(0, 1), keepdims=True)
    z_max = np.maximum(z_max, 1e-8)

    pgr = np.floor((num_levels - 1) * y_power / z_max).astype(np.int64)
    pgr = np.clip(pgr, 0, num_levels - 1)
    return pgr


def process_time_label(time_label, horizon=None):
    t = np.asarray(time_label)
    if t.ndim == 1:
        if horizon is None:
            horizon = 52
        t = np.tile(np.arange(horizon, dtype=np.int64)[None, :], (t.shape[0], 1))
    elif t.ndim == 2:
        t = t.astype(np.int64)
    elif t.ndim == 3:
        t = np.argmax(t, axis=-1).astype(np.int64)
    else:
        raise ValueError('Unsupported time_label shape.')
    if horizon is not None and t.shape[1] != horizon:
        if t.shape[1] > horizon:
            t = t[:, :horizon]
        else:
            pad = np.repeat(t[:, -1:], horizon - t.shape[1], axis=1)
            t = np.concatenate([t, pad], axis=1)
    return t


def STDataLoader_T(X, X_m, Y, time_label, c, batch_size, device, shuffle=True, drop_last=True):
    x_tensor = torch.FloatTensor(X).to(device)
    xm_tensor = torch.FloatTensor(X_m).to(device)
    y_tensor = torch.FloatTensor(Y).to(device)
    t_tensor = torch.LongTensor(time_label).to(device)
    c_tensor = torch.LongTensor(c).to(device)

    data = torch.utils.data.TensorDataset(x_tensor, xm_tensor, y_tensor, t_tensor, c_tensor)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def get_dataloader(data_dir, dataset, batch_size, test_batch_size, device, scalar_type='MinMax01'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x'].astype(np.float32)
        data['x_m_' + category] = cat_data['x_m'].astype(np.float32)
        data['y_' + category] = cat_data['y'].astype(np.float32)
        data['time_' + category] = cat_data['time_label']

    horizon = data['y_train'].shape[1]
    num_nodes = data['y_train'].shape[2]

    data['time_train'] = process_time_label(data['time_train'], horizon=horizon)
    data['time_val'] = process_time_label(data['time_val'], horizon=horizon)
    data['time_test'] = process_time_label(data['time_test'], horizon=horizon)

    data['c_train'] = build_pgr_labels(data['y_train'], data['y_train'])
    data['c_val'] = build_pgr_labels(data['y_val'], data['y_train'])
    data['c_test'] = build_pgr_labels(data['y_test'], data['y_train'])

    scaler = normalize_data(
        np.concatenate([data['x_train'], data['x_val']], axis=0),
        axis=(0, 1, 3),
        scalar_type=scalar_type
    )
    scaler_m = normalize_data(
        np.concatenate([data['x_m_train'], data['x_m_val']], axis=0),
        axis=(0, 1),
        scalar_type=scalar_type
    )

    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category]).astype(np.float32)
        data['x_m_' + category] = scaler_m.transform(data['x_m_' + category]).astype(np.float32)
        data['y_' + category] = scaler.transform(data['y_' + category]).astype(np.float32)

        data['time_' + category] = data['time_' + category].astype(np.int64)
        data['c_' + category] = data['c_' + category].astype(np.int64)

        if data['c_' + category].shape[1] != horizon:
            if data['c_' + category].shape[1] > horizon:
                data['c_' + category] = data['c_' + category][:, :horizon, :]
            else:
                pad = np.repeat(data['c_' + category][:, -1:, :], horizon - data['c_' + category].shape[1], axis=1)
                data['c_' + category] = np.concatenate([data['c_' + category], pad], axis=1)

        if data['c_' + category].shape[2] != num_nodes:
            if data['c_' + category].shape[2] > num_nodes:
                data['c_' + category] = data['c_' + category][:, :, :num_nodes]
            else:
                pad = np.repeat(data['c_' + category][:, :, -1:], num_nodes - data['c_' + category].shape[2], axis=2)
                data['c_' + category] = np.concatenate([data['c_' + category], pad], axis=2)

    dataloader = {}
    dataloader['train'] = STDataLoader_T(
        data['x_train'],
        data['x_m_train'],
        data['y_train'],
        data['time_train'],
        data['c_train'],
        batch_size,
        device=device,
        shuffle=True,
        drop_last=True
    )
    dataloader['val'] = STDataLoader_T(
        data['x_val'],
        data['x_m_val'],
        data['y_val'],
        data['time_val'],
        data['c_val'],
        test_batch_size,
        device=device,
        shuffle=False,
        drop_last=True
    )
    dataloader['test'] = STDataLoader_T(
        data['x_test'],
        data['x_m_test'],
        data['y_test'],
        data['time_test'],
        data['c_test'],
        test_batch_size,
        device=device,
        shuffle=False,
        drop_last=False
    )
    dataloader['scaler'] = scaler
    dataloader['scaler_m'] = scaler_m
    return dataloader
