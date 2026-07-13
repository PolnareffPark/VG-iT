# Attribution: this file follows public THUML iTransformer / Time-Series-Library
# forecasting scaffold conventions (MIT); see repository-level THIRD_PARTY_NOTICES.md.

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_StressTest, Dataset_Synthetic
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'stress': Dataset_StressTest,
    'synthetic': Dataset_Synthetic,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if getattr(args, 'measure_efficiency_only', '') == 'real':
            # Efficiency profiling (real mode): use each model's training batch
            # for apples-to-apples inference-latency comparison. Default per-sample
            # bsz=1 would measure "deployment-style single-sample serving" instead,
            # which mixes unequally across models whose baselines use own batch.
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation (per-sample MSE/MAE, default)
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None)
    return data_set, data_loader
