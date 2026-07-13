import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from utils.grouping import GroupingManager
from utils.custom_grouping import compute_grouping_plan
import random
import numpy as np
import sys
import os


class Logger(object):
    """Tee stdout/stderr to a log file under test_results/."""
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        os.fsync(self.log.fileno())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTransformer')

    def _parse_grouping_method(raw):
        try:
            normalized = raw.strip()
            GroupingManager.parse_method(normalized)
            return normalized
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))

    def _parse_grouping_strategy(raw):
        try:
            normalized = raw.strip()
            return GroupingManager.parse_group_strategy(normalized)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iFlashformer, iNystromformer, VG_iTransformer, VG_iFlashformer]')
    parser.add_argument('--num_landmarks', type=int, default=64,
                        help='number of landmarks for Nystrom attention (iNystromformer only)')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--debug', type=int, default=0, help='Enable debug mode (1 epoch, limited iterations)')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'],
                        help='AMP compute dtype when --use_amp is enabled')
    parser.add_argument('--allow_tf32', type=int, default=1, help='allow TF32 on CUDA matmul/cudnn')
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='max train iterations per epoch (0 means full epoch)')
    parser.add_argument('--max_eval_steps', type=int, default=0,
                        help='max iterations for val/test/predict loops (0 means full loop)')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='use torch.compile() for model optimization (PyTorch 2.x)')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # Output dir settings
    parser.add_argument('--output_subdir', type=str, default=None, help='subdirectory for checkpoints')

    # VG-iT
    parser.add_argument('--num_groups', type=int, default=32, help='number of groups for VG-iT')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'statistical', 'learnable', 'dynamic'], help='pooling strategy for hierarchical attention')
    parser.add_argument('--use_variable_resolution', type=int, default=1, help='whether to use hierarchical structure (Ablation 3)')
    parser.add_argument('--use_interaction_bridge', type=int, default=1, help='whether to use gated bridge (Ablation 6)')
    parser.add_argument('--use_global_interact', type=int, default=1, help='whether to use global interaction in hierarchical attention')
    parser.add_argument('--use_shifted_grouping', type=int, default=0,
                        help='enable Swin-inspired shifted grouping across layers (0/1)')
    parser.add_argument('--use_gated_broadcast', type=int, default=0,
                        help='enable gated broadcast integration for global context (0/1)')
    parser.add_argument('--use_multi_shift', type=int, default=0,
                        help='enable per-layer unique shift offsets (0/1), overrides shifted_grouping')
    parser.add_argument('--use_film_broadcast', type=int, default=0,
                        help='enable FiLM multiplicative broadcast: gamma*local + beta (0/1)')
    parser.add_argument('--film_rank', type=str, default='full',
                        help='FiLM rank constraint: full/rank_N/rank_N_relu/diagonal (N=bottleneck dim)')
    parser.add_argument('--use_sdpa', type=int, default=0,
                        help='use SDPA (FlashAttention) for local attention in VG-iT (0/1)')

    parser.add_argument('--skip_flops_profiling', type=int, default=0,
                        help='skip FLOPs/Params profiling in test() to save ~30-120s per job (0/1). '
                             'Use when FLOPs already measured for this model×dataset×PL with another seed.')
    parser.add_argument('--skip_test', type=int, default=0,
                        help='skip exp.test() after training (0/1). Use for RAM-constrained datasets '
                             'where metric will be recovered via streaming eval from checkpoint.')
    parser.add_argument('--use_learnable_grouping', action='store_true', help='whether to use learnable grouping layer')
    
    # Phase 1: Ablation Flags
    parser.add_argument('--use_reorder', type=int, default=0, help='apply deterministic variate reordering by grouping method (Ablation 1)')
    parser.add_argument('--summary_file', type=str, default='summary.csv', help='summary file name for isolated experiments')
    parser.add_argument('--noise_std', type=float, default=0.0, help='standard deviation of Gaussian noise to inject into inputs (Ablation 2)')
    
    parser.add_argument('--partition_strategy', type=str, default='softmax', choices=['softmax', 'gumbel', 'topk'],
                        help='partitioning strategy for dynamic grouping')
    parser.add_argument('--dynamic_tokens_per_group', type=int, default=1, help='number of tokens per group in dynamic pooling')
    parser.add_argument(
        '--grouping_method',
        type=_parse_grouping_method,
        default='ordered',
        choices=GroupingManager.list_supported_methods(),
        help='grouping strategy for variable indices'
    )
    parser.add_argument('--grouping_horizon', type=int, default=None, help='horizon for horizon-specific clustering')
    parser.add_argument(
        '--grouping_strategy',
        type=_parse_grouping_strategy,
        default='auto',
        choices=GroupingManager.list_supported_group_strategies(include_auto=True),
        help='grouping strategy axis: auto|A_window|B_pack|B_ragged|C_horizon|D_feature'
    )
    parser.add_argument('--group_partition', type=str, default='window', choices=['window', 'pack', 'ragged'],
                        help='group partition strategy: window=equal split, pack=balanced packing, ragged=non-uniform groups')
    parser.add_argument('--group_balance_alpha', type=float, default=0.15,
                        help='balance regularization weight for pack/ragged partition')
    parser.add_argument('--custom_grouping_method', type=str, default='none',
                        choices=['none', 'ordered', 'finch_like', 'coarsening', 'mi_based', 'random',
                                 'anti_clustering', 'score_stratified', 'maximin_dispersion'],
                        help='Custom graph-based grouping method (research05 plan). none=use existing grouping_method.')
    parser.add_argument('--group_alpha', type=float, default=1.5,
                        help='Max efficiency degradation factor for group size bound (1.0=balanced, higher=more lenient)')
    parser.add_argument('--group_min_size', type=int, default=1,
                        help='Minimum group size (1=allow singletons)')

    # Efficiency profiling mode (plans/streamed-inventing-dove.md)
    parser.add_argument('--measure_efficiency_only', type=str, default='',
                        choices=['', 'dummy', 'real'],
                        help='If set to dummy/real, run 3-epoch efficiency profiling instead of full training. '
                             'MSE/MAE results are NOT logged to result_long_term_forecast.txt or summary.csv. '
                             'Output: --measure_efficiency_output JSON.')
    parser.add_argument('--measure_efficiency_output', type=str, default='',
                        help='Output JSON path for efficiency record. '
                             'Default: ./test_results/efficiency_profile_{dummy|real}.json')

    args = parser.parse_args()


    # Set seed
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_amp_requested = bool(args.use_amp)
    if args.use_amp_requested and not args.use_gpu:
        print("WARNING: --use_amp requested but CUDA is unavailable; AMP is disabled.")
    args.use_amp = bool(args.use_amp_requested and args.use_gpu)
    if args.use_amp and args.amp_dtype == 'bf16':
        bf16_supported = bool(
            hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
        )
        if not bf16_supported:
            print("WARNING: BF16 AMP requested but unsupported on this CUDA device; falling back to FP16 AMP.")
            args.amp_dtype = 'fp16'
    args.amp_effective = bool(args.use_amp)

    # Custom grouping: enable use_reorder so the precomputed perm is applied
    if args.custom_grouping_method != 'none':
        if args.use_reorder != 1:
            print(f"INFO: Custom grouping (--custom_grouping_method={args.custom_grouping_method}) "
                  f"requires use_reorder=1 for permutation. Setting use_reorder=1.")
            args.use_reorder = 1

    # GroupingManager grouping: also requires use_reorder=1
    if (args.model.startswith('VG_') and
        args.custom_grouping_method == 'none' and
        (args.grouping_method != 'ordered' or
         args.group_partition != 'window' or
         args.grouping_strategy != 'auto')):
        if args.use_reorder != 1:
            print(f"INFO: GroupingManager grouping (method={args.grouping_method}, "
                  f"partition={args.group_partition}) requires use_reorder=1. Setting use_reorder=1.")
            args.use_reorder = 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy,
                ii)

            # Setup logging to test_results/<output_subdir>/<model_id>.log
            # Skip if stdout is already redirected by a shell script
            _logger_active = False
            if sys.stdout.isatty():
                log_dir = os.path.join('./test_results', args.output_subdir) if args.output_subdir else './test_results'
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, '{}.log'.format(args.model_id))
                sys.stdout = Logger(log_path, sys.stdout)
                sys.stderr = Logger(log_path, sys.stderr)
                _logger_active = True

            # Setup checkpoints path
            if args.output_subdir:
                real_checkpoints_path = os.path.join('./checkpoints', args.output_subdir)
            else:
                real_checkpoints_path = './checkpoints/'

            import copy
            exp_args = copy.deepcopy(args)
            exp_args.checkpoints = real_checkpoints_path

            exp = Exp(exp_args)  # set experiments
            args.group_sizes = ''
            exp.args.group_sizes = ''
            
            # --- Custom Grouping Initialization ---
            use_custom_grouping = (
                exp_args.model.startswith('VG_') and
                getattr(exp_args, 'custom_grouping_method', 'none') != 'none'
            )
            requires_group_plan = (
                exp_args.model.startswith('VG_') and
                (use_custom_grouping or
                 getattr(exp_args, 'grouping_method', 'ordered') != 'ordered' or
                 getattr(exp_args, 'group_partition', 'window') != 'window' or
                 getattr(exp_args, 'grouping_strategy', 'auto') != 'auto')
            )
            if requires_group_plan:
                train_data, _ = exp._get_data(flag='train')

                if use_custom_grouping:
                    # Use graph-based custom grouping (research05 plan)
                    print(
                        f"Initializing Custom Graph-Based Grouping: "
                        f"method={exp_args.custom_grouping_method}, G={exp_args.num_groups}, seed={exp_args.seed}..."
                    )
                    # Use full train data for similarity computation (no sampling)
                    X_train = train_data.data_x  # (T, N)

                    # Compute grouping plan (returns: method, N, G, seed, groups, perm, group_sizes, metrics)
                    custom_plan = compute_grouping_plan(
                        X_train=X_train,
                        method=exp_args.custom_grouping_method,
                        G=exp_args.num_groups,
                        seed=exp_args.seed,
                        verbose=True,
                        alpha=exp_args.group_alpha,
                        min_size=exp_args.group_min_size,
                        pred_len=exp_args.pred_len
                    )

                    # Extract outputs
                    indices = np.array(custom_plan['perm'], dtype=np.int64)
                    group_sizes = None  # Force uniform window partitioning — permutation만 사용
                    resolved_partition = 'window'
                    resolved_strategy = 'custom_graph'  # Mark as custom

                    # Print metrics
                    metrics = custom_plan.get('metrics', {})
                    print(f">>> Custom Grouping Complete:")
                    print(f"    Permutation (first 10): {indices[:10].tolist()}")
                    print(f"    Group Sizes: uniform (window path forced)")
                    print(f"    CV(size): {metrics.get('CV_size', 'N/A'):.3f}")
                    if 'intra_cohesion' in metrics:
                        print(f"    Intra-cohesion: {metrics['intra_cohesion']:.3f}")
                        print(f"    Inter-separation: {metrics['inter_separation']:.3f}")
                else:
                    # Use existing GroupingManager
                    print(
                        f"Initializing Custom Grouping: method={exp_args.grouping_method}, "
                        f"strategy={exp_args.grouping_strategy}, partition={exp_args.group_partition}..."
                    )
                    # Sample some data for clustering (e.g., first 10000 samples to keep it fast)
                    sample_size = min(10000, len(train_data))
                    data_sample = train_data.data_x[:sample_size]

                    group_plan = GroupingManager.get_group_plan(
                        data_sample,
                        exp_args.grouping_method,
                        exp_args.num_groups,
                        exp_args.grouping_horizon,
                        partition=exp_args.group_partition,
                        balance_alpha=exp_args.group_balance_alpha,
                        strategy=exp_args.grouping_strategy,
                    )
                    indices = group_plan['indices']
                    group_sizes = group_plan.get('group_sizes', None)
                    resolved_partition = group_plan.get('partition', exp_args.group_partition)
                    resolved_strategy = group_plan.get('group_strategy', exp_args.grouping_strategy)

                    print(f"Custom Grouping Initialized. Indices: {indices[:10]}...")
                    if group_sizes is not None:
                        print(f"Ragged Group Sizes (first 10): {group_sizes[:10]}")

                # Inject indices into the model (common for both paths)
                if hasattr(exp.model, 'module'): # DataParallel
                    exp.model.module.variate_indices = torch.from_numpy(indices).long().to(exp.device)
                    exp.model.module.inverse_indices = torch.from_numpy(np.argsort(indices)).long().to(exp.device)
                    exp.model.module.group_sizes = group_sizes
                    exp.model.module.group_partition = resolved_partition
                    exp.model.module.group_strategy = resolved_strategy
                else:
                    exp.model.variate_indices = torch.from_numpy(indices).long().to(exp.device)
                    exp.model.inverse_indices = torch.from_numpy(np.argsort(indices)).long().to(exp.device)
                    exp.model.group_sizes = group_sizes
                    exp.model.group_partition = resolved_partition
                    exp.model.group_strategy = resolved_strategy

                args.group_partition = resolved_partition
                args.grouping_strategy = resolved_strategy
                exp.args.group_partition = resolved_partition
                exp.args.grouping_strategy = resolved_strategy

                if group_sizes is not None:
                    args.group_sizes = ",".join([str(int(s)) for s in group_sizes])
                    exp.args.group_sizes = args.group_sizes

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            if args.model.startswith('VG_'):
                print('--- VG-iT Configurations ---')
                print('Pooling: {}'.format(args.pooling))
                print('Num Groups: {}'.format(args.num_groups))
                print('Grouping Strategy: {}'.format(args.grouping_strategy))
                print('Group Partition: {}'.format(args.group_partition))
                print('Grouping Method: {}'.format(args.grouping_method))
                if args.pooling == 'dynamic':
                    print('Variable Resolution: {}'.format(bool(args.use_variable_resolution)))
                    print('Interaction Bridge: {}'.format(bool(args.use_interaction_bridge)))
                print('----------------------------')
            
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if not args.skip_test:
                exp.test(setting)
            else:
                print(f'INFO: --skip_test=1 → exp.test() skipped. Checkpoint at '
                      f'{real_checkpoints_path}/{setting}/checkpoint.pth')

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            if _logger_active:
                sys.stdout = sys.stdout.terminal
                sys.stderr = sys.stderr.terminal

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy,
            ii)

        # Setup checkpoints path
        if args.output_subdir:
            real_checkpoints_path = os.path.join('./checkpoints', args.output_subdir)
        else:
            real_checkpoints_path = './checkpoints/'
        
        import copy
        exp_args = copy.deepcopy(args)
        exp_args.checkpoints = real_checkpoints_path

        exp = Exp(exp_args)  # set experiments
        args.group_sizes = ''
        exp.args.group_sizes = ''

        use_custom_grouping = (
            args.model.startswith('VG_') and
            getattr(args, 'custom_grouping_method', 'none') != 'none'
        )
        requires_group_plan = (
            args.model.startswith('VG_') and
            (use_custom_grouping or
             getattr(args, 'grouping_method', 'ordered') != 'ordered' or
             getattr(args, 'group_partition', 'window') != 'window' or
             getattr(args, 'grouping_strategy', 'auto') != 'auto')
        )
        if requires_group_plan:
            train_data, _ = exp._get_data(flag='train')

            if use_custom_grouping:
                # Use graph-based custom grouping (research05 plan)
                print(
                    f"[Test-only] Initializing Custom Graph-Based Grouping: "
                    f"method={args.custom_grouping_method}, G={args.num_groups}, seed={args.seed}..."
                )
                X_train = train_data.data_x
                custom_plan = compute_grouping_plan(
                    X_train=X_train,
                    method=args.custom_grouping_method,
                    G=args.num_groups,
                    seed=args.seed,
                    verbose=True,
                    alpha=args.group_alpha,
                    min_size=args.group_min_size,
                    pred_len=args.pred_len
                )
                indices = np.array(custom_plan['perm'], dtype=np.int64)
                group_sizes = None  # Force uniform window partitioning
                resolved_partition = 'window'
                resolved_strategy = 'custom_graph'

                metrics = custom_plan.get('metrics', {})
                print(f">>> Custom Grouping Complete: CV(size)={metrics.get('CV_size', 'N/A'):.3f}")
            else:
                # Use existing GroupingManager
                sample_size = min(10000, len(train_data))
                data_sample = train_data.data_x[:sample_size]
                group_plan = GroupingManager.get_group_plan(
                    data_sample,
                    args.grouping_method,
                    args.num_groups,
                    args.grouping_horizon,
                    partition=args.group_partition,
                    balance_alpha=args.group_balance_alpha,
                    strategy=args.grouping_strategy,
                )
                indices = group_plan['indices']
                group_sizes = group_plan.get('group_sizes', None)
                resolved_partition = group_plan.get('partition', args.group_partition)
                resolved_strategy = group_plan.get('group_strategy', args.grouping_strategy)

            # Inject indices into the model (common for both paths)
            if hasattr(exp.model, 'module'):
                exp.model.module.variate_indices = torch.from_numpy(indices).long().to(exp.device)
                exp.model.module.inverse_indices = torch.from_numpy(np.argsort(indices)).long().to(exp.device)
                exp.model.module.group_sizes = group_sizes
                exp.model.module.group_partition = resolved_partition
                exp.model.module.group_strategy = resolved_strategy
            else:
                exp.model.variate_indices = torch.from_numpy(indices).long().to(exp.device)
                exp.model.inverse_indices = torch.from_numpy(np.argsort(indices)).long().to(exp.device)
                exp.model.group_sizes = group_sizes
                exp.model.group_partition = resolved_partition
                exp.model.group_strategy = resolved_strategy
            args.group_partition = resolved_partition
            args.grouping_strategy = resolved_strategy
            exp.args.group_partition = resolved_partition
            exp.args.grouping_strategy = resolved_strategy
            if group_sizes is not None:
                args.group_sizes = ",".join([str(int(s)) for s in group_sizes])
                exp.args.group_sizes = args.group_sizes
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
