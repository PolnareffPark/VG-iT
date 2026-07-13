# Attribution: this file follows public THUML iTransformer / Time-Series-Library
# forecasting scaffold conventions (MIT); see repository-level THIRD_PARTY_NOTICES.md.

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import gc
import time
import warnings
import numpy as np
import csv
import datetime
from contextlib import nullcontext

try:
    from thop import profile
except ImportError:
    profile = None

from utils.efficiency_measure import (
    EfficiencyCollector,
    atomic_append_json_record,
    collect_environment_metadata,
    pre_allocate_dummy_tensors_train,
    pre_allocate_dummy_tensors_test,
)

warnings.filterwarnings('ignore')


def _load_checkpoint(model, path):
    """Load checkpoint with automatic _orig_mod. prefix handling for torch.compile'd models."""
    # v6 security (M-1): weights_only=True prevents arbitrary code exec via malicious pickle.
    # state_dict is plain tensor storage; no need for full pickle deserializer.
    state_dict = torch.load(path, weights_only=True)
    model_keys = set(model.state_dict().keys())

    # Case 1: checkpoint has _orig_mod. but model doesn't → strip prefix
    if any(k.startswith('_orig_mod.') for k in state_dict) and not any(k.startswith('_orig_mod.') for k in model_keys):
        state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
    # Case 2: checkpoint lacks _orig_mod. but model expects it → add prefix
    elif not any(k.startswith('_orig_mod.') for k in state_dict) and any(k.startswith('_orig_mod.') for k in model_keys):
        state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._eff_collector = None  # set by train() when --measure_efficiency_only is dummy/real

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        if getattr(self.args, 'use_compile', False):
            model = torch.compile(model, mode='reduce-overhead')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = str(getattr(self.args, 'forecast_loss', 'mse')).lower()
        if loss_name in ('huber', 'smooth_l1'):
            return nn.HuberLoss(delta=float(getattr(self.args, 'huber_delta', 1.0)))
        if loss_name in ('mae', 'l1'):
            return nn.L1Loss()
        criterion = nn.MSELoss()
        return criterion

    def _amp_enabled(self):
        if hasattr(self.args, 'amp_effective'):
            return bool(getattr(self.args, 'amp_effective', False) and self.device.type == 'cuda')
        return bool(getattr(self.args, 'use_amp', False) and self.device.type == 'cuda')

    def _autocast_context(self):
        if not self._amp_enabled():
            return nullcontext()
        amp_dtype = getattr(self.args, 'amp_dtype', 'fp16')
        dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
        return torch.autocast(device_type='cuda', dtype=dtype)

    def _use_grad_scaler(self):
        # GradScaler is typically required for fp16, not bf16.
        return self._amp_enabled() and getattr(self.args, 'amp_dtype', 'fp16') == 'fp16'

    def _pre_sync_group_plan(self):
        """Pre-sync group plan outside compiled region to avoid CUDA graph breaks."""
        model_inner = getattr(self.model, '_orig_mod', self.model)
        if not hasattr(model_inner, '_sync_group_plan_to_encoder'):
            return
        has_marks = not ('PEMS' in self.args.data or 'Solar' in self.args.data)
        n_tokens = self.args.enc_in + (4 if has_marks else 0)
        model_inner._sync_group_plan_to_encoder(n_tokens)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self._pre_sync_group_plan()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                if getattr(self.args, 'noise_std', 0.0) > 0:
                    batch_x = batch_x + torch.randn_like(batch_x) * self.args.noise_std

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                    batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)
                # encoder - decoder
                if getattr(self.args, 'use_compile', False):
                    torch.compiler.cudagraph_mark_step_begin()
                if self._amp_enabled():
                    with self._autocast_context():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking=True)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
                if getattr(self.args, 'max_eval_steps', 0) > 0 and (i + 1) >= self.args.max_eval_steps:
                    break
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        measure_mode_peek = (getattr(self.args, 'measure_efficiency_only', '') or '').strip()
        if measure_mode_peek == 'dummy':
            # dummy efficiency mode는 vali 건너뜀 (worker/RAM 절약)
            vali_data, vali_loader = None, None
        else:
            vali_data, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # v6: dummy 모드에서는 DataLoader worker contention 제거를 위해
        #     iters_per_epoch만 추출 후 train_loader 즉시 해제.
        #     test_loader는 test()에서 재생성되므로 여기서 다루지 않음.
        _v6_dummy_train_iters = None
        if measure_mode_peek == 'dummy':
            _v6_dummy_train_iters = len(train_loader)
            del train_loader, train_data
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            train_loader = None
            train_data = None
            train_steps = _v6_dummy_train_iters
        else:
            train_steps = len(train_loader)

        # --- Efficiency profiling mode (plans/streamed-inventing-dove.md) ---
        measure_mode = (getattr(self.args, 'measure_efficiency_only', '') or '').strip()
        if measure_mode in ('dummy', 'real'):
            self.args.train_epochs = 3
            self._eff_collector = EfficiencyCollector(
                self.device, mode=measure_mode, dummy_warmup_iters=20
            )
            early_stopping = EarlyStopping(patience=999, verbose=False)
            early_stopping.save_checkpoint = lambda *a, **k: None  # disable disk write
            print(f"[EFFICIENCY] measure_efficiency_only={measure_mode}: 3 epochs, no validation, no checkpoint")
        else:
            self._eff_collector = None
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        time_now_overall = time.time()
        is_cuda = (hasattr(self, 'device') and self.device.type == 'cuda')
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        scaler = torch.cuda.amp.GradScaler() if self._use_grad_scaler() else None

        # Build dummy iter source if in dummy efficiency mode (real DataLoader bypassed).
        # v6: pre-allocate GPU tensors once and reuse every iter → no H2D, no
        #     torch.randn on CPU, no allocator churn. has_marks=False yields
        #     x_mark/y_mark as None, matching real path's PEMS/Solar branch.
        if measure_mode == 'dummy':
            _dummy_iters_per_epoch = _v6_dummy_train_iters  # extracted before train_loader was freed
            _has_marks_eff = not ('PEMS' in self.args.data or 'Solar' in self.args.data)
            _v6_dummy_tensors_train = pre_allocate_dummy_tensors_train(
                device=self.device,
                batch_size=self.args.batch_size,
                seq_len=self.args.seq_len,
                label_len=self.args.label_len,
                pred_len=self.args.pred_len,
                N=self.args.enc_in,
                has_marks=_has_marks_eff,
                seed=int(getattr(self.args, 'seed', 2021) or 2021),
            )
            _v6_x = _v6_dummy_tensors_train['x']
            _v6_y = _v6_dummy_tensors_train['y']
            _v6_x_mark = _v6_dummy_tensors_train['x_mark']
            _v6_y_mark = _v6_dummy_tensors_train['y_mark']
            _v6_dec_inp_pre = _v6_dummy_tensors_train['dec_inp']
            def _dummy_train_iter():
                for _ in range(_dummy_iters_per_epoch):
                    # Reuse same GPU tensors every iter — pure compute, no H2D.
                    yield (_v6_x, _v6_y, _v6_x_mark, _v6_y_mark)
        else:
            _dummy_train_iter = None
            _v6_dummy_tensors_train = None
            _v6_dec_inp_pre = None

        self._pre_sync_group_plan()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            iter_source = _dummy_train_iter() if _dummy_train_iter else train_loader
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(iter_source):
                iter_count += 1
                if self._eff_collector is not None:
                    self._eff_collector.begin_train_iter()
                model_optim.zero_grad(set_to_none=True)
                if measure_mode == 'dummy':
                    # v6: pre-alloc GPU tensors already float32 on device.
                    #     Skip .float()/.to()/noise_std/dec_inp recompute — pure compute.
                    dec_inp = _v6_dec_inp_pre
                else:
                    batch_x = batch_x.float().to(self.device, non_blocking=True)
                    batch_y = batch_y.float().to(self.device, non_blocking=True)
                    if getattr(self.args, 'noise_std', 0.0) > 0:
                        batch_x = batch_x + torch.randn_like(batch_x) * self.args.noise_std

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                        batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)

                # encoder - decoder
                if getattr(self.args, 'use_compile', False):
                    torch.compiler.cudagraph_mark_step_begin()
                if self._amp_enabled():
                    with self._autocast_context():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        # v6: batch_y is already on device in BOTH modes
                        #   - dummy: pre-allocated on GPU
                        #   - real: .float().to(device) at line 254
                        # → slice is a view on same device; drop redundant .to()
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]  # v6: redundant .to() dropped
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    
                    # --- A3 Stat Extraction ---
                    try:
                        stats = None
                        if getattr(self.args, 'pooling', 'mean') == 'dynamic':
                            # Plan A: Hierarchical Layer
                            # Assuming single layer encoder, first attn layer
                            if hasattr(self.model, 'encoder'):
                                layer = self.model.encoder.attn_layers[0].attention.inner_attention
                                stats = getattr(layer, 'last_stats', None)
                        elif getattr(self.args, 'use_learnable_grouping', False):
                            # Plan B: VG_iTransformer level
                            stats = getattr(self.model, 'last_group_stats', None)
                        
                        if stats:
                            print("\tGrouping Stats | Entropy: {:.4f} | Eff Groups: {:.2f} | Load Imb: {:.6f}".format(
                                stats['entropy'], stats['eff_groups'], stats['load_imbalance']))
                    except Exception:
                        pass

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self._eff_collector is not None:
                    self._eff_collector.end_train_iter(epoch_idx=epoch)

                if getattr(self.args, 'max_train_steps', 0) > 0 and (i + 1) >= self.args.max_train_steps:
                    print(f"MAX_TRAIN_STEPS reached: {self.args.max_train_steps}")
                    break
                if self.args.debug and (i + 1) >= 10:
                    print("DEBUG MODE: Breaking epoch early after 10 iterations")
                    break


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if self._eff_collector is not None:
                vali_loss = 0.0  # skip validation in efficiency mode
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # After all epochs, capture peak reserved memory (including grads, optimizer states)
        if is_cuda:
            self.train_peak_vram = torch.cuda.max_memory_reserved(self.device) / 1024 / 1024 / 1024  # GB
        else:
            self.train_peak_vram = 0.0
        self.total_train_time = time.time() - time_now_overall
        print("Final Training Peak VRAM: {:.2f}GB".format(self.train_peak_vram))

        if self._eff_collector is not None:
            self._eff_collector.capture_train_vram_peak()
            # v6: release dummy pre-alloc tensors so test()'s VRAM measurement
            #     starts from a clean allocator state.
            if measure_mode == 'dummy' and _v6_dummy_tensors_train is not None:
                del _v6_dummy_tensors_train, _v6_x, _v6_y, _v6_x_mark, _v6_y_mark, _v6_dec_inp_pre
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        if self._eff_collector is None:
            best_model_path = path + '/' + 'checkpoint.pth'
            _load_checkpoint(self.model, best_model_path)

        return self.model

    def profile_model(self):
        """Separate profiling run to avoid impact on main experiment timing"""
        if profile is not None:
            from layers.SelfAttention_Family import FullAttention
            from layers.Hierarchical_Attention import HierarchicalAttention
            # SDPAttentionLayer: SDPA fused kernel is invisible to thop
            try:
                from model.iFlashformer import SDPAttentionLayer
            except ImportError:
                SDPAttentionLayer = None

            def count_full_attention(m, x, y):
                queries, keys, values = x[0], x[1], x[2]
                B, L, H, E = queries.shape
                _, S, _, D = values.shape
                flops_qk = B * H * L * S * E * 2
                flops_av = B * H * L * S * D * 2
                m.total_ops += torch.DoubleTensor([flops_qk + flops_av])

            def count_sdp_attention_layer(m, x, y):
                """Count FLOPs for SDPAttentionLayer (SDPA fused kernel).
                thop's dfs_count does NOT recurse into modules with custom handlers,
                so we must include children's ops (QKV projections) here."""
                queries = x[0]  # (B, L, D_model)
                keys = x[1]     # (B, S, D_model)
                B, L, _ = queries.shape
                _, S, _ = keys.shape
                H = m.n_heads
                D = m.query_projection.out_features // H
                flops_qk = B * H * L * S * D * 2   # Q @ K^T
                flops_av = B * H * L * S * D * 2   # A @ V
                # Aggregate children's ops (nn.Linear projections) since thop won't recurse
                child_ops = sum(c.total_ops.item() for c in m.children() if hasattr(c, 'total_ops'))
                m.total_ops += torch.DoubleTensor([flops_qk + flops_av + child_ops])

            def count_hierarchical_attention(m, x, y):
                queries, keys, values = x[0], x[1], x[2]
                B, N, H, D = queries.shape
                G = m.num_groups
                custom_sizes = getattr(m, 'custom_group_sizes', None)

                if custom_sizes is not None and len(custom_sizes) == G and sum(custom_sizes) == N:
                    flops_local = 0
                    for s in custom_sizes:
                        if s <= 0:
                            continue
                        flops_local += B * H * s * s * D * 2 * 2
                else:
                    pad_len = (G - (N % G)) % G
                    N_padded = N + pad_len
                    M = N_padded // G
                    flops_local = B * G * H * M * M * D * 2 * 2

                # Global attention FLOPs
                if m.pooling == 'statistical':
                    G_global = G * 3
                else:
                    G_global = G
                flops_global = B * H * G_global * G_global * D * 2 * 2

                m.total_ops += torch.DoubleTensor([flops_local + flops_global])

            # Unwrap OptimizedModule to avoid register_buffer conflicts
            model_to_profile = getattr(self.model, '_orig_mod', self.model)
            model_to_profile.eval()

            # Snapshot existing forward hooks so we only remove thop's additions
            pre_hooks = {id(m): set(m._forward_hooks.keys()) for m in model_to_profile.modules()}

            try:
                train_data, _ = self._get_data(flag='train')
                actual_N = train_data.data_x.shape[1]

                dummy_x = torch.randn(1, self.args.seq_len, actual_N).to(self.device, non_blocking=True)
                dummy_mark = torch.randn(1, self.args.seq_len, 4).to(self.device, non_blocking=True) if not ('PEMS' in self.args.data or 'Solar' in self.args.data) else None
                dummy_dec = torch.randn(1, self.args.label_len + self.args.pred_len, actual_N).to(self.device, non_blocking=True)
                dummy_y_mark = torch.randn(1, self.args.label_len + self.args.pred_len, 4).to(self.device, non_blocking=True) if not ('PEMS' in self.args.data or 'Solar' in self.args.data) else None

                # Sync with n_tokens including time marks
                if hasattr(model_to_profile, '_sync_group_plan_to_encoder'):
                    has_marks = not ('PEMS' in self.args.data or 'Solar' in self.args.data)
                    n_tokens = actual_N + (4 if has_marks else 0)
                    model_to_profile._sync_group_plan_to_encoder(n_tokens)

                custom_ops = {
                    FullAttention: count_full_attention,
                    HierarchicalAttention: count_hierarchical_attention
                }
                if SDPAttentionLayer is not None:
                    custom_ops[SDPAttentionLayer] = count_sdp_attention_layer

                flops, params = profile(model_to_profile, inputs=(dummy_x, dummy_mark, dummy_dec, dummy_y_mark),
                                        custom_ops=custom_ops, verbose=False)
                return flops, params
            except Exception as e:
                print(f"Error profiling model: {e}")
                return 0, 0
            finally:
                # Clean up thop residuals: hooks and buffers
                for m in model_to_profile.modules():
                    orig_keys = pre_hooks.get(id(m), set())
                    for key in list(m._forward_hooks.keys()):
                        if key not in orig_keys:
                            del m._forward_hooks[key]
                    m._buffers.pop('total_ops', None)
                    m._buffers.pop('total_params', None)
        return 0, 0

    def profile_actual_flops(self):
        """
        PyTorch profiler를 사용하여 실제 FLOPs 측정.
        학습과 분리하여 overhead 방지.
        """
        try:
            import torch.profiler as profiler
        except ImportError:
            return 0

        # Unwrap OptimizedModule
        model_to_profile = getattr(self.model, '_orig_mod', self.model)
        model_to_profile.eval()

        try:
            train_data, _ = self._get_data(flag='train')
            actual_N = train_data.data_x.shape[1]

            dummy_x = torch.randn(1, self.args.seq_len, actual_N).to(self.device, non_blocking=True)
            dummy_mark = torch.randn(1, self.args.seq_len, 4).to(self.device, non_blocking=True) if not ('PEMS' in self.args.data or 'Solar' in self.args.data) else None
            dummy_dec = torch.randn(1, self.args.label_len + self.args.pred_len, actual_N).to(self.device, non_blocking=True)
            dummy_y_mark = torch.randn(1, self.args.label_len + self.args.pred_len, 4).to(self.device, non_blocking=True) if not ('PEMS' in self.args.data or 'Solar' in self.args.data) else None

            # Sync with n_tokens including time marks
            if hasattr(model_to_profile, '_sync_group_plan_to_encoder'):
                has_marks = not ('PEMS' in self.args.data or 'Solar' in self.args.data)
                n_tokens = actual_N + (4 if has_marks else 0)
                model_to_profile._sync_group_plan_to_encoder(n_tokens)

            with torch.no_grad():
                for _ in range(3):
                    _ = model_to_profile(dummy_x, dummy_mark, dummy_dec, dummy_y_mark)

            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True
            ) as prof:
                with torch.no_grad():
                    _ = model_to_profile(dummy_x, dummy_mark, dummy_dec, dummy_y_mark)

            total_flops = 0
            for event in prof.key_averages():
                if hasattr(event, 'flops') and event.flops > 0:
                    total_flops += event.flops

            return total_flops
        except Exception as e:
            print(f"Error profiling actual FLOPs: {e}")
            return 0


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            _load_checkpoint(self.model, os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'))

        self.model.eval()
        model_inner = getattr(self.model, 'module', self.model)
        model_inner = getattr(model_inner, '_orig_mod', model_inner)

        # 1. Metric-Only Run (isolated)
        print("Running independent metric profiling...")

        # DEBUG: Check if custom_group_sizes are set (Phase 1)
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'attn_layers'):
            for i, layer in enumerate(self.model.encoder.attn_layers):
                attn = layer.attention
                inner = getattr(attn, 'inner_attention', None)
                if inner is not None and hasattr(inner, 'custom_group_sizes'):
                    if inner.custom_group_sizes is not None:
                        print(f"DEBUG [Layer {i}]: custom_group_sizes = {inner.custom_group_sizes[:5]}... (len={len(inner.custom_group_sizes)})")
                    else:
                        print(f"DEBUG [Layer {i}]: custom_group_sizes = None (will use uniform path)")
                    break  # Only check first layer

        if getattr(self.args, 'skip_flops_profiling', 0):
            print("Skipping FLOPs profiling (--skip_flops_profiling=1)")
            total_flops, total_params = 0, 0
        else:
            total_flops, total_params = self.profile_model()
            # Print theoretical FLOPs only (torch.profiler is unreliable for custom ops)
            if total_flops > 0:
                print(f"FLOPs: {total_flops/1e9:.2f}G")

        self._pre_sync_group_plan()
        model_inner = getattr(self.model, 'module', self.model)
        model_inner = getattr(model_inner, '_orig_mod', model_inner)

        # Build dummy/real infer sources. Warmup and measurement must share
        # the same source type (shape + mark handling) so cudnn.benchmark
        # autotune matches the measurement path exactly.
        # v6: dummy uses pre-allocated GPU tensors reused every iter.
        if self._eff_collector is not None and self._eff_collector.mode == 'dummy':
            _t_iters = len(test_loader)
            # v6: free DataLoader workers + pinned memory BEFORE pre-alloc so
            # infer VRAM peak excludes DataLoader pinned buffer overhead and so
            # workers don't contend with main thread during pure-compute measurement.
            # test_data referenced only in `_eff_collector is None` block → safe to del.
            del test_loader, test_data
            import gc as _gc
            _gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            test_loader = None
            test_data = None
            _t_marks = not ('PEMS' in self.args.data or 'Solar' in self.args.data or 'stress' in self.args.data)
            _v6_dummy_tensors_test = pre_allocate_dummy_tensors_test(
                device=self.device,
                batch_size=self.args.batch_size,
                seq_len=self.args.seq_len,
                label_len=self.args.label_len,
                pred_len=self.args.pred_len,
                N=self.args.enc_in,
                has_marks=_t_marks,
                seed=int(getattr(self.args, 'seed', 2021) or 2021),
            )
            _v6_t_x = _v6_dummy_tensors_test['x']
            _v6_t_y = _v6_dummy_tensors_test['y']
            _v6_t_x_mark = _v6_dummy_tensors_test['x_mark']
            _v6_t_y_mark = _v6_dummy_tensors_test['y_mark']
            _v6_t_dec_inp = _v6_dummy_tensors_test['dec_inp']
            def _build_dummy_test_iter(n_iters):
                def _gen():
                    for _ in range(n_iters):
                        yield (_v6_t_x, _v6_t_y, _v6_t_x_mark, _v6_t_y_mark)
                return _gen()
            _warmup_source = _build_dummy_test_iter(5)
            _test_source = _build_dummy_test_iter(_t_iters)
        else:
            _warmup_source = iter(test_loader)
            _test_source = test_loader
            _v6_dummy_tensors_test = None
            _v6_t_dec_inp = None

        print("Warming up...")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(_warmup_source):
                if i >= 5: break
                if self._eff_collector is not None and self._eff_collector.mode == 'dummy':
                    # v6: pre-alloc tensors on GPU already, dec_inp precomputed.
                    dec_inp = _v6_t_dec_inp
                else:
                    batch_x = batch_x.float().to(self.device, non_blocking=True)
                    batch_y = batch_y.float().to(self.device, non_blocking=True)
                    if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'stress' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                        batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)
                if getattr(self.args, 'use_compile', False):
                    torch.compiler.cudagraph_mark_step_begin()
                self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # 2. Main Latency & VRAM Tracking
        is_cuda = (hasattr(self, 'device') and self.device.type == 'cuda')
        if is_cuda:
            torch.cuda.synchronize(self.device) # Barrier
            # efficiency 측정 시 train fragmentation이 infer peak 오염 방지
            if self._eff_collector is not None:
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        start_time = time.time()

        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(_test_source):
                if self._eff_collector is not None and self._eff_collector.mode == 'dummy':
                    # v6: pre-alloc GPU tensors, dec_inp precomputed, no H2D.
                    dec_inp = _v6_t_dec_inp
                else:
                    batch_x = batch_x.float().to(self.device, non_blocking=True)
                    batch_y = batch_y.float().to(self.device, non_blocking=True)
                    if getattr(self.args, 'noise_std', 0.0) > 0:
                        batch_x = batch_x + torch.randn_like(batch_x) * self.args.noise_std

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'stress' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                        batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)
                # encoder - decoder
                if self._eff_collector is not None:
                    self._eff_collector.begin_infer_iter()
                if getattr(self.args, 'use_compile', False):
                    torch.compiler.cudagraph_mark_step_begin()
                if self._amp_enabled():
                    with self._autocast_context():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self._eff_collector is not None:
                    self._eff_collector.end_infer_iter()


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # v6: batch_y already on device; drop redundant .to() (see train loop note)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # Efficiency measure mode: skip pred/true CPU transfer + accumulation.
                # These are only needed for MSE/MAE computation which is non-authoritative
                # in measure mode (record.mse_skipped=True). Skipping saves ~30-300s per combo
                # by avoiding GPU→CPU sync on 282M-element tensors.
                if self._eff_collector is None:
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    pred = outputs
                    true = batch_y
                    preds.append(pred)
                    trues.append(true)
                if getattr(self.args, 'max_eval_steps', 0) > 0 and (i + 1) >= self.args.max_eval_steps:
                    print(f"MAX_EVAL_STEPS reached in test loop: {self.args.max_eval_steps}")
                    break

        # Get peak VRAM usage for inference
        if is_cuda:
            inference_peak_vram = torch.cuda.max_memory_reserved(self.device) / 1024 / 1024 / 1024 # GB
        else:
            inference_peak_vram = 0.0
        if self._eff_collector is not None:
            self._eff_collector.capture_infer_vram_peak()
            # v6: release dummy pre-alloc test tensors.
            if self._eff_collector.mode == 'dummy' and _v6_dummy_tensors_test is not None:
                del _v6_dummy_tensors_test, _v6_t_x, _v6_t_y, _v6_t_x_mark, _v6_t_y_mark, _v6_t_dec_inp
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        inference_time_total = time.time() - start_time
        processed_steps = max(1, len(preds)) if self._eff_collector is None else 1
        inference_latency = inference_time_total / processed_steps  # s/iter
        inference_speed = (processed_steps * self.args.batch_size) / max(inference_time_total, 1e-12)  # items/sec

        if self._eff_collector is not None:
            # Skip np.array + reshape + metric() on empty lists (fast path for measure mode)
            mse, mae = 0.0, 0.0
            self._dump_efficiency_record(setting, total_flops, total_params, mse, mae)
            return

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # Log BOTH training and inference metrics for transparency
        train_vram = getattr(self, 'train_peak_vram', 0)
        train_speed = getattr(self, 'total_train_time', 0) / max(self.args.train_epochs, 1)
        
        print('FLOPs: {:.2f}G, Params: {:.2f}M'.format(total_flops / 1e9, total_params / 1e6))
        print('Training: Peak VRAM {:.2f}GB, Speed {:.4f}s/epoch'.format(train_vram, train_speed))
        print('Inference: Peak VRAM {:.2f}GB, Latency {:.4f}s/iter, Speed {:.2f} items/sec'.format(inference_peak_vram, inference_latency, inference_speed))
        
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, flops:{:.4f}G, params:{:.4f}M, train_vram:{:.4f}GB, train_time:{:.4f}s, test_vram:{:.4f}GB, infer_latency:{:.4f}s, infer_speed:{:.2f}items/s\n'.format(
                mse, mae, total_flops/1e9, total_params/1e6, train_vram, train_speed, inference_peak_vram, inference_latency, inference_speed))
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        print("Logged to result_long_term_forecast.txt")

        # New CSV logging with isolation support
        summary_filename = getattr(self.args, 'summary_file', 'summary.csv')
        summary_path = os.path.join('./test_results', summary_filename)
        
        if not os.path.exists('./test_results'):
            os.makedirs('./test_results')
            
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as csvfile:
            headers = ['timestamp', 'model_id', 'model', 'data', 'mse', 'mae', 'flops_G', 'params_M',
                       'train_vram_GB', 'train_time_s', 'test_vram_GB',
                       'infer_latency_s', 'infer_speed_items_s', 'num_groups', 'pooling', 'dynamic_VR', 'dynamic_Bridge', 'dynamic_tokens',
                       'reorder', 'group_strategy', 'group_partition', 'grouping_method', 'group_sizes', 'cv_group_sizes',
                       'amp', 'amp_dtype', 'seed', 'setting']
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            
            # Calculate additional metrics: CV of group sizes
            cv_group_sizes_str = ""
            group_sizes_str = getattr(self.args, 'group_sizes', '')
            if group_sizes_str:
                try:
                    group_sizes_list = [int(s) for s in group_sizes_str.split(',')]
                    if len(group_sizes_list) > 0:
                        cv_group_sizes = np.std(group_sizes_list) / np.mean(group_sizes_list)
                        cv_group_sizes_str = f"{cv_group_sizes:.4f}"
                except:
                    pass

            writer.writerow({
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_id': self.args.model_id,
                'model': self.args.model,
                'data': self.args.data,
                'mse': mse,
                'mae': mae,
                'flops_G': f"{total_flops / 1e9:.4f}",
                'params_M': f"{total_params / 1e6:.4f}",
                'train_vram_GB': f"{train_vram:.4f}",
                'train_time_s': f"{train_speed:.4f}",
                'test_vram_GB': f"{inference_peak_vram:.4f}",
                'infer_latency_s': f"{inference_latency:.4f}",
                'infer_speed_items_s': f"{inference_speed:.2f}",
                'num_groups': getattr(self.args, 'num_groups', 0),
                'pooling': getattr(self.args, 'pooling', 'statistical'),
                'dynamic_VR': getattr(self.args, 'use_variable_resolution', 1),
                'dynamic_Bridge': getattr(self.args, 'use_interaction_bridge', 1),
                'dynamic_tokens': getattr(self.args, 'dynamic_tokens_per_group', 1),
                'reorder': getattr(self.args, 'use_reorder', 0),
                'group_strategy': getattr(self.args, 'grouping_strategy', 'auto'),
                'group_partition': getattr(self.args, 'group_partition', 'window'),
                'grouping_method': getattr(self.args, 'grouping_method', 'ordered'),
                'group_sizes': group_sizes_str,
                'cv_group_sizes': cv_group_sizes_str,
                'amp': int(getattr(self.args, 'amp_effective', getattr(self.args, 'use_amp', False))),
                'amp_dtype': getattr(self.args, 'amp_dtype', 'fp16'),
                'seed': getattr(self.args, 'seed', -1),
                'setting': setting
            })
            csvfile.flush()
            os.fsync(csvfile.fileno())
        print("Logged to summary.csv")

        return


    def _dump_efficiency_record(self, setting, total_flops, total_params, mse, mae):
        """Run batch=1 infer sub-measurement and atomically append a record to JSON.

        Triggered when --measure_efficiency_only is 'dummy' or 'real'. Skips the
        normal result_long_term_forecast.txt + summary.csv side-effects so that
        ground-truth MSE/MAE files remain untouched (per plan: no MSE/MAE rerun).
        """
        collector = self._eff_collector
        measure_mode = (getattr(self.args, 'measure_efficiency_only', '') or '').strip()

        # --- Compose record ---
        record = {
            'model': self.args.model,
            'dataset': self.args.data,
            'data_path': getattr(self.args, 'data_path', ''),
            'N': self.args.enc_in,
            'pred_len': self.args.pred_len,
            'seq_len': self.args.seq_len,
            'label_len': self.args.label_len,
            'batch_size': self.args.batch_size,
            'num_workers': getattr(self.args, 'num_workers', 0),
            'e_layers': self.args.e_layers,
            'd_model': self.args.d_model,
            'd_ff': self.args.d_ff,
            'n_heads': self.args.n_heads,
            'amp': bool(getattr(self.args, 'amp_effective', getattr(self.args, 'use_amp', False))),
            'amp_dtype': getattr(self.args, 'amp_dtype', 'bf16'),
            'seed': getattr(self.args, 'seed', -1),
            'params_M': round(total_params / 1e6, 6),
            'flops_G_thop_hook': round(total_flops / 1e9, 6),
            'mse_skipped': True,
            'mae_skipped': True,
            'mse_observed_short_train': float(mse),
            'mae_observed_short_train': float(mae),
            'setting': setting,
            'model_id': self.args.model_id,
            'entry_point': 'run.py',
        }
        # VG-iT specific
        if str(self.args.model).startswith('VG_'):
            record['num_groups'] = getattr(self.args, 'num_groups', 0)
            record['pooling'] = getattr(self.args, 'pooling', 'mean')
            record['group_partition'] = getattr(self.args, 'group_partition', 'window')
            record['use_shifted_grouping'] = bool(getattr(self.args, 'use_shifted_grouping', 0))
            record['use_film_broadcast'] = bool(getattr(self.args, 'use_film_broadcast', 0))
            record['global_kernel_type'] = getattr(self.args, 'global_kernel_type', 'dot')
            record['local_kernel_type'] = getattr(self.args, 'local_kernel_type', 'dot')

        # Train sec/epoch (wall clock, 전체 실험 수행시간 보고용)
        total_time = getattr(self, 'total_train_time', 0.0)
        record['train_sec_per_epoch'] = round(total_time / max(self.args.train_epochs, 1), 6)
        record['total_train_time_sec'] = round(total_time, 6)
        record['train_epochs'] = int(self.args.train_epochs)

        record.update(collector.to_dict())

        # --- Output JSON path ---
        default_out = f'./test_results/efficiency_profile_{measure_mode}.json'
        out_path = getattr(self.args, 'measure_efficiency_output', default_out) or default_out
        metadata = {
            'data_source': measure_mode,
            'protocol': 'plans/streamed-inventing-dove.md',
            'epochs_equivalent': 3,
            'time_warmup_epochs_excluded': 1 if measure_mode == 'real' else 0,
            'dummy_iter_warmup': 20 if measure_mode == 'dummy' else 0,
            'vram_measurement_scope': 'full_3_epochs',
            'validation_excluded': True,
            'checkpoint_saving_disabled': True,
            'early_stopping_disabled': True,
            'time_statistic_primary': 'median',
            # v6 protocol identifiers
            'dummy_tensor_placement': 'gpu_preallocated' if measure_mode == 'dummy' else 'n/a',
            'dummy_worker_contention': 'excluded' if measure_mode == 'dummy' else 'included',
            'data_source_scope': 'compute_only' if measure_mode == 'dummy' else 'full_pipeline',
        }
        metadata.update(collect_environment_metadata(self.device, seed=record['seed']))
        atomic_append_json_record(out_path, record, metadata=metadata)
        print(f"[EFFICIENCY] record appended to {out_path}")
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            _load_checkpoint(self.model, best_model_path)

        preds = []

        self.model.eval()
        self._pre_sync_group_plan()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)
                # encoder - decoder
                if getattr(self.args, 'use_compile', False):
                    torch.compiler.cudagraph_mark_step_begin()
                if self._amp_enabled():
                    with self._autocast_context():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)
                if getattr(self.args, 'max_eval_steps', 0) > 0 and (i + 1) >= self.args.max_eval_steps:
                    print(f"MAX_EVAL_STEPS reached in predict loop: {self.args.max_eval_steps}")
                    break

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        return
