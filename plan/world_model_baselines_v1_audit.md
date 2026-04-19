# world_model_baselines_v1 audit notes

## Dataset reuse (world_model_baselines_v1.py:969-981)

```text
969: def prepare_dataset(
970:     dataset_id: str,
971:     *,
972:     variant_spec: ExperimentVariant | None = None,
973:     cache_root: str | Path = _CACHE_ROOT,
974:     max_train_origins: int | None = None,
975:     max_eval_origins: int | None = None,
976: ) -> state_base.PreparedDataset:
977:     _validate_dataset_ids((dataset_id,))
978:     resolved_variant = variant_spec or VARIANT_SPECS[0]
979:     prepared = state_base.prepare_dataset(
980:         dataset_id,
981:         cache_root=cache_root,
```

## Default profiles (world_model_baselines_v1.py:430-556)

```text
430:     max_epochs=DEFAULT_MAX_EPOCHS,
431:     early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
432:     d_model=DEFAULT_D_MODEL,
433:     lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
434:     attention_heads=DEFAULT_ATTENTION_HEADS,
435:     patch_len=None,
436:     encoder_layers=None,
437:     ff_hidden_dim=None,
438:     residual_channels=None,
439:     skip_channels=None,
440:     end_channels=None,
441:     gcn_depth=None,
442:     mtgnn_layers=None,
443:     subgraph_size=None,
444:     node_embed_dim=None,
445:     dilation_exponential=None,
446:     propalpha=None,
447:     hidden_dim=None,
448:     embed_dim=None,
449:     num_layers=None,
450:     cheb_k=None,
451:     teacher_forcing_ratio=None,
452:     dropout=DEFAULT_DROPOUT,
453:     grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
454:     weight_decay=DEFAULT_WEIGHT_DECAY,
455:     bounded_output_epsilon=DEFAULT_BOUNDED_OUTPUT_EPSILON,
456: )
457: _DEFAULT_TIMEXER_PROFILE = HyperparameterProfile(
458:     batch_size=DEFAULT_BATCH_SIZE,
459:     learning_rate=DEFAULT_LEARNING_RATE,
460:     max_epochs=DEFAULT_MAX_EPOCHS,
461:     early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
462:     d_model=DEFAULT_D_MODEL,
463:     lstm_hidden_dim=None,
464:     attention_heads=DEFAULT_ATTENTION_HEADS,
465:     patch_len=DEFAULT_TIMEXER_PATCH_LEN,
466:     encoder_layers=DEFAULT_TIMEXER_ENCODER_LAYERS,
467:     ff_hidden_dim=DEFAULT_TIMEXER_FF_HIDDEN_DIM,
468:     residual_channels=None,
469:     skip_channels=None,
470:     end_channels=None,
471:     gcn_depth=None,
472:     mtgnn_layers=None,
473:     subgraph_size=None,
474:     node_embed_dim=None,
475:     dilation_exponential=None,
476:     propalpha=None,
477:     hidden_dim=None,
478:     embed_dim=None,
479:     num_layers=None,
480:     cheb_k=None,
481:     teacher_forcing_ratio=None,
482:     dropout=DEFAULT_DROPOUT,
483:     grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
484:     weight_decay=DEFAULT_WEIGHT_DECAY,
485:     bounded_output_epsilon=DEFAULT_BOUNDED_OUTPUT_EPSILON,
486: )
487: _DEFAULT_DGCRN_PROFILE = HyperparameterProfile(
488:     batch_size=256,
489:     learning_rate=5e-4,
490:     max_epochs=20,
491:     early_stopping_patience=5,
492:     d_model=None,
493:     lstm_hidden_dim=None,
494:     attention_heads=None,
495:     patch_len=None,
496:     encoder_layers=None,
497:     ff_hidden_dim=None,
498:     residual_channels=None,
499:     skip_channels=None,
500:     end_channels=None,
501:     gcn_depth=None,
502:     mtgnn_layers=None,
503:     subgraph_size=None,
504:     node_embed_dim=None,
505:     dilation_exponential=None,
506:     propalpha=None,
507:     hidden_dim=DEFAULT_HIDDEN_DIM,
508:     embed_dim=DEFAULT_EMBED_DIM,
509:     num_layers=DEFAULT_NUM_LAYERS,
510:     cheb_k=DEFAULT_CHEB_K,
511:     teacher_forcing_ratio=DEFAULT_TEACHER_FORCING_RATIO,
512:     dropout=None,
513:     grad_clip_norm=5.0,
514:     weight_decay=0.0,
515:     bounded_output_epsilon=None,
516: )
517: _DEFAULT_ITRANSFORMER_PROFILE = HyperparameterProfile(
518:     batch_size=64,
519:     learning_rate=1e-4,
520:     max_epochs=30,
521:     early_stopping_patience=10,
522:     d_model=64,
523:     lstm_hidden_dim=None,
524:     attention_heads=4,
525:     patch_len=None,
526:     encoder_layers=None,
527:     ff_hidden_dim=None,
528:     residual_channels=None,
529:     skip_channels=None,
530:     end_channels=None,
531:     gcn_depth=None,
532:     mtgnn_layers=None,
533:     subgraph_size=None,
534:     node_embed_dim=None,
535:     dilation_exponential=None,
536:     propalpha=None,
537:     hidden_dim=None,
538:     embed_dim=None,
539:     num_layers=None,
540:     cheb_k=None,
541:     teacher_forcing_ratio=None,
542:     dropout=0.1,
543:     grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
544:     weight_decay=1e-4,
545:     bounded_output_epsilon=DEFAULT_BOUNDED_OUTPUT_EPSILON,
546: )
547: _DEFAULT_MTGNN_PROFILE = HyperparameterProfile(
548:     batch_size=128,
549:     learning_rate=1e-3,
550:     max_epochs=DEFAULT_MAX_EPOCHS,
551:     early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
552:     d_model=None,
553:     lstm_hidden_dim=None,
554:     attention_heads=None,
555:     patch_len=None,
556:     encoder_layers=None,
```

## TFT early stopping (world_model_baselines_v1.py:3319-3331)

```text
3319:             val_mae_pu = float(val_metrics.mae_pu)
3320:             val_rmse_pu = float(val_metrics.rmse_pu)
3321:             is_best_epoch = False
3322:             if best_state is None or (
3323:                 math.isfinite(val_mae_pu)
3324:                 and (not math.isfinite(best_val_mae_pu) or val_mae_pu < best_val_mae_pu - 1e-12)
3325:             ):
3326:                 best_val_mae_pu = val_mae_pu
3327:                 best_val_rmse_pu = val_rmse_pu
3328:                 best_epoch = epoch_index
3329:                 best_state = copy.deepcopy(model.state_dict())
3330:                 epochs_without_improvement = 0
3331:                 is_best_epoch = True
```

## TimeXer early stopping (world_model_baselines_v1.py:3625-3637)

```text
3625:             val_mae_pu = float(val_metrics.mae_pu)
3626:             val_rmse_pu = float(val_metrics.rmse_pu)
3627:             is_best_epoch = False
3628:             if best_state is None or (
3629:                 math.isfinite(val_mae_pu)
3630:                 and (not math.isfinite(best_val_mae_pu) or val_mae_pu < best_val_mae_pu - 1e-12)
3631:             ):
3632:                 best_val_mae_pu = val_mae_pu
3633:                 best_val_rmse_pu = val_rmse_pu
3634:                 best_epoch = epoch_index
3635:                 best_state = copy.deepcopy(model.state_dict())
3636:                 epochs_without_improvement = 0
3637:                 is_best_epoch = True
```

## iTransformer early stopping (world_model_baselines_v1.py:3943-3955)

```text
3943:             val_mae_pu = float(val_metrics.mae_pu)
3944:             val_rmse_pu = float(val_metrics.rmse_pu)
3945:             is_best_epoch = False
3946:             if best_state is None or (
3947:                 math.isfinite(val_mae_pu)
3948:                 and (not math.isfinite(best_val_mae_pu) or val_mae_pu < best_val_mae_pu - 1e-12)
3949:             ):
3950:                 best_val_mae_pu = val_mae_pu
3951:                 best_val_rmse_pu = val_rmse_pu
3952:                 best_epoch = epoch_index
3953:                 best_state = copy.deepcopy(model.state_dict())
3954:                 epochs_without_improvement = 0
3955:                 is_best_epoch = True
```

## MTGNN early stopping (world_model_baselines_v1.py:4284-4294)

```text
4284:             val_mae_pu = float(val_metrics.mae_pu)
4285:             val_rmse_pu = float(val_metrics.rmse_pu)
4286:             is_best_epoch = False
4287:             if best_state is None or (
4288:                 math.isfinite(val_mae_pu)
4289:                 and (not math.isfinite(best_val_mae_pu) or val_mae_pu < best_val_mae_pu - 1e-12)
4290:             ):
4291:                 best_val_mae_pu = val_mae_pu
4292:                 best_val_rmse_pu = val_rmse_pu
4293:                 best_epoch = epoch_index
4294:                 best_state = copy.deepcopy(model.state_dict())
```

## DGCRN early stopping (world_model_baselines_v1.py:4599-4607)

```text
4599:             val_rmse_pu = float(val_metrics.rmse_pu)
4600:             is_best_epoch = False
4601:             if best_state is None or (
4602:                 math.isfinite(val_rmse_pu)
4603:                 and (not math.isfinite(best_val_rmse_pu) or val_rmse_pu < best_val_rmse_pu - 1e-12)
4604:             ):
4605:                 best_val_mae_pu = val_mae_pu
4606:                 best_val_rmse_pu = val_rmse_pu
4607:                 best_epoch = epoch_index
```

## Loss choices TFT (world_model_baselines_v1.py:3277-3283)

```text
3277:                         predictions = model(
3278:                             batch_local_history,
3279:                             batch_context_history,
3280:                             batch_context_future,
3281:                             batch_static,
3282:                         )
3283:                         loss = masked_huber_loss(predictions, batch_targets, batch_valid, torch_module=resolved_torch)
```

## Loss choices TimeXer (world_model_baselines_v1.py:3584-3589)

```text
3584:                         predictions = model(
3585:                             batch_endogenous_history,
3586:                             batch_exogenous_history,
3587:                             batch_history_marks,
3588:                         )
3589:                         loss = masked_mse_loss(predictions, batch_targets, batch_valid, torch_module=resolved_torch)
```

## Loss choices iTransformer (world_model_baselines_v1.py:3896-3903)

```text
3896:                         predictions = model(
3897:                             batch_local_history,
3898:                             batch_context_history,
3899:                             batch_context_future,
3900:                             batch_static,
3901:                         )
3902:                         loss = masked_huber_loss(
3903:                             predictions,
```

## Loss choices MTGNN (world_model_baselines_v1.py:4238-4246)

```text
4238:                         predictions = model(
4239:                             batch_local_history,
4240:                             batch_context_history,
4241:                             batch_future_calendar,
4242:                         )
4243:                         loss = masked_huber_loss(
4244:                             predictions,
4245:                             batch_targets,
4246:                             batch_valid,
```

## Loss choices DGCRN (world_model_baselines_v1.py:4560-4568)

```text
4560:                     predictions = model(
4561:                         batch_history,
4562:                         batch_known_future,
4563:                         batch_targets,
4564:                         teacher_forcing_ratio=teacher_forcing_ratio,
4565:                     )
4566:                     loss = masked_mse_loss(
4567:                         predictions,
4568:                         batch_targets,
```

## Dataset surfaces TFT/TimeXer/iTransformer/Graph (world_model_baselines_v1.py:1412-1522)

```text
1412:             item = (
1413:                 prepared.local_history_tensor[history_slice, node_index, :].astype(np.float32, copy=True),
1414:                 prepared.context_history_tensor[history_slice, :].astype(np.float32, copy=True),
1415:                 prepared.context_future_tensor[future_slice, :].astype(np.float32, copy=True),
1416:                 prepared.static_tensor[node_index, :].astype(np.float32, copy=True),
1417:                 prepared.target_pu_filled[future_slice, node_index, None].astype(np.float32, copy=True),
1418:                 prepared.target_valid_mask[future_slice, node_index, None].astype(np.float32, copy=True),
1419:             )
1420:             if not self.include_indices:
1421:                 return item
1422:             return (*item, np.int64(window_pos), np.int64(node_index))
1423: 
1424: 
1425:     class TimeXerWindowDataset(Dataset):
1426:         def __init__(
1427:             self,
1428:             prepared_dataset: state_base.PreparedDataset,
1429:             windows: world_model_base.FarmWindowDescriptorIndex,
1430:             *,
1431:             include_indices: bool = False,
1432:         ) -> None:
1433:             self.prepared_dataset = prepared_dataset
1434:             self.windows = windows
1435:             self.include_indices = include_indices
1436: 
1437:         def __len__(self) -> int:
1438:             return len(self.windows) * self.prepared_dataset.node_count
1439: 
1440:         def __getitem__(self, index: int):
1441:             prepared = self.prepared_dataset
1442:             window_pos = int(index) // prepared.node_count
1443:             node_index = int(index) % prepared.node_count
1444:             target_index = int(self.windows.target_indices[window_pos])
1445:             history_slice = slice(target_index - prepared.history_steps, target_index)
1446:             future_slice = slice(target_index, target_index + prepared.forecast_steps)
1447:             endogenous_history, exogenous_history, history_marks = _timexer_history_inputs(
1448:                 prepared,
1449:                 history_slice=history_slice,
1450:                 node_index=node_index,
1451:             )
1452:             item = (
1453:                 endogenous_history,
1454:                 exogenous_history,
1455:                 history_marks,
1456:                 prepared.target_pu_filled[future_slice, node_index, None].astype(np.float32, copy=True),
1457:                 prepared.target_valid_mask[future_slice, node_index, None].astype(np.float32, copy=True),
1458:             )
1459:             if not self.include_indices:
1460:                 return item
1461:             return (*item, np.int64(window_pos), np.int64(node_index))
1462: 
1463: 
1464:     class ITransformerWindowDataset(Dataset):
1465:         def __init__(
1466:             self,
1467:             prepared_dataset: state_base.PreparedDataset,
1468:             windows: world_model_base.FarmWindowDescriptorIndex,
1469:             *,
1470:             include_indices: bool = False,
1471:         ) -> None:
1472:             self.prepared_dataset = prepared_dataset
1473:             self.windows = windows
1474:             self.include_indices = include_indices
1475: 
1476:         def __len__(self) -> int:
1477:             return len(self.windows)
1478: 
1479:         def __getitem__(self, index: int):
1480:             prepared = self.prepared_dataset
1481:             window_pos = int(index)
1482:             target_index = int(self.windows.target_indices[window_pos])
1483:             history_slice = slice(target_index - prepared.history_steps, target_index)
1484:             future_slice = slice(target_index, target_index + prepared.forecast_steps)
1485:             item = (
1486:                 prepared.local_history_tensor[history_slice, :, :].astype(np.float32, copy=True),
1487:                 prepared.context_history_tensor[history_slice, :].astype(np.float32, copy=True),
1488:                 prepared.context_future_tensor[future_slice, :].astype(np.float32, copy=True),
1489:                 prepared.static_tensor[:, :].astype(np.float32, copy=True),
1490:                 prepared.target_pu_filled[future_slice, :, None].astype(np.float32, copy=True),
1491:                 prepared.target_valid_mask[future_slice, :, None].astype(np.float32, copy=True),
1492:             )
1493:             if not self.include_indices:
1494:                 return item
1495:             return (*item, np.int64(window_pos))
1496: 
1497: 
1498:     class GraphWindowDataset(Dataset):
1499:         def __init__(
1500:             self,
1501:             prepared_dataset: state_base.PreparedDataset,
1502:             windows: world_model_base.FarmWindowDescriptorIndex,
1503:             *,
1504:             include_indices: bool = False,
1505:         ) -> None:
1506:             self.prepared_dataset = prepared_dataset
1507:             self.windows = windows
1508:             self.include_indices = include_indices
1509: 
1510:         def __len__(self) -> int:
1511:             return len(self.windows)
1512: 
1513:         def __getitem__(self, index: int):
1514:             prepared = self.prepared_dataset
1515:             window_pos = int(index)
1516:             target_index = int(self.windows.target_indices[window_pos])
1517:             history_slice = slice(target_index - prepared.history_steps, target_index)
1518:             future_slice = slice(target_index, target_index + prepared.forecast_steps)
1519:             item = (
1520:                 prepared.local_history_tensor[history_slice, :, :].astype(np.float32, copy=True),
1521:                 prepared.context_history_tensor[history_slice, :].astype(np.float32, copy=True),
1522:                 prepared.context_future_tensor[future_slice, :].astype(np.float32, copy=True),
```

## DGCRN forward and teacher forcing (world_model_baselines_v1.py:2280-2367)

```text
2280:             return cell.dynamic_adjacency(hidden_state, static_embeddings, pairwise_bias)
2281: 
2282:         def forward(
2283:             self,
2284:             history,
2285:             known_future,
2286:             targets=None,
2287:             teacher_forcing_ratio: float = DEFAULT_TEACHER_FORCING_RATIO,
2288:         ):
2289:             if history.ndim != 4:
2290:                 raise ValueError(
2291:                     f"Expected history with shape [batch, history, nodes, channels], got {history.shape!r}."
2292:                 )
2293:             if known_future.ndim != 3:
2294:                 raise ValueError(
2295:                     f"Expected known_future with shape [batch, horizon, channels], got {known_future.shape!r}."
2296:                 )
2297:             if history.shape[2] != self.node_count:
2298:                 raise ValueError(f"Expected node_count={self.node_count}, received {history.shape[2]}.")
2299:             if history.shape[3] != self.history_input_channels:
2300:                 raise ValueError(
2301:                     f"Expected history_input_channels={self.history_input_channels}, received {history.shape[3]}."
2302:                 )
2303:             if known_future.shape[1] != self.forecast_steps:
2304:                 raise ValueError(f"Expected forecast_steps={self.forecast_steps}, received {known_future.shape[1]}.")
2305:             if known_future.shape[2] != self.context_future_channels:
2306:                 raise ValueError(
2307:                     f"Expected context_future_channels={self.context_future_channels}, received {known_future.shape[2]}."
2308:                 )
2309: 
2310:             batch_size = history.shape[0]
2311:             static_embeddings = self.static_node_embeddings().to(device=history.device, dtype=history.dtype)
2312:             pairwise_bias = self.pairwise_bias().to(device=history.device, dtype=history.dtype)
2313:             encoder_states = [
2314:                 history.new_zeros((batch_size, self.node_count, self.hidden_dim)) for _ in range(self.num_layers)
2315:             ]
2316:             current_inputs = history
2317:             for layer_index, cell in enumerate(self.encoder):
2318:                 state = encoder_states[layer_index]
2319:                 layer_outputs: list[Any] = []
2320:                 for time_index in range(history.shape[1]):
2321:                     state, supports = cell(current_inputs[:, time_index, :, :], state, static_embeddings, pairwise_bias)
2322:                     layer_outputs.append(state)
2323:                     self.last_dynamic_adjacency = supports.detach()
2324:                 encoder_states[layer_index] = state
2325:                 current_inputs = torch.stack(layer_outputs, dim=1)
2326: 
2327:             decoder_states = [state.clone() for state in encoder_states]
2328:             previous_target = history[:, -1, :, :1]
2329:             outputs: list[Any] = []
2330:             for horizon_index in range(self.forecast_steps):
2331:                 future_step = known_future[:, horizon_index, :][:, None, :].expand(-1, self.node_count, -1)
2332:                 current_inputs = torch.cat((previous_target, future_step), dim=-1)
2333:                 next_states: list[Any] = []
2334:                 for layer_index, cell in enumerate(self.decoder):
2335:                     state, supports = cell(current_inputs, decoder_states[layer_index], static_embeddings, pairwise_bias)
2336:                     next_states.append(state)
2337:                     current_inputs = state
2338:                     self.last_dynamic_adjacency = supports.detach()
2339:                 decoder_states = next_states
2340:                 projected = self.output_projection(current_inputs)
2341:                 outputs.append(projected)
2342:                 use_teacher = (
2343:                     targets is not None
2344:                     and teacher_forcing_ratio > 0.0
2345:                     and bool(self.training)
2346:                     and float(torch.rand((), device=projected.device).item()) < teacher_forcing_ratio
2347:                 )
2348:                 previous_target = targets[:, horizon_index, :, :] if use_teacher else projected
2349:             return torch.stack(outputs, dim=1)
2350: 
2351: 
2352:     class TimeXerEndogenousEmbedding(nn.Module):
2353:         def __init__(self, *, history_steps: int, patch_len: int, d_model: int, dropout: float) -> None:
2354:             super().__init__()
2355:             if history_steps % patch_len != 0:
2356:                 raise ValueError(f"patch_len must evenly divide history_steps={history_steps}.")
2357:             self.history_steps = history_steps
2358:             self.patch_len = patch_len
2359:             self.patch_count = history_steps // patch_len
2360:             self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
2361:             self.position_embedding = SinusoidalPositionalEmbedding(d_model, max_len=self.patch_count)
2362:             self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
2363:             self.dropout = nn.Dropout(dropout)
2364: 
2365:         def forward(self, endogenous_history):
2366:             if endogenous_history.ndim != 3 or endogenous_history.shape[2] != 1:
2367:                 raise ValueError("TimeXer endogenous history must have shape [batch, history_steps, 1].")
```

## Tests: persistence no leakage (test_world_model_baselines_v1.py:211-237)

```text
211: def test_persistence_uses_last_history_value_without_future_leak(tmp_path, monkeypatch) -> None:
212:     module = _load_module()
213:     prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
214:     windows = prepared.val_rolling_windows
215:     target_index = int(windows.target_indices[0])
216:     history_slice = slice(target_index - prepared.history_steps, target_index)
217:     future_slice = slice(target_index, target_index + prepared.forecast_steps)
218: 
219:     predictions, _targets, _valid = module.persistence_predictions(prepared, windows)
220:     values = prepared.local_history_tensor[history_slice, :, module.state_base._LOCAL_VALUE_START]
221:     unavailable = prepared.local_history_tensor[history_slice, :, module.state_base._LOCAL_MASK_START]
222:     fallback = module._train_history_target_mean(prepared)
223:     expected = fallback.copy()
224:     for node_index in range(prepared.node_count):
225:         valid_positions = np.flatnonzero(unavailable[:, node_index] < 0.5)
226:         if valid_positions.size:
227:             expected[node_index] = values[int(valid_positions[-1]), node_index]
228: 
229:     assert np.allclose(predictions[0, :, :, 0], expected[None, :])
230: 
231:     mutated_targets = prepared.target_pu_filled.copy()
232:     mutated_targets[future_slice] = 1.0 - mutated_targets[future_slice]
233:     mutated_prepared = replace(prepared, target_pu_filled=mutated_targets)
234:     mutated_predictions, _mutated_targets, _mutated_valid = module.persistence_predictions(mutated_prepared, windows)
235: 
236:     assert np.allclose(mutated_predictions, predictions)
237: 
```

## Tests: Chronos adapter (test_world_model_baselines_v1.py:278-318)

```text
278: def test_chronos_input_adapter_restores_nan_and_calendar_covariates(tmp_path, monkeypatch) -> None:
279:     module = _load_module()
280:     prepared = _prepare_temp_dataset(
281:         module,
282:         tmp_path,
283:         monkeypatch,
284:         variant_spec=_chronos_variant_spec(module),
285:     )
286:     target_index = int(prepared.val_rolling_windows.target_indices[0])
287:     history_slice = slice(target_index - prepared.history_steps, target_index)
288:     future_slice = slice(target_index, target_index + prepared.forecast_steps)
289: 
290:     local_history = prepared.local_history_tensor.copy()
291:     context_history = prepared.context_history_tensor.copy()
292:     local_history[history_slice.start, 0, module.state_base._LOCAL_VALUE_START] = 0.4321
293:     local_history[history_slice.start, 0, module.state_base._LOCAL_MASK_START] = 1.0
294:     local_history[history_slice.start, 0, module.state_base._LOCAL_VALUE_START + 1] = 0.6543
295:     local_history[history_slice.start, 0, module.state_base._LOCAL_MASK_START + 1] = 1.0
296:     context_history[history_slice.start, module.state_base._CONTEXT_GLOBAL_VALUE_START] = 0.9876
297:     context_history[history_slice.start, module.state_base._CONTEXT_GLOBAL_MASK_START] = 1.0
298:     mutated_prepared = replace(prepared, local_history_tensor=local_history, context_history_tensor=context_history)
299: 
300:     chronos_input = module.build_chronos_zero_shot_input(
301:         mutated_prepared,
302:         target_index=target_index,
303:         node_index=0,
304:     )
305: 
306:     past_covariates = chronos_input["past_covariates"]
307:     future_covariates = chronos_input["future_covariates"]
308:     assert isinstance(past_covariates, dict)
309:     assert isinstance(future_covariates, dict)
310:     assert np.isnan(np.asarray(chronos_input["target"])[0])
311:     assert np.isnan(
312:         np.asarray(past_covariates[prepared.local_input_feature_names[module.state_base._LOCAL_VALUE_START + 1]])[0]
313:     )
314:     assert np.isnan(
315:         np.asarray(past_covariates[prepared.context_history_feature_names[module.state_base._CONTEXT_GLOBAL_VALUE_START]])[0]
316:     )
317:     assert tuple(future_covariates) == _KNOWN_FUTURE_COLUMNS
318:     assert set(future_covariates).issubset(set(past_covariates))
```

## Tests: persistence/Chronos jobs (test_world_model_baselines_v1.py:779-886)

```text
779: def test_persistence_job_writes_synthetic_training_history(tmp_path, monkeypatch) -> None:
780:     module = _load_module()
781:     prepared = _prepare_temp_dataset(
782:         module,
783:         tmp_path,
784:         monkeypatch,
785:         max_train_origins=2,
786:         max_eval_origins=1,
787:         variant_spec=module.VARIANT_SPECS[0],
788:     )
789:     history_path = tmp_path / "persistence.training_history.csv"
790: 
791:     rows = module.execute_training_job(
792:         prepared,
793:         variant_spec=module.VARIANT_SPECS[0],
794:         seed=123,
795:         training_history_path=history_path,
796:     )
797: 
798:     history = pl.read_csv(history_path)
799:     assert history["epoch"].to_list() == [0]
800:     assert history["baseline_type"].to_list() == ["persistence_last_value"]
801:     assert history["train_loss_mean"].null_count() == 1
802:     assert history["val_rmse_pu"].null_count() == 0
803:     assert len(rows) == 148
804:     assert rows[0]["model_variant"] == module.PERSISTENCE_VARIANT
805:     assert rows[0]["uses_graph"] is False
806: 
807: 
808: def test_chronos_zero_shot_job_reaggregates_batches_and_writes_synthetic_history(tmp_path, monkeypatch) -> None:
809:     module = _load_module()
810:     prepared = _prepare_temp_dataset(
811:         module,
812:         tmp_path,
813:         monkeypatch,
814:         max_train_origins=2,
815:         max_eval_origins=1,
816:         variant_spec=_chronos_variant_spec(module),
817:     )
818:     history_path = tmp_path / "chronos.training_history.csv"
819:     captured_shapes: list[tuple[int, ...]] = []
820:     original_metrics = module._metrics_from_arrays
821: 
822:     def _capture_metrics(predictions, targets, valid_mask, *, rated_power_kw):
823:         captured_shapes.append(predictions.shape)
824:         assert predictions.shape == targets.shape == valid_mask.shape
825:         return original_metrics(predictions, targets, valid_mask, rated_power_kw=rated_power_kw)
826: 
827:     class _FakeChronosPipeline:
828:         def __init__(self) -> None:
829:             self.calls: list[dict[str, object]] = []
830: 
831:         def predict_quantiles(
832:             self,
833:             inputs,
834:             *,
835:             prediction_length,
836:             batch_size,
837:             context_length,
838:             cross_learning,
839:             limit_prediction_length,
840:         ):
841:             self.calls.append(
842:                 {
843:                     "count": len(inputs),
844:                     "prediction_length": prediction_length,
845:                     "batch_size": batch_size,
846:                     "context_length": context_length,
847:                     "cross_learning": cross_learning,
848:                     "limit_prediction_length": limit_prediction_length,
849:                 }
850:             )
851:             means = []
852:             for item in inputs:
853:                 target = np.asarray(item["target"], dtype=np.float32)
854:                 anchor = float(np.nan_to_num(target[-1], nan=0.0))
855:                 means.append(np.full((1, prediction_length), anchor, dtype=np.float32))
856:             return [None] * len(means), means
857: 
858:     fake_pipeline = _FakeChronosPipeline()
859:     monkeypatch.setattr(module, "_metrics_from_arrays", _capture_metrics)
860:     monkeypatch.setattr(module, "_load_chronos_zero_shot_pipeline", lambda *, device: fake_pipeline)
861: 
862:     rows = module.execute_training_job(
863:         prepared,
864:         variant_spec=_chronos_variant_spec(module),
865:         device="cpu",
866:         seed=123,
867:         batch_size=3,
868:         eval_batch_size=2,
869:         training_history_path=history_path,
870:     )
871: 
872:     expected_shape = (len(prepared.val_rolling_windows), prepared.forecast_steps, prepared.node_count, 1)
873:     assert captured_shapes
874:     assert all(shape == expected_shape for shape in captured_shapes)
875:     assert len(fake_pipeline.calls) == 8
876:     assert all(call["batch_size"] == 2 for call in fake_pipeline.calls)
877:     assert all(call["context_length"] == prepared.history_steps for call in fake_pipeline.calls)
878:     assert all(call["cross_learning"] is False for call in fake_pipeline.calls)
879:     history = pl.read_csv(history_path)
880:     assert history["epoch"].to_list() == [0]
881:     assert history["baseline_type"].to_list() == ["chronos_2_zero_shot"]
882:     assert history["train_loss_mean"].null_count() == 1
883:     assert history["device"].to_list() == ["cpu"]
884:     assert len(rows) == 148
885:     assert rows[0]["model_variant"] == module.CHRONOS_VARIANT
886:     assert rows[0]["baseline_type"] == "chronos_2_zero_shot"
```

## Tests: run_experiment output/resume (test_world_model_baselines_v1.py:1151-1290)

```text
1151: def test_run_experiment_writes_results_and_hashed_resume_state(tmp_path, monkeypatch) -> None:
1152:     module = _load_module()
1153:     prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, max_train_origins=2, max_eval_origins=1)
1154:     output_path = tmp_path / "published" / "latest.csv"
1155:     work_root = tmp_path / ".work"
1156: 
1157:     def _dataset_loader(*_args, **_kwargs):
1158:         return prepared
1159: 
1160:     def _job_runner(prepared_dataset, **kwargs):
1161:         variant_spec = kwargs["variant_spec"]
1162:         profile = module.resolve_hyperparameter_profile(
1163:             variant_spec.model_variant,
1164:             dataset_id=prepared_dataset.dataset_id,
1165:             batch_size=kwargs["batch_size"],
1166:             learning_rate=kwargs["learning_rate"],
1167:             max_epochs=kwargs["max_epochs"],
1168:             early_stopping_patience=kwargs["early_stopping_patience"],
1169:             d_model=kwargs["d_model"],
1170:             lstm_hidden_dim=kwargs["lstm_hidden_dim"],
1171:             attention_heads=kwargs["attention_heads"],
1172:             patch_len=kwargs["patch_len"],
1173:             encoder_layers=kwargs["encoder_layers"],
1174:             ff_hidden_dim=kwargs["ff_hidden_dim"],
1175:             residual_channels=kwargs["residual_channels"],
1176:             skip_channels=kwargs["skip_channels"],
1177:             end_channels=kwargs["end_channels"],
1178:             gcn_depth=kwargs["gcn_depth"],
1179:             mtgnn_layers=kwargs["mtgnn_layers"],
1180:             subgraph_size=kwargs["subgraph_size"],
1181:             node_embed_dim=kwargs["node_embed_dim"],
1182:             dilation_exponential=kwargs["dilation_exponential"],
1183:             propalpha=kwargs["propalpha"],
1184:             hidden_dim=kwargs["hidden_dim"],
1185:             embed_dim=kwargs["embed_dim"],
1186:             num_layers=kwargs["num_layers"],
1187:             cheb_k=kwargs["cheb_k"],
1188:             teacher_forcing_ratio=kwargs["teacher_forcing_ratio"],
1189:             dropout=kwargs["dropout"],
1190:             grad_clip_norm=kwargs["grad_clip_norm"],
1191:             weight_decay=kwargs["weight_decay"],
1192:             bounded_output_epsilon=kwargs["bounded_output_epsilon"],
1193:         )
1194:         metrics = module.EvaluationMetrics(
1195:             window_count=1,
1196:             prediction_count=prepared_dataset.forecast_steps * prepared_dataset.node_count,
1197:             mae_kw=1.0,
1198:             rmse_kw=2.0,
1199:             mae_pu=0.1,
1200:             rmse_pu=0.2,
1201:             horizon_window_count=np.ones((prepared_dataset.forecast_steps,), dtype=np.int64),
1202:             horizon_prediction_count=np.full(
1203:                 (prepared_dataset.forecast_steps,),
1204:                 prepared_dataset.node_count,
1205:                 dtype=np.int64,
1206:             ),
1207:             horizon_mae_kw=np.ones((prepared_dataset.forecast_steps,), dtype=np.float64),
1208:             horizon_rmse_kw=np.full((prepared_dataset.forecast_steps,), 2.0, dtype=np.float64),
1209:             horizon_mae_pu=np.full((prepared_dataset.forecast_steps,), 0.1, dtype=np.float64),
1210:             horizon_rmse_pu=np.full((prepared_dataset.forecast_steps,), 0.2, dtype=np.float64),
1211:         )
1212:         evaluation_results = [
1213:             (split_name, eval_protocol, windows, metrics)
1214:             for split_name, eval_protocol, windows in module.iter_evaluation_specs(prepared_dataset)
1215:         ]
1216:         training_outcome = module.TrainingOutcome(
1217:             best_epoch=0,
1218:             epochs_ran=0,
1219:             best_val_rmse_pu=0.2,
1220:             best_val_mae_pu=0.1,
1221:             device="cpu",
1222:             amp_enabled=False,
1223:             model=None,
1224:         )
1225:         return module.build_result_rows(
1226:             prepared_dataset,
1227:             variant_spec=variant_spec,
1228:             training_outcome=training_outcome,
1229:             runtime_seconds=0.5,
1230:             seed=kwargs["seed"],
1231:             profile=profile,
1232:             evaluation_results=evaluation_results,
1233:         )
1234: 
1235:     results = module.run_experiment(
1236:         dataset_ids=(prepared.dataset_id,),
1237:         output_path=output_path,
1238:         device="cpu",
1239:         max_epochs=1,
1240:         seed=7,
1241:         batch_size=2,
1242:         learning_rate=1e-3,
1243:         d_model=8,
1244:         lstm_hidden_dim=8,
1245:         attention_heads=2,
1246:         patch_len=24,
1247:         encoder_layers=1,
1248:         ff_hidden_dim=16,
1249:         residual_channels=8,
1250:         skip_channels=8,
1251:         end_channels=16,
1252:         gcn_depth=2,
1253:         mtgnn_layers=2,
1254:         subgraph_size=3,
1255:         node_embed_dim=4,
1256:         dilation_exponential=2,
1257:         propalpha=0.1,
1258:         hidden_dim=8,
1259:         embed_dim=4,
1260:         num_layers=1,
1261:         cheb_k=2,
1262:         teacher_forcing_ratio=0.0,
1263:         dropout=0.0,
1264:         work_root=work_root,
1265:         dataset_loader=_dataset_loader,
1266:         job_runner=_job_runner,
1267:     )
1268: 
1269:     assert output_path.exists()
1270:     assert module.training_history_output_path(output_path).exists()
1271:     assert results.height == 1036
1272:     assert set(results["model_variant"].unique().to_list()) == {
1273:         module.PERSISTENCE_VARIANT,
1274:         module.TFT_VARIANT,
1275:         module.TIMEXER_VARIANT,
1276:         module.DGCRN_VARIANT,
1277:         module.CHRONOS_VARIANT,
1278:         module.ITRANSFORMER_VARIANT,
1279:         module.MTGNN_VARIANT,
1280:     }
1281:     paths = module._resume_paths_for_output(output_path=output_path, work_root=work_root)
1282:     expected_slot = hashlib.sha256(str(output_path.resolve()).encode("utf-8")).hexdigest()
1283:     assert paths.slot_dir.name == expected_slot
1284:     assert paths.partial_results_path.exists()
1285:     assert paths.checkpoints_dir.exists()
1286:     assert module._job_checkpoint_path(
1287:         paths,
1288:         dataset_id=prepared.dataset_id,
1289:         model_variant=module.TFT_VARIANT,
1290:     ).name == f"{prepared.dataset_id}__{module.TFT_VARIANT}.pt"
```
