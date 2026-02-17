PARAMS=911580 TARGET=900000 IN_RANGE=True
  adaptive_band_predictor=100
  cross_band_attn=17219
  collaborative=254089
  hat_feature_proj=23168
  dat_feature_proj=23168
  nafnet_feature_proj=8320
  multi_res_fusion=355131
  freq_router=77469
  multiscale=17856
  dynamic_selector=57668
  refine_net=77379
CREATE=PASS
SHAPE=[2, 3, 256, 256] OK=True
RANGE=[0.0130,0.9476] OK=True
NAN=False OK=True
FORWARD=PASS
GRADS=84/134 (62.7%) OK=True
NO_GRAD=['expert_weights', 'band_importance', 'collaborative.align_layers.hat.weight', 'collaborative.align_layers.hat.bias', 'collaborative.align_layers.dat.weight']
GRADIENT=PASS
YAML={'fusion_dim': True, 'num_heads': True, 'refine_depth': True, 'refine_channels': True, 'enable_hierarchical': True} ALL_OK=True
CONFIG=PASS
SUMMARY: 4 PASS, 0 FAIL