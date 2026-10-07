"""Immutable experiment definitions. No evaluation numbers are supplied here."""
import copy,hashlib,json
from pathlib import Path

DEFAULT={
 'schema':'agcr-experiment-2','backbone':'Wan2.2-TI2V-5B','training_seeds':[17,29,43],
 'sample_seeds':[101,211,307,401,503,601,701,809],'history':6,'future':16,'fps':5,
 'width':720,'height':480,'depth_unit':'mm','state_unit':'m','padding_columns':8,
 'evaluation_starts':[0,25,50,75],'v_min':.1,'area_min':144,'association_threshold':.5,
 'stages':{'observation':40000,'dynamics_geometry':40000,'depth_warmup':10000,'video':80000},
 'optimizer':{'name':'AdamW','new_lr':1e-4,'lora_lr':1e-5,'betas':[.9,.999],
              'weight_decay':.01,'gradient_clip':1.,'effective_batch':16,'warmup':1000,'final_lr_ratio':.1},
 'validation_every':2000,'validation_panel_minimum':64,'validation_panel_seed':17029,'sample_steps':50,'flow_shift':5.,'cfg':1.,
 'backbone_shape':{'blocks':30,'width':3072,'heads':24,'latent_channels':48},
 'modules':{'width':256,'heads':8,'queries':64,'observation_layers':6,'dynamics_layers':6,
            'geometry_layers':4,'scene_tokens':256,'view_scene_tokens':128,'modes':8,
            'lora_rank':16,'lora_alpha':16,'adapter_blocks':list(range(3,31,3))},
 'statistics':{'bootstrap_replicates':2000,'unit':'camera_path_group','within_layout':True,
               'layout_weighting':'equal','ci':.95,'multiple_testing':'benjamini_hochberg'},
 'downstream':{'sizes':[250,1000,4000],'generated_fractions':[0.,.25,.5,.75,1.],
               'total_clips':4000,'quota_policy':'fixed_target_supplement_capture_no_silent_reduction','fixed_window_starts':[0,25,50,75],'acceptance_schema':'agcr-acceptance-2','updates':40000,'batch':16,'detector':'RT-DETR-R18',
               'identity_width':128,'train_seeds':[17,29,43]},
 'readers':{'auxiliary':'generator_auxiliary','annotation':'generated_label_reader','evaluation':'heldout_evaluation_reader'},
 'split_policy':{'id_groups_per_layout':2,'annotation_groups_per_layout':2,'evaluation_reader_groups_per_layout':2},
 'physical_data':'qualitative_only','primary_sample_aggregation':'mean_of_8',
 'best_of_8_selection':'minimum_paired_RGB_LPIPS; same sample index for every modality',
 'flags':{'input_views':'paired','fusion':'weighted','use_visibility_weight':True,
          'use_association_weight':True,'use_reliability_weight':True,'weight_control':'learned',
          'scene_mode':'shared','future_state':'shared','appearance_gate':'visible',
          'depth_input':True,'depth_ordering':'soft','state_path':'ungated',
          'cross_view_exchange':True,'loss_pair':True,'loss_surface':True,'loss_correspondence':True},
}
VARIANTS={
 'full':{},'aerial_only_input':{'input_views':'air'},'ground_only_input':{'input_views':'ground'},
 'concat_fusion':{'fusion':'concat'},'attention_fusion':{'fusion':'attention'},
 'independent_views':{'input_views':'independent','future_state':'independent','cross_view_exchange':False},
 'no_visibility_weight':{'use_visibility_weight':False},'uniform_evidence':{'fusion':'uniform'},
 'no_association_weight':{'use_association_weight':False},'no_reliability_weight':{'use_reliability_weight':False},
 'random_evidence_weight':{'weight_control':'random'},
 'no_shared_scene_mode':{'scene_mode':'independent_targets'},
 'independent_future':{'future_state':'independent'},
 'no_appearance_gate':{'appearance_gate':'full'},
 'rgb_only':{'depth_input':False},
 'no_state_path':{'state_path':'none'},'gated_state_path':{'state_path':'visibility_gated'},
 'hard_depth_ordering':{'depth_ordering':'hard'},'no_depth_ordering':{'depth_ordering':'none'},
 'no_cross_view_exchange':{'cross_view_exchange':False},
 'no_pair_loss':{'loss_pair':False},'no_surface_loss':{'loss_surface':False},
 'no_correspondence_loss':{'loss_correspondence':False},
}
# Earliest affected stage; earlier frozen dependencies may be reused with hashes.
RETRAIN={k:'observation' for k in VARIANTS}
for k in ['no_shared_scene_mode']:RETRAIN[k]='dynamics_geometry'
for k in ['no_appearance_gate','no_state_path','gated_state_path','no_cross_view_exchange',
          'no_pair_loss','no_surface_loss','no_correspondence_loss']:RETRAIN[k]='video'
for k in ['hard_depth_ordering','no_depth_ordering']:RETRAIN[k]='dynamics_geometry'
RETRAIN['independent_future']='inference_intervention'
EXTERNAL={
 'panacea':{'venue':'CVPR 2024','source':'https://github.com/wenyuqing/panacea','modality':'RGB','status':'requires_complete_factory_adaptation'},
 'drivingdiffusion':{'venue':'ECCV 2024','source':'https://github.com/shalfun/DrivingDiffusion','modality':'RGB','status':'requires_complete_factory_adaptation'},
 'magicdrive_v2':{'venue':'ICCV 2025','source':'https://github.com/flymin/MagicDrive-V2','modality':'RGB','status':'requires_complete_factory_adaptation'},
 'syncammaster':{'venue':'ICLR 2025','source':'https://github.com/KlingAIResearch/SynCamMaster','modality':'RGB','status':'requires_complete_factory_adaptation'},
}
EXPERIMENTS={
 'E1':{'claim':'complementary evidence','variants':['full','aerial_only_input','ground_only_input','independent_views'],
       'interventions':['none','drop_air','drop_ground','mismatched_partner'],'primary':['occluded_ADE_m','recovery_accuracy']},
 'E2':{'claim':'visibility association reliability fusion','variants':['full','concat_fusion','attention_fusion','uniform_evidence',
       'no_visibility_weight','no_association_weight','no_reliability_weight','random_evidence_weight'],'interventions':['none','shuffle_partner'],
       'primary':['occluded_ADE_m','association_error']},
 'E3':{'claim':'shared future and receiving-view geometry','variants':['full','no_shared_scene_mode','independent_future',
       'no_appearance_gate','no_state_path','gated_state_path','hard_depth_ordering','no_depth_ordering','no_cross_view_exchange',
       'no_pair_loss','no_surface_loss','no_correspondence_loss'],'interventions':['none'],
       'primary':['paired_center_error_m','false_appearance_rate','recovery_accuracy']},
 'E4':{'claim':'depth robustness','variants':['full','rgb_only'],'interventions':['none','depth_missing_30','depth_missing_50',
       'depth_noise_005','depth_noise_02','depth_clip_15000','static_completion_error'],'primary':['depth_RMSE_mm','occluded_ADE_m']},
 'E5':{'claim':'partner evidence robustness','variants':['full','attention_fusion'],'interventions':['none','delay_1','delay_3','delay_5',
       'time_offset_1','shuffle_partner','mismatched_partner','wrong_association'],'primary':['occluded_ADE_m','recovery_accuracy']},
 'E6':{'claim':'camera robustness','variants':['full'],'interventions':['none','extrinsics_005m_1deg','extrinsics_02m_3deg',
       'physical_tracking_error'],'primary':['paired_center_error_m','correspondence_error_m']},
 'E7':{'claim':'separate state and rendering error','variants':['full'],'interventions':['none','oracle_state','oracle_state_geometry'],
       'primary':['paired_LPIPS','false_appearance_rate','depth_RMSE_mm']},
 'E8':{'claim':'uncertainty calibration','variants':['full','no_shared_scene_mode'],'interventions':['none'],
       'primary':['joint_trajectory_NLL','presence_Brier','state_coverage_95']},
 'E9':{'claim':'generated-data utility','variants':['full','independent_views','attention_fusion']+list(EXTERNAL),
       'regimes':['R','G','R_G','R_repeat','R_plus_G','R_plus_new_render'],
       'primary':['HOTA','IDF1','entry_F1','recovery_accuracy']},
 'E10':{'claim':'generalization','variants':['full','attention_fusion'],'cohorts':['id_episode','ood_layout','ood_class','ood_layout_class','physical_qualitative'],
        'primary':['HOTA','IDF1','occluded_ADE_m']},
}

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sample_seed(train_seed,history_id,sample_index,stream):
    base=DEFAULT['sample_seeds'][sample_index]
    return int(digest([train_seed,history_id,base,stream])[:15],16)%(2**63-1)
def configuration(variant='full',seed=17):
    if variant not in VARIANTS:raise ValueError('Unknown internal variant '+variant)
    c=copy.deepcopy(DEFAULT);c['variant']=variant;c['seed']=seed;c['flags'].update(VARIANTS[variant]);c['retrain_from']=RETRAIN[variant]
    c['protocol_sha256']=digest(c);return c

def write_protocol(destination):
    path=Path(destination);path.mkdir(parents=True,exist_ok=True)
    for name,value in [('protocol',DEFAULT),('matrix',EXPERIMENTS),('external_methods',EXTERNAL)]:
        (path/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    (path/'variants.json').write_text(json.dumps({k:configuration(k) for k in VARIANTS},indent=2)+'\n')
