import importlib

attack_zoo = {
    # gradient
    'fgsm': ('.gradient.fgsm', 'FGSM'),
    'ifgsm': ('.gradient.ifgsm', 'IFGSM'),
    'mifgsm': ('.gradient.mifgsm', 'MIFGSM'),
    'nifgsm': ('.gradient.nifgsm', 'NIFGSM'),
    'pifgsm': ('.gradient.pifgsm', 'PIFGSM'),
    'vmifgsm': ('.gradient.vmifgsm', 'VMIFGSM'),
    'vnifgsm': ('.gradient.vnifgsm', 'VNIFGSM'),
    'emifgsm': ('.gradient.emifgsm', 'EMIFGSM'),
    'aifgtm': ('.gradient.aifgtm', 'AIFGTM'),
    'ifgssm': ('.gradient.ifgssm', 'IFGSSM'),
    'smifgrm': ('.gradient.smifgrm', 'SMIFGRM'),
    'vaifgsm': ('.gradient.vaifgsm', 'VAIFGSM'),
    'rap': ('.gradient.rap', 'RAP'),
    'pcifgsm': ('.gradient.pcifgsm', 'PCIFGSM'),
    'iefgsm': ('.gradient.iefgsm', 'IEFGSM'),
    'gra': ('.gradient.gra', 'GRA'),
    'gnp': ('.gradient.gnp', 'GNP'),
    'mig': ('.gradient.mig', 'MIG'),
    'dta': ('.gradient.dta', 'DTA'),
    'pgn': ('.gradient.pgn', 'PGN'),
    'ncs': ('.gradient.ncs', 'NCS'),
    'anda': ('.gradient.anda', 'ANDA'),
    'gifgsm': ('.gradient.gifgsm', 'GIFGSM'),
    'rgmifgsm': ('.gradient.mifgsm_with_tricks', 'RGMIFGSM'),
    'dual_mifgsm': ('.gradient.mifgsm_with_tricks', 'DualMIFGSM'),
    'ens_mifgsm': ('.gradient.mifgsm_with_tricks', 'Ens_FGSM_MIFGSM'),

    # input transformation
    ## Untargeted
    'dim': ('.input_transformation.dim', 'DIM'),
    'tim': ('.input_transformation.tim', 'TIM'),
    'sim': ('.input_transformation.sim', 'SIM'),
    'dem': ('.input_transformation.dem', 'DEM'),
    'admix': ('.input_transformation.admix', 'Admix'),
    'atta': ('.input_transformation.atta', 'ATTA'),
    'maskblock': ('.input_transformation.maskblock', 'MaskBlock'),
    'ssm': ('.input_transformation.ssm', 'SSM'),
    'aitl': ('.input_transformation.aitl', 'AITL'),
    'pam': ('.input_transformation.pam', 'PAM'),
    'lpm': ('.input_transformation.lpm', 'LPM'),
    'sia': ('.input_transformation.sia', 'SIA'),
    'stm': ('.input_transformation.stm', 'STM'),
    'usmm': ('.input_transformation.usmm', 'USMM'),
    'decowa': ('.input_transformation.decowa', 'DeCowA'),
    'l2t': ('.input_transformation.l2t', 'L2T'),
    'bsr': ('.input_transformation.bsr', 'BSR'),
    ## Targeted
    'odi': ('.input_transformation.odi.odi', 'ODI'),
    'su': ('.input_transformation.su', 'SU'),
    'idaa': ('.input_transformation.idaa', 'IDAA'),
    'ssm_p': ('.input_transformation.ssm_with_tricks', 'SSA_P'),
    'ssm_h': ('.input_transformation.ssm_with_tricks', 'SSM_H'),
    
    # advanced_objective
    ## Untargeted
    'tap': ('.advanced_objective.tap', 'TAP'),
    'ila': ('.advanced_objective.ila', 'ILA'),
    'ata': ('.advanced_objective.ata', 'ATA'),
    'yaila': ('.advanced_objective.yaila.yaila', 'YAILA'),
    'fia': ('.advanced_objective.fia', 'FIA'),
    'ir': ('.advanced_objective.ir', 'IR'),
    'trap': ('.advanced_objective.trap', 'TRAP'),
    'taig': ('.advanced_objective.taig', 'TAIG'),
    'fmaa': ('.advanced_objective.fmaa', 'FMAA'),
    'naa': ('.advanced_objective.naa', 'NAA'),
    'rpa': ('.advanced_objective.rpa', 'RPA'),
    'fuzziness_tuned': ('.advanced_objective.fuzziness_tuned', 'Fuzziness_Tuned'),
    'danaa': ('.advanced_objective.danaa', 'DANAA'),
    'ilpd': ('.advanced_objective.ilpd', 'ILPD'),
    ## Targeted
    'aa': ('.advanced_objective.aa', 'AA'),
    'potrip': ('.advanced_objective.potrip', 'POTRIP'),
    'logit': ('.advanced_objective.logit', 'LOGIT'),
    'logit_margin': ('.advanced_objective.logit_margin', 'Logit_Margin'),
    'cfm': ('.advanced_objective.cfm', 'CFM'),
    'fft': ('.advanced_objective.fft', 'FFT'),
    
    # model_related
    'sgm': ('.model_related.sgm', 'SGM'),
    'pna_patchout': ('.model_related.pna_patchout', 'PNA_PatchOut'),
    'iaa': ('.model_related.iaa', 'IAA'),
    'sapr': ('.model_related.sapr', 'SAPR'),
    'setr': ('.model_related.setr', 'SETR'),
    'mta': ('.model_related.mta', 'MTA'),
    'mup': ('.model_related.mup', 'MUP'),
    'tgr': ('.model_related.tgr', 'TGR'),
    'dsm': ('.model_related.dsm', 'DSM'),
    'dhf': ('.model_related.dhf', 'DHF_MIFGSM'),
    'bpa': ('.model_related.bpa', 'BPA'),
    'ags': ('.model_related.ags', 'AGS'),
    
    # ensemble
    'ens': ('.ensemble.ens', 'ENS'),
    'ghost': ('.model_related.ghost', 'GhostNetwork_MIFGSM'),
    'svre': ('.ensemble.svre', 'SVRE'),
    'lgv': ('.ensemble.lgv', 'LGV'),
    'mba': ('.ensemble.mba', 'MBA'),
    'adaea': ('.ensemble.adaea', 'AdaEA'),
    'cwa': ('.ensemble.cwa', 'CWA'),
    
    # generation
    ## Untargeted
    'cdtp': ('.generation.cdtp', 'CDTP'),
    'ltp': ('.generation.ltp', 'LTP'),
    'ada': ('.generation.ada', 'ADA'),
    'ge_advgan': ('.generation.ge_advgan', 'GE_ADVGAN'),
    ## Targeted
    'ttp': ('.generation.ttp', 'TTP'),
    
}


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise Exception('Unspported attack algorithm {}'.format(attack_name))
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    attack_class = getattr(module, class_name)
    return attack_class


__version__ = '1.0.0'
