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
    'ifgssm': ('.gradient.ifgssm', 'IFGSSM'),
    'vaifgsm': ('.gradient.vaifgsm', 'VAIFGSM'),
    'aifgtm': ('.gradient.aifgtm', 'AIFGTM'),
    'rap': ('.gradient.rap', 'RAP'),
    'gifgsm': ('.gradient.gifgsm', 'GIFGSM'),
    'pcifgsm': ('.gradient.pcifgsm', 'PCIFGSM'),
    'iefgsm': ('.gradient.iefgsm', 'IEFGSM'),
    'dta': ('.gradient.dta', 'DTA'),
    'gra': ('.gradient.gra', 'GRA'),
    'pgn': ('.gradient.pgn', 'PGN'),
    'smifgrm': ('.gradient.smifgrm', 'SMIFGRM'),
    'ncs': ('.gradient.ncs', 'NCS'),
    
    # input transformation
    'dim': ('.input_transformation.dim', 'DIM'),
    'tim': ('.input_transformation.tim', 'TIM'),
    'sim': ('.input_transformation.sim', 'SIM'),
    'atta': ('.input_transformation.atta', 'ATTA'),
    'admix': ('.input_transformation.admix', 'Admix'),
    'dem': ('.input_transformation.dem', 'DEM'),
    'odi': ('.input_transformation.odi.odi', 'ODI'),
    'su': ('.input_transformation.su', 'SU'),
    'smm': ('.input_transformation.ssm', 'SSM'),
    'aitl': ('.input_transformation.aitl', 'AITL'),
    'maskblock': ('.input_transformation.maskblock', 'MaskBlock'),
    'sia': ('.input_transformation.sia', 'SIA'),
    'stm': ('.input_transformation.stm', 'STM'),
    'lpm': ('.input_transformation.lpm', 'LPM'),
    'bsr': ('.input_transformation.bsr', 'BSR'),
    'decowa': ('.input_transformation.decowa', 'DeCowA'),
    'l2t': ('.input_transformation.l2t', 'L2T'),
    'idaa': ('.input_transformation.idaa', 'IDAA'),
    
    # advanced_objective
    'tap': ('.advanced_objective.tap', 'TAP'),
    'ata': ('.advanced_objective.ata', 'ATA'),
    'ila': ('.advanced_objective.ila', 'ILA'),
    'potrip': ('.advanced_objective.potrip', 'POTRIP'),
    'yaila': ('.advanced_objective.yaila.yaila', 'YAILA'),
    'logit': ('.advanced_objective.logit', 'LOGIT'),
    'fia': ('.advanced_objective.fia', 'FIA'),
    'trap': ('.advanced_objective.trap', 'TRAP'),
    'naa': ('.advanced_objective.naa', 'NAA'),
    'rpa': ('.advanced_objective.rpa', 'RPA'),
    'taig': ('.advanced_objective.taig', 'TAIG'),
    'fmaa': ('.advanced_objective.fmaa', 'FMAA'),
    'cfm': ('.advanced_objective.cfm', 'CFM'),
    'logit_margin': ('.advanced_objective.logit_margin', 'Logit_Margin'),
    'fuzziness_tuned': ('.advanced_objective.fuzziness_tuned', 'Fuzziness_Tuned'),
    'ilpd': ('.advanced_objective.ilpd', 'ILPD'),
    'fft': ('.advanced_objective.fft', 'FFT'),
    'ir': ('.advanced_objective.ir', 'IR'),
    'danaa': ('.advanced_objective.danaa', 'DANAA'),
    
    # model_related
    'sgm': ('.model_related.sgm', 'SGM'),
    'iaa': ('.model_related.iaa', 'IAA'),
    'dsm': ('.model_related.dsm', 'DSM'),
    'mta': ('.model_related.mta', 'MTA'),
    'mup': ('.model_related.mup', 'MUP'),
    'bpa': ('.model_related.bpa', 'BPA'),
    'dhf': ('.model_related.dhf', 'DHF_MIFGSM'),
    'pna_patchout': ('.model_related.pna_patchout', 'PNA_PatchOut'),
    'sapr': ('.model_related.sapr', 'SAPR'),
    'tgr': ('.model_related.tgr', 'TGR'),
    'ghost': ('.model_related.ghost', 'GhostNetwork_MIFGSM'),
    'setr': ('.model_related.setr', 'SETR'),
    'ags': ('.model_related.ags', 'AGS'),
    
    # ensemble
    'ens': ('.ensemble.ens', 'ENS'),
    'svre': ('.ensemble.svre', 'SVRE'),
    'lgv': ('.ensemble.lgv', 'LGV'),
    'mba': ('.ensemble.mba', 'MBA'),
    'cwa': ('.ensemble.cwa', 'CWA'),
    'adaea': ('.ensemble.adaea', 'AdaEA'),

    # generation
    'ltp': ('.generation.ltp', 'LTP'),
    'ada': ('.generation.ada', 'ADA'),
    'cdtp': ('.generation.cdtp', 'CDTP'),
    'ttp': ('.generation.ttp', 'TTP'),
    'ge_advgan': ('.generation.ge_advgan', 'GE_ADVGAN'),
}


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise Exception('Unspported attack algorithm {}'.format(attack_name))
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    attack_class = getattr(module, class_name)
    return attack_class


__version__ = '1.0.0'
