from .gradient.fgsm import FGSM
from .gradient.ifgsm import IFGSM
from .gradient.mifgsm import MIFGSM
from .gradient.nifgsm import NIFGSM
from .gradient.pifgsm import PIFGSM
from .gradient.vmifgsm import VMIFGSM
from .gradient.vnifgsm import VNIFGSM
from .gradient.emifgsm import EMIFGSM
from .gradient.ifgssm import IFGSSM
from .gradient.vaifgsm import VAIFGSM
from .gradient.aifgtm import AIFGTM
from .gradient.rap import RAP
from .gradient.gifgsm import GIFGSM
from .gradient.pcifgsm import PCIFGSM
from .gradient.iefgsm import IEFGSM
from .gradient.dta import DTA
from .gradient.gra import GRA
from .gradient.pgn import PGN
from .gradient.smifgrm import SMIFGRM


from .input_transformation.dim import DIM
from .input_transformation.tim import TIM
from .input_transformation.sim import SIM
from .input_transformation.atta import ATTA
from .input_transformation.admix import Admix
from .input_transformation.dem import DEM
from .input_transformation.odi.odi import ODI
from .input_transformation.ssm import SSM
from .input_transformation.aitl import AITL
from .input_transformation.maskblock import MaskBlock
from .input_transformation.sia import SIA
from .input_transformation.stm import STM
from .input_transformation.bsr import BSR
from .input_transformation.decowa import DeCowA
from .input_transformation.l2t import L2T


from .advanced_objective.tap import TAP
from .advanced_objective.ila import ILA
from .advanced_objective.potrip import POTRIP
from .advanced_objective.yaila.yaila import YAILA
from .advanced_objective.logit import LOGIT
from .advanced_objective.fia import FIA
from .advanced_objective.trap import TRAP
from .advanced_objective.naa import NAA
from .advanced_objective.rpa import RPA
from .advanced_objective.taig import TAIG
from .advanced_objective.fmaa import FMAA
from .advanced_objective.cfm import CFM
from .advanced_objective.logit_margin import Logit_Margin
from .advanced_objective.ilpd import ILPD
from .advanced_objective.fft import FFT
from .advanced_objective.ir import IR

from .model_related.ghost import GhostNetwork_MIFGSM
from .model_related.sgm import SGM
from .model_related.iaa import IAA
from .model_related.dsm import DSM
from .model_related.mta import MTA
from .model_related.mup import MUP
from .model_related.bpa import BPA
from .model_related.dhf import DHF_MIFGSM
from .model_related.pna_patchout import PNA_PatchOut
from .model_related.sapr import SAPR
from .model_related.tgr import TGR
from .model_related.setr import SETR
from .model_related.ags import AGS


attack_zoo = {
            # gredient
            'fgsm': FGSM,
            'ifgsm': IFGSM,
            'mifgsm': MIFGSM,
            'nifgsm': NIFGSM,
            'pifgsm': PIFGSM,
            'vmifgsm': VMIFGSM,
            'vnifgsm': VNIFGSM,
            'emifgsm': EMIFGSM,
            'ifgssm': IFGSSM,
            'vaifgsm': VAIFGSM,
            'aifgtm': AIFGTM,
            'rap': RAP,
            'gifgsm': GIFGSM,
            'pcifgsm': PCIFGSM,
            'iefgsm': IEFGSM,
            'dta': DTA,
            'gra': GRA,
            'pgn': PGN,
            'smifgrm': SMIFGRM,

            # input transformation
            'dim': DIM,
            'tim': TIM,
            'sim': SIM,
            'atta': ATTA,
            'admix': Admix,
            'dem': DEM,
            'odi': ODI,
            'ssm': SSM,
            'aitl': AITL,
            'maskblock': MaskBlock,
            'sia': SIA,
            'stm': STM,
            'bsr': BSR,
            'decowa': DeCowA,
            'l2t': L2T,

            # advanced_objective
            'tap': TAP,
            'ila': ILA,
            'potrip': POTRIP,
            'fia': FIA,
            'yaila': YAILA,
            'logit': LOGIT,
            'trap': TRAP,
            'naa': NAA,
            'rpa': RPA,
            'taig': TAIG,
            'fmaa': FMAA,
            'cfm': CFM,
            'logit_margin': Logit_Margin,
            'ilpd': ILPD,
            'fft': FFT,
            'ir': IR,

            # model_related
            'sgm': SGM,
            'iaa': IAA,
            'dsm': DSM,
            'mta': MTA,
            'mup': MUP,
            'bpa': BPA,
            'dhf': DHF_MIFGSM,
            'pna_patchout': PNA_PatchOut,
            'sapr': SAPR,
            'tgr': TGR,
            'ghost': GhostNetwork_MIFGSM,
            'setr': SETR,
            'ags': AGS
        }

__version__ = '1.0.0'
