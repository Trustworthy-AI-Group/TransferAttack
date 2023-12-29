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
from .gradient.pcifgsm import PCIFGSM
from .gradient.dta import DTA
from .gradient.gra import GRA
from .gradient.pgn import PGN


from .input_transformation.dim import DIM
from .input_transformation.tim import TIM
from .input_transformation.sim import SIM
from .input_transformation.admix import Admix
from .input_transformation.dem import DEM
from .input_transformation.ssm import SSM
from .input_transformation.sia import SIA
from .input_transformation.stm import STM
from .input_transformation.bsr import BSR


from .advanced_objective.tap import TAP
from .advanced_objective.ila import ILA
from .advanced_objective.yaila.yaila import YAILA
from .advanced_objective.fia import FIA
from .advanced_objective.trap import TRAP
from .advanced_objective.naa import NAA
from .advanced_objective.rpa import RPA
from .advanced_objective.taig import TAIG
from .advanced_objective.fmaa import FMAA
from .advanced_objective.ilpd import ILPD


from .architecture.sgm import SGM
from .architecture.dsm import DSM
from .architecture.mta import MTA
from .architecture.mup import MUP
from .architecture.bpa import BPA
from .architecture.pna_patchout import PNA_PatchOut
from .architecture.sapr import SAPR
from .architecture.tgr import TGR




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
            'pcifgsm': PCIFGSM,
            'dta': DTA,
            'gra': GRA,
            'pgn': PGN,

            # input transformation
            'dim': DIM,
            'tim': TIM,
            'sim': SIM,
            'admix': Admix,
            'dem': DEM,
            'ssm': SSM,
            'sia': SIA,
            'stm': STM,
            'bsr': BSR,

            # advanced_objective
            
            'tap': TAP,
            'ila': ILA,
            'fia': FIA,
            'yaila': YAILA,
            'trap': TRAP,
            'naa': NAA,
            'rpa': RPA,
            'taig': TAIG,
            'fmaa': FMAA,
            'ilpd':ILPD,

            # architecture
            'sgm': SGM,
            'dsm': DSM,
            'mta': MTA,
            'mup': MUP,
            'bpa': BPA,
            'pna_patchout': PNA_PatchOut,
            'sapr': SAPR,
            'tgr': TGR,
        }

__version__ = '1.0.0'
