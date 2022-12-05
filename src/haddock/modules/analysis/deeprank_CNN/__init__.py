"""deeprank CNN scoring."""
from pathlib import Path

from haddock.modules import BaseHaddockModule
from haddock.modules.analysis.deeprank_CNN import CNN_score

RECIPE_PATH = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path(RECIPE_PATH, "defaults.yaml")

class HaddockModule(BaseHaddockModule):
    """HADDOCK3 module for ranking docking poses with deeprank_CNN"""

    name = RECIPE_PATH.name

    def __init__(self, order, path, *ignore, init_params=DEFAULT_CONFIG,
                 **everything):
        super().__init__(order, path, init_params)

    @classmethod
    def confirm_installation(cls):
        return

    def _run(self):
        '''
        input parameters
        '''
        #models_to_select = [p for p in self.previous_io.output if p.file_type == Format.PDB ]
        CNN = CNN_score(pdb_source=pdb_source, self.params['chain1'], self.params['chain2'], output_dir)
        CNN.generate_pssm()
        CNN.create_database()
        CNN.test_CNN()
        result = CNN.analysis_result()
        return result
  