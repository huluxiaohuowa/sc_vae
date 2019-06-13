from os import path
import json
from multiprocessing import cpu_count

import dgl

from mol_spec import *

__all__ = [
]

ms = MoleculeSpec.get_default()
