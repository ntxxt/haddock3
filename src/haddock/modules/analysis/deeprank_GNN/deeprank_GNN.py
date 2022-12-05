import glob 
import sys 
import time
import datetime 
import numpy as np

from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet

import torch
import h5py
import torch.nn.functional as F
import numpy as np
from pssmgen import PSSM

class GNN_score():
    def __init__(self, pdb_source, output_dir):
        self.pdb_source = pdb_source
        self.output_dir = output_dir

    def generate_pssm(self):
        gen = PSSM(work_dir=self.pdb_source)
        gen.configure(blast_exe='/trinity/login/xxu/software/ncbi-blast-2.13.0+/bin/psiblast',
            database='/trinity/login/xxu/data/DBs/nr',
            num_threads=4, evalue=0.0001, comp_based_stats='T',
            max_target_seqs=2000, num_iterations=3, outfmt=7,
            save_each_pssm=True, save_pssm_after_last_round=True)
        gen.get_fasta(pdb_dir='', chain=(self.chain1,self.chain2), out_dir='fasta')
        gen.get_pssm(fasta_dir='fasta', out_dir='pssm_raw', run=True)
        gen.map_pssm(pssm_dir='pssm_raw', pdb_dir='', out_dir='pssm', chain=(self.chain1,self.chain2))
        gen.get_mapped_pdb(pdbpssm_dir='pssm', pdb_dir='', pdbnonmatch_dir='pdb_nonmatch')
    
    def generate_graph(self):
        GraphHDF5(pdb_path=self.pdb_source, pssm_path=os.path.join(self.pdb_source, 'pssm'),
        graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
    
    def test_GNN(self):
        path = os.path.join(self.output_dir, 'prediction.hdf5')
        pretrained_model = '/trinity/login/xxu/software/Deeprank-GNN/paper_pretrained_models/scoring_of_docking_models/fold6_treg_yfnat_b128_e20_lr0.001_4.pt'
        gnn = GINet
        database_test = glob.glob('./*.hdf5')
        model = NeuralNet(database_test, gnn, pretrained_model = pretrained_model)
        model.test(threshold=None,hdf5=path)
        f = h5py.File(path)
        out_class = f['epoch_0000']['test']['outputs'][()]
        probility = F.softmax(torch.FloatTensor(out_class),dim=0)
        pred = probility[:, 0] <= probility[:, 1]



