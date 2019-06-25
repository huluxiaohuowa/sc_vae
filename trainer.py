from data import *
from utils import *
from ops import *
from ipypb import ipb

batch_size = 256

num_epochs = 10
for epoch in ipb(range(num_epochs), decs="epochs"):
    dataloader = Dataloader(batch_size=batch_size)
    num_data = dataloader.num_id_block
    for s, c in ipb(dataloader, desc="step", total=dataloader.num_id_block):
        s_nfeat, s_einfo, s_adj = graph_to_whole_graph(
            s.adjacency_matrix(),
            torch.stack(s.edges(), dim=0),
            s.ndata['feat'],
            s.edata['feat']
        )
        c_nfeat, c_einfo, c_adj = graph_to_whole_graph(
            c.adjacency_matrix(),
            torch.stack(c.edges(), dim=0),
            c.ndata['feat'],
            c.edata['feat']
        )
        

