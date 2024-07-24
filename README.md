## General overview

For introduction to the problem refer to these [notes](https://github.com/lrigolli/gnn_gametheory/blob/main/introduction.pdf)  
## Code

### Project set up

1) Clone git repo 

2) Create conda environment 
 > conda env create gnn_env python=3.11
 
3) Activate conda environment
 > conda activate gnn_env
 
4) Install requirements
 > pip install -r requirements.txt  
 
5) Add conda environment kernel to jupyter by activating conda environment and running
 > python -m ipykernel install --user --name gnn_env --display-name 'gnn_env'
 
6) Run [notebook](https://github.com/lrigolli/gnn_gametheory/blob/main/notebooks/gnn_egt.ipynb) 
 
### Scope
Let us consider a graph *G* with:  
- *n >= 1* nodes, each of them representing an infinite population,  
- *n(n-1)/2* weighted edges representing connections (likelihood of interaction) between populations,  
- *m >= 1* strategies being the allowed behaviours of individuals in population,   
- *n* payoff matrices (one for each node) prescribing the variation of fitness when two individuals meet.  

Such a graph describes the fitness of different strategies in populations that are spatially linked and in which environment determines the payoff of a strategy.


If *n=1* the graph consists of a single population, which is the original setting considered by Maynard Smith in his seminal work on ESS. 
Even in this simple setting, up to the best of author's knowledge, there are not many tools for ESS detection and the existing ones like [egttolls](https://pypi.org/project/egttools/) are more focused on exploring dynamics in low dimensions, but do not allow to find ESS when many strategies are possible. 

*Example: perturbed rock-scissors-paper game in single node. ESS = (1/3 R, 1/3 S, 1/3 P)*   
![](https://github.com/lrigolli/gnn_gametheory/blob/main/latex_docs/figures/rsp_perturbed.png?raw=true)

If *n>1* the situation becomes more complex and this tool is the first attempt to study it, with the help of Graph Neural Networks.
  
*Example: hawk-dove game in two nodes with constant adjacency matrix (v1=0.2, v2=0.6, c1=c2=1). ESS is (v1+v2)/(c1+c2) hawks.*  
![](https://github.com/lrigolli/gnn_gametheory/blob/main/latex_docs/figures/hawk_dove_two_nodes_ex.png?raw=true)


### Get started
Once the project is set up, you should choose the graph to be analyzed by specifying:  
- payoff matrices (one for each node)  
- adjacency matrix (each row must sum to 1)  
You can either do it manually or by picking some predefined examples.  
If you run the cells you will:  
1) train a NN model to detect one critical points of Replicator Equations,  
2) check if the detected point is also ESS,  
3) visualize the graph structure with population frequency corresponding to detected point


### Limitations/WIP

Current implementation does not provide an exhaustuive list of ESS and it is not guaranteed it will find one, even if that exists.
Indeed  
- the NN output may fail to converge to a critical point,
- the detected critical point may be a Nash equilibrium which is not ESS  
