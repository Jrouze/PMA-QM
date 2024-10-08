# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import pygad_mpi_d2 as gad
import time
from qiskit import QuantumCircuit,transpile,ClassicalRegister
# from qiskit_ibm_runtime.fake_provider import FakeLimaV2,FakeSingaporeV2,FakeWashingtonV2
from qiskit.providers.fake_provider import FakeLimaV2,FakeSingaporeV2,FakeWashingtonV2
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import argparse
parser = argparse.ArgumentParser()

from mpi4py import MPI
# Get the MPI communicator and process rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

##-------------------------------------------------------
##      Fitness definition
##-------------------------------------------------------
def fitness1(ga_instance,layout, layout_idx):
    l=list(layout)
    QC=transpile(qc,backend,initial_layout=l,optimization_level=3)
    return -QC.depth()

def fitness4_GA(ga_instance,layout, layout_idx):
    return fitness4(layout)

def fitness4(layout):
    init_layout={qr[i]:layout[i] for i in range(len(layout))}
    init_layout=Layout(init_layout)

    pm = generate_preset_pass_manager(3,backend,initial_layout=init_layout)
    pm.layout.remove(1)
    pm.layout.remove(1)

    QC=pm.run(qc)
    return -QC.depth()

##-------------------------------------------------------
##      Circuit selector
##-------------------------------------------------------
def circuit_selector(name,nb_qubit):
    if name=="ghzall":
        qc=QuantumCircuit(nb_qubit)
        qc.h(0)
        for i in range(1,nb_qubit):
            qc.cx(0,i)
        qc.measure_all()
          
        qr=qc.qregs[0]

        name="ghzall"
    else:
        filename=f"{name}_indep_qiskit_{nb_qubit}"
        qasmfile=f"../Benchmarks/MQTBench_all/{filename}.qasm"
        qc=QuantumCircuit().from_qasm_file(qasmfile)
        qr=qc.qregs[0]
    return qc,qr

parser.add_argument("-c", "--circuit", nargs=2, default=["ghzall", "80"], help="Name and number of qubits of the circuit")

##-------------------------------------------------------
##      Reading GA parameters
##------------------------------------------------------- 
parser.add_argument("--pga", type=int, help="PGA set of parameters", choices=[0,1,2,3])
parser.add_argument("-s","--stop_crit", type=int, help="Specifies the stopping criteria. n<=0 means None, n>0 means saturate_n")
parser.add_argument("-i","--enable_initilize", action="store_true", help="Enable the neighbourhood exploration at initilization of GA")
parser.add_argument("-m","--enable_mutation", action="store_true", help="Enable the neighbourhood exploration at each generation after the mutation")
parser.add_argument("-f","--enable_finalize", action="store_true", help="Enable the neighbourhood exploration at finalization of GA")
parser.add_argument("-k","--chunk_size", type=int, default=1, help="Define the chunk_size of the local search operator")

args = parser.parse_args()
num_pga = args.pga

def pga_param_selec(num_pga):
    if num_pga==3:
        num_gen = 35
        elitism = 15
        pop = 20
    elif num_pga==2:
        num_gen = 30
        elitism = 20
        pop = 30
    elif num_pga == 0: ## For testing purpose
        num_gen = 2
        elitism = 5   # 20
        pop = 10   # 40
    else:   ### Includes num_pga==1 or any other values (mistake)
        num_gen = 30
        elitism = 20
        pop = 40
    return num_gen,elitism,pop

num_gen,elitism,pop = pga_param_selec(num_pga)
parent_selec = "random"
cross_type = "two_points"
cross_prob = 0.5
muta_type = "random"
muta_prob = 0.1

if args.stop_crit<=0:
    stop_criteria=None
else:
    stop_criteria=f"saturate_{args.stop_crit}"
    
### Small test values 
# num_gen = 3
# num_mating = 20
# pop = 40
# parent_selec = "random"
# cross_type = "two_points"
# cross_prob = 0.5
# muta_type = "random"
# muta_prob = 0.1
# stop_criteria=None

##-------------------------------------------------------
##      Circuit & Backend selection
##-------------------------------------------------------
### Backend
backend = FakeWashingtonV2()
# backend = FakeSingaporeV2()

### Circuit
name = args.circuit[0]
nb_qubit = int(args.circuit[1])

qc,qr=circuit_selector(name, nb_qubit)

##-------------------------------------------------------
##      Neigborhood exploration
##------------------------------------------------------- 
chunk_size = args.chunk_size

def explo_vois(indiv,backend,indiv_depth=None):
    indiv=list(indiv)
    if indiv_depth is None:
        best_d=fitness4(indiv)
    else:
        best_d=indiv_depth
    best_indiv=indiv
    S=set(range(backend.num_qubits))-set(indiv)
    N=len(indiv)
    for i in range(N):
        a=indiv[i]
        neighbors=set()
        for s in S:
            if backend.coupling_map.distance(a, s)==1:
                neighbors.add(s)
        for s in neighbors:
            indiv_tempo=indiv[:i]+[s]+indiv[i+1:]
            d=fitness4(indiv_tempo)
            if best_d<d:
                best_d=d
                best_indiv=indiv_tempo
    return best_d,best_indiv

##-------------------------------------------------------
##      Neighbourhood of one individual
##-------------------------------------------------------  
def get_indiv_neighbourhood(indiv,backend):
    indiv=list(indiv)
    neighbors={}
    S=set(range(backend.num_qubits))-set(indiv)
    N=len(indiv)
    for i in range(N):
        a=indiv[i]
        for s in S:
            if backend.coupling_map.distance(a, s)==1:
                neighbors[tuple(indiv[:i]+[s]+indiv[i+1:])]=0
    return neighbors

def get_pop_neighbourhood(pop,backend):
    Neighbourhood=[]
    for indiv in pop:
        Neighbourhood.append(get_indiv_neighbourhood(indiv, backend))
    return Neighbourhood

##-------------------------------------------------------
##      Best neighbours extraction
##-------------------------------------------------------
def get_best_neighbours(pop,value,Neighbourhood):
    pop_res = []
    pop_val = []
    for i in range(len(pop)):
        indiv = pop[i]
        val = value[i]
        min_key = max(Neighbourhood[i], key=Neighbourhood[i].get) #max of negative = min of positive
        if Neighbourhood[i][min_key] > val:
            pop_res.append(list(min_key))
            pop_val.append(Neighbourhood[i][min_key])
        else:
            pop_res.append(indiv)
            pop_val.append(val)
    return pop_res,pop_val

##-------------------------------------------------------
##      Core of neighbourhood exploration
##-------------------------------------------------------
def run_exploration(population_to_explore,population_fitness,chunk_size):
    # Get the neighbourhood of all individuals in the population
    Neighbourhood = get_pop_neighbourhood(population_to_explore, backend)

    # Transform Neighbourhood into a single dictionary
    Neighbourhood_dico={}
    for neighbourhood in Neighbourhood:
        Neighbourhood_dico.update(neighbourhood)
        
    Neighbourhood_dico_keys = list(Neighbourhood_dico.keys())
    
    # Define master process
    if rank == 0:
        total_tasks = len(Neighbourhood_dico_keys)
        tasks_sent = 0
        tasks_completed = 0
        termination_not_sent=set(range(1,size))
        
        # Loop until all tasks are completed
        while tasks_completed < total_tasks:
            # Receive message from any worker
            status = MPI.Status()
            result_size = np.empty(1, dtype=np.int32)
            comm.Recv([result_size, MPI.INT], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            
            if status.tag == 0: # It's a task request
                # Assign new task to the worker
                if tasks_sent < total_tasks:
                    tasks_to_send = np.arange(tasks_sent, min(tasks_sent + chunk_size, total_tasks), dtype=np.int32)
                    comm.Send([np.array([len(tasks_to_send)], dtype=np.int32), 1, MPI.INT], dest=status.source, tag=1)
                    comm.Send([tasks_to_send, len(tasks_to_send), MPI.INT], dest=status.source, tag=1)
                    tasks_sent += len(tasks_to_send)
                # else:
                #     # No more tasks, send termination signal
                #     comm.send(None, dest=status.source, tag=2)
                #     if status.source in termination_not_sent:
                #         termination_not_sent.remove(status.source) 
           
            else:  # It's a completed task
                result_size_int = result_size[0]
                results = np.empty(result_size_int, dtype=np.int32)
                comm.Recv([results, MPI.INT], source=status.source, tag=status.tag)

                for task_id, result in enumerate(results,start=status.tag-1):
                    Neighbourhood_dico[Neighbourhood_dico_keys[task_id]] = result
                    tasks_completed += 1
            
        # Terminate workers
        for i in termination_not_sent:
            comm.Send([np.array([0]), 1, MPI.INT], dest=i, tag=2)
        
        # Update each neighbourhood in Neighbourhood
        for neighbourhood in Neighbourhood:
            for key in neighbourhood:
                neighbourhood[key]=Neighbourhood_dico[key]
            
        # Extract the best neighbour of each neighbourhood        
        new_pop,new_fitness=get_best_neighbours(population_to_explore,population_fitness,Neighbourhood)
    
    # Define worker processes
    else:
        while True:
            # Request task from master
            comm.Send([np.array([0], dtype=np.int32), 1, MPI.INT], dest=0, tag=0)
            
            # Receive task or termination signal from master
            task_chunk_size = np.array([0], dtype=np.int32)
            comm.Recv([task_chunk_size, MPI.INT], source=0, tag=MPI.ANY_TAG)   
            if task_chunk_size[0] == 0:
                break  # No more tasks, terminate
            
            # Loop did not break. There are some task, receive them
            task_chunk = np.empty(task_chunk_size[0], dtype=np.int32)
            comm.Recv([task_chunk, MPI.INT], source=0, tag=MPI.ANY_TAG)
            
            # Perform computation
            results = np.array([fitness4(Neighbourhood_dico_keys[task]) for task in task_chunk], dtype=np.int32)
                    
            # Send result back to master
            comm.Send([np.array([len(results)], dtype=np.int32), 1, MPI.INT], dest=0, tag=task_chunk[0] + 1)
            comm.Send([results, len(results), MPI.INT], dest=0, tag=task_chunk[0] + 1)
    
    # comm.barrier()
    if rank == 0:
        return new_pop, new_fitness
    else:
        return None,None

##-------------------------------------------------------
##      on_start exploration
##------------------------------------------------------- 
def on_start_exploration(ga_instance):
    # Send the population to explore to all process
    population_to_explore = None
    if rank == 0:
        population_to_explore = ga_instance.population
    population_to_explore = comm.bcast(population_to_explore, root=0)
    
    # Run the neighbourhood exploration
    new_pop,new_fitness = run_exploration(population_to_explore, 
                                          ga_instance.last_generation_fitness,chunk_size)
    # comm.barrier()
    # Store the results
    if rank == 0:
        ga_instance.population = np.array(new_pop)
        ga_instance.last_generation_fitness = np.array(new_fitness)

##-------------------------------------------------------
##      on_mutation exploration
##------------------------------------------------------- 
##### Neighbourhood evaluation
def on_mutation_exploration(ga_instance, last_gen_offspring_mut):    
    comm.barrier()
    # Get the neighbourhood of all individuals in the population
    last_gen_offspring_mut = comm.bcast(last_gen_offspring_mut, root=0)
    
    comm.barrier()
    # Run the neighbourhood 
    new_pop,new_fitness = run_exploration(last_gen_offspring_mut, 
                                          ga_instance.last_generation_fitness,chunk_size)        
    comm.barrier()
    # Store the results
    if rank == 0:
        # ga_instance.last_generation_offspring_mutation = np.array(new_pop)
        ga_instance.last_generation_mutation_fitness = new_fitness
        return np.array(new_pop)
        
##-------------------------------------------------------
##      on_stop exploration
##-------------------------------------------------------
def on_stop_exploration(ga_instance, last_pop_fitness):
    comm.barrier()
    # Send the population to explore to all process
    population_to_explore = None
    if rank == 0:
        population_to_explore = ga_instance.population
    population_to_explore = comm.bcast(population_to_explore, root=0)
    
    comm.barrier()
    # Run the neighbourhood exploration
    new_pop,new_fitness = run_exploration(population_to_explore, 
                                          last_pop_fitness,chunk_size)
    
    # Run the neighbourhood exploration again (ILS)
    for i in range(9):
        population_to_explore = comm.bcast(new_pop, root=0)
        last_pop_fitness = comm.bcast(new_fitness, root=0)
        new_pop,new_fitness = run_exploration(population_to_explore, 
                                              last_pop_fitness,chunk_size)
    comm.barrier()
    # Store the results
    if rank == 0:
        ga_instance.population = np.array(new_pop)
        ga_instance.last_generation_fitness = np.array(new_fitness)

##-------------------------------------------------------
##      Create a GA instance
##------------------------------------------------------- 
on_start = None
on_mutation = None
on_stop = None

resname =""
if args.enable_initilize:
    on_start = on_start_exploration
    resname += "INIT_"    
if args.enable_mutation:
    on_mutation = on_mutation_exploration
    resname += "MUT_"
if args.enable_finalize:
    on_stop = on_stop_exploration
    resname += "STOP_"
if resname:
    resname = resname[:-1]
else:
    resname = "NONE"
if resname == "MUT":
    resname = "MUTATION"

ga_instance = gad.GA(num_generations = num_gen,
                        num_parents_mating = pop//2,
                        keep_elitism=elitism,
                        stop_criteria=stop_criteria,
                        fitness_func = fitness4_GA,
                        sol_per_pop = pop,
                        num_genes = nb_qubit,
                        gene_type = int,
                        gene_space = range(0,backend.num_qubits),
                        parent_selection_type = parent_selec,
                        crossover_type = cross_type,
                        crossover_probability = cross_prob,
                        mutation_type = muta_type,
                        mutation_probability = muta_prob,
                        mutation_by_replacement = True,
                        allow_duplicate_genes = False,
                        parallel_processing = ['process',32], ## Leave it like that for now. Anything but None should do it one day
                        on_start=on_start,
                        on_mutation=on_mutation,
                        on_stop=on_stop
                        # on_fitness=on_fitness,
                        # on_crossover=on_crossover,
                        # on_generation=on_generation,                        
                        )

##-------------------------------------------------------
##      Time measurement
##------------------------------------------------------- 
### Measuring GA time
comm.barrier()
start = time.time()
ga_instance.run()
comm.barrier()
end = time.time()
    
comm.barrier()

### Extracting information and transpile the chosen solution
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
# QC_ga=transpile(qc,backend,initial_layout=list(solution),optimization_level=3)
# ga_instance.plot_fitness()

if rank == 0:
    print(ga_instance.best_solutions_fitness)
##-------------------------------------------------------
##      Writing some result
##-------------------------------------------------------
if rank == 0:
    stop_crit_name=""
    if stop_criteria:
        stop_crit_name=stop_criteria
    with open(f"bench_MPIvoisin_{resname}_{name}_{nb_qubit}_pga{num_pga}_{stop_crit_name}_STOP10.txt","a") as fout:
        fout.write(f"{name}_{nb_qubit} {-solution_fitness} {end - start} {size} {ga_instance.generations_completed} {resname}\n")
               
    print(f"{name}_{nb_qubit} {-solution_fitness} {end - start} {ga_instance.generations_completed}\n")

# Finalize MPI
MPI.Finalize()
