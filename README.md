# PMA-QM

## Description
This repository aims to provide all the necessary resources and instructions to reproduce the results presented in [LINK TO PAPER2 TO ADD]. It contains the following files and folders:

- PMA-QM.py provides an implementation of how to use the parallel memetic algorithm (PMA) to map a quantum circuit to some hardware. Running it writes a "bench_..._.txt" file that can be used to study the performance of the algorithms.
- pygad_mpi.py provides our modified version of PYGAD (original codes are available at https://github.com/ahmedfgad/GeneticAlgorithmPython) that we modified for our own usage, namely to leverage mpi4py parallelism.
- Benchmarks is a folder containing all the circuits studied in the paper. They were taken from https://www.cda.cit.tum.de/mqtbench/.

## Installation Instructions
The Python codes provided in this repository rely on several Python libraries, namely qiskit.
It can be installed using pip: `pip install qiskit==0.43.3`
Note that all codes have been implemented and tested using qiskit 0.43.3 only. There is no guarantee that older or newer versions would be compatible with this code.
Any other libraries should work fine with their latest versions.

## How to Use
- PMA-QM.py can be executed using `mpirun -n N python ./PMA-QM.py -c CIRCUIT SIZE --pga PGA -s STOP -k CHUNKSIZE;`. The arguments are as follows:
  - CIRCUIT is the type of circuit to be used (e.g., ghzall, ghz, dj, qft).
  - SIZE is a strictly positive integer, the number of qubits of the circuit.
  - PGA is either 1, 2, or 3 and selects which PGA parameters are wanted (see [LINK TO PAPER1 TO ADD]). To reproduce the results presented in [LINK TO PAPER2 TO ADD], PGA=1 should be used.
  - STOP is an integer indicating the stopping criterion of the PGA. Values <= 0 will result in stopping the PGA after a number of generations (30 or 35 here). Values > 0 mean that the PGA may stop earlier if the best solution didn't improve for STOP consecutive generations.
  - CHUNKSIZE is a strictly positive interger indicating the number of jobs submitted at once to the worker processes when they request a task. It choose the granularity of the parallel parts of the code.
    
Three more options are available. At least one of them should be used, otherwise it would run a genetic algorithm:
- -f indicates that the local search operator of the PMA-QM should be run after the final generation have been executed.
- -m indicates that the local search operator of the PMA-QM should be run after the mutation operator at each generations.
- -i indicates that the local search operator of the PMA-QM should be run after the initial generation have been randomly generated.

Extra information about SIZE:
Note that not every circuit is available for every size. Unless one downloads more benchmarks from https://www.cda.cit.tum.de/mqtbench/, the only sizes available are 80 and 120 qubits for both dj and ghz circuits, and 40 qubits for the qft circuit. The ghzall circuit, another way to write the ghz circuit, is the only one available for all sizes, which can be beneficial for small tests to ensure that everything works properly.

Example: `mpirun -n 1280 python ./mpi_Voisin_Custom_d2.py -c dj 120 --pga 1 -s 0 -k 1 -m -f;`
