# Refine CoarsE-grained IndePendent Tasks
Parallel Tip Decomposition (butterfly-based vertex peeling) of Bipartite Graphs

## Introduction
This is an implementation of the RECEIPT algorithm for parallel tip decomposition
on shared-memory multi core servers. RECEIPT partitions the tip number ranges
to create independent subgraphs that can be peeled concurrently. It can peel 
the largest open bipartite datasets in few minutes, orders of magnitude faster
than the baselines.



## Compile
```
make
```


## Prerequisites
[Boost Sort Parallel Library](https://github.com/fjtapia/sort_parallel)


## Run
```
./decompose -i <inputFile> -o <outputFile> -t <# threads> -p <# partitions to create> -s <peelSide>

```
Arguments:

1. **-i** - text file containing input graph in edge list format
2. **-o** - output file where tip numbers will be written (optional)
3. **-t** - numeric value specifying number of threads to use for decomposition (optional, default = 1)
4. **-p** - numeric value specifying number of partitions to create in coarse-grained decomposition (optional, default and recommended = 150)
5. **-s** - enum. Use **-s 0** to peel vertex set $U$ (LHS in input file) and **-s 1** to peel set $V$ (RHS in input file) (optional, default = 0)


## Input
The input file should represent graph in an edge list format where each line is a tuple of two integers as shown below:
```
u v
```
This indicates that there is an edge between vertices `u` and `v`.<br /><br />


The vertices in left column constitute the $U$ set and those in right column constitute set $V$.
Both $U$ and $V$ should be 0 or 1-indexed. <br />

An example input file is given in the *datasets* directory.<br />
RECEIPT has been tested extensively on large bipartite graphs from [KOBLENZ collection](http://konect.cc/).


## Output
The output is written in the specified file in a text format.<br />
Every line in the output file is a tuple of two integers as shown below: 
```
u t
```
where `u` is the vertex id and `t` is it's tip number


## Baselines
[Sequential Tip Decomposition algorithm](http://sariyuce.com/bnd.tar)
[ParButterfly with Julienne's bucketing](https://github.com/jeshi96/parbutterfly)
