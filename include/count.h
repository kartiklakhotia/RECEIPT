#include "graph.h"
#include "csr.h"

//functions in this file assume that adjacency lists are sorted
//and the graph is reordered as per degree



/*****************************************************************************
Estimate complexities for counting and peeeling vertices
Inputs:
    1. G -> graph object
Outputs:
    1. countWork -> per vertex work required to count
    2. peelWork -> per vertex work required to peel
******************************************************************************/
intB estimate_total_workload (Graph &G, std::vector<intE> &countWork, std::vector<intE> &peelWork)
{
    countWork.resize(G.numT);
    peelWork.resize(G.numT);
    intB totalCountComplexity = 0;
    intB totalPeelComplexity = 0;
    #pragma omp parallel 
    {
        #pragma omp for schedule(dynamic, 5)
        for (intV i=0; i<G.numT; i++) 
        {
            countWork[i] = 0;
            peelWork[i] = 0;
            if (G.is_deleted(i))
            {
                peelWork[i]++;
                countWork[i]++;
            }
            else
            {
                intV deg;
                std::vector<intV> &neighList = G.get_neigh(i, deg);
                peelWork[i] += deg;
                countWork[i] += deg;
                for (intV j=0; j<deg; j++)
                {
                    intV neigh = neighList[j];
                    if (G.is_deleted(neigh)) continue;
                    intV neighDeg;
                    std::vector<intV> &neighOfNeighList = G.get_neigh(neigh, neighDeg); 
                    intV loc = std::lower_bound(neighOfNeighList.begin(), neighOfNeighList.begin() + neighDeg, std::min(i, neigh)) - neighOfNeighList.begin();
                    countWork[i] += loc;
                    peelWork[i] += neighDeg; 
                }
            }
            //printf("coutinng complexity for %u\n", i);
        } 
        #pragma omp barrier
        #pragma omp for reduction (+:totalCountComplexity)
        for (intV i=0; i<G.numT; i++)
            totalCountComplexity += countWork[i]; 
    }
//    printf("cc=%lld, pc=%lld\n", totalCountComplexity, totalPeelComplexity);
    return totalCountComplexity;
}


/*****************************************************************************
Estimate complexities for counting and peeeling vertices of a partition
Don't touch peelWork of deleted vertices
Inputs:
    1. G -> graph object
Outputs:
    1. countWork -> per vertex work required to count
    2. peelWork -> per vertex work required to peel
******************************************************************************/
intB estimate_total_workload_part (Graph &G, std::vector<intE> &countWork, std::vector<intE> &peelWork)
{
    intB totalCountComplexity = 0;
    intB totalPeelComplexity = 0;
    #pragma omp parallel 
    {
        #pragma omp for schedule(dynamic, 5)
        for (intV i=0; i<G.numT; i++) 
        {
            if (!G.is_deleted(i))
            {
                countWork[i] = 0;
                peelWork[i] = 0;
                intV deg;
                std::vector<intV> &neighList = G.get_neigh(i, deg);
                peelWork[i] += deg;
                countWork[i] += deg;
                for (intV j=0; j<deg; j++)
                {
                    intV neigh = neighList[j];
                    if (G.is_deleted(neigh)) continue;
                    intV neighDeg;
                    std::vector<intV> &neighOfNeighList = G.get_neigh(neigh, neighDeg); 
                    intV loc = std::lower_bound(neighOfNeighList.begin(), neighOfNeighList.begin() + neighDeg, std::min(i, neigh)) - neighOfNeighList.begin();
                    countWork[i] += loc;
                    peelWork[i] += neighDeg; 
                }
            }
            //printf("coutinng complexity for %u\n", i);
        } 
        #pragma omp barrier
        #pragma omp for reduction (+:totalCountComplexity)
        for (intV i=0; i<G.numT; i++)
        {
            if (!G.is_deleted(i))
                totalCountComplexity += countWork[i]; 
        }
    }
//    printf("cc=%lld, pc=%lld\n", totalCountComplexity, totalPeelComplexity);
    return totalCountComplexity;
}

/*****************************************************************************
Estimate complexities for counting and peeeling vertices from CSR MATRIX
OVERLOADED FUNCTION
SEQUENTIAL
Inputs:
    1. G -> CSR matrix
Outputs:
    1. countWork -> per vertex work required to count
    2. peelWork -> per vertex work required to peel
******************************************************************************/
intB estimate_total_workload_part (myCSR &G, std::vector<intE> &countWork, std::vector<intE> &peelWork)
{
    if (countWork.size() != G.numT)
        countWork.resize(G.numT);
    if (peelWork.size() != G.numT)
        peelWork.resize(G.numT);
    intB totalCountComplexity = 0;
    intB totalPeelComplexity = 0;
    for (intV i=0; i<G.numT; i++) 
    {
        countWork[i] = 0;
        peelWork[i] = 0;
        if (G.deg[i]==0)
        {
            peelWork[i]++;
            countWork[i]++;
        }
        else
        {
            peelWork[i] += G.deg[i];
            countWork[i] += G.deg[i];
            for (intV j=G.VI[i]; j<G.VI[i]+G.deg[i]; j++)
            {
                intV neigh = G.EI[j];
                intE loc = std::lower_bound(G.EI.begin() + G.VI[neigh], G.EI.begin() + G.VI[neigh] + G.deg[neigh], std::min(i, neigh)) - G.EI.begin() - G.VI[neigh];
                assert(loc < G.numT);
                countWork[i] += loc;
                peelWork[i] += G.deg[neigh]; 
            }
        }
        //printf("coutinng complexity for %u\n", i);
    } 
    for (intV i=0; i<G.numT; i++)
        totalCountComplexity += countWork[i]; 
    return totalCountComplexity;
}



/*****************************************************************************
Count butterflies per vertex
Inputs:
    1. G -> graph object (vertices reordered by degree)
Outputs:
    1. opCnt -> per vertex butterfly counts
Arguments:
    1. wedgeCnt -> 2D array used to count wedges during vertex counting/peeling
******************************************************************************/
void count_per_vertex (Graph &G, std::vector<intB> &opCnt, std::vector<std::vector<intV>> &wedgeCnt)
{
    if (opCnt.size() != G.numT)
        opCnt.resize(G.numT);
    #pragma omp parallel for
    for (intV i=0; i<G.numT; i++)
        opCnt[i] = 0;

    intV pvtThresh = std::min(G.numT, (intV)1000000); //thread private count array for top vertices, reduce coherency traffic
    #pragma omp parallel 
    {
        size_t tid = omp_get_thread_num();
        std::vector<intB> numB (pvtThresh, 0); 
        std::vector<intV> &numW = wedgeCnt[tid];
        std::vector<intV> hop2Neighs (4096);
        hop2Neighs.clear();
        #pragma omp for schedule(dynamic, 5)
        for (intV i=0; i<G.numT; i++)
        {
            //printf("starting %d\n", i);
            if(!G.is_deleted(i)) 
            {
                //count wedges
                intV deg = 0;
                std::vector<intV>& neighList = G.get_neigh(i, deg);
                for (intV j=0; j<deg; j++)
                {
                    intV neigh = neighList[j];
                    if(G.is_deleted(neigh)) continue;
                    intV neighDeg = 0;
                    std::vector<intV>& neighOfNeighList = G.get_neigh(neigh, neighDeg);
                    for (intV k=0; k<neighDeg; k++)
                    {
                        intV neighOfNeigh = neighOfNeighList[k];
                        if(G.is_deleted(neighOfNeigh)) continue;
                        if ((neighOfNeigh>=neigh)||(neighOfNeigh>=i))
                            break;
                        if (numW[neighOfNeigh]==0)
                            hop2Neighs.push_back(neighOfNeigh);
                        numW[neighOfNeigh]++;
                    }
                }


                //add to the butterfly count of self and 2-hop neighbors
                for (auto x : hop2Neighs)
                {
                    intB sameHalfButterflies = choose2<intB, intV>(numW[x]);
                    if (i < pvtThresh)
                        numB[i] += sameHalfButterflies;
                    else
                        __sync_fetch_and_add(&opCnt[i], sameHalfButterflies);
                    if (x < pvtThresh)
                        numB[x] += sameHalfButterflies;
                    else
                        __sync_fetch_and_add(&opCnt[x], sameHalfButterflies);
                }



                //add to butterflies of neighboring vertices (opposite half)
                for (intV j=0; j<deg; j++)
                {
                    intV neigh = neighList[j];
                    if(G.is_deleted(neigh)) continue;
                    intV neighDeg = 0;
                    std::vector<intV>& neighOfNeighList = G.get_neigh(neigh, neighDeg);
                    for (intV k=0; k<neighDeg; k++)
                    {
                        intV neighOfNeigh = neighOfNeighList[k];
                        if(G.is_deleted(neighOfNeigh)) continue;
                        if ((neighOfNeigh>=neigh)||(neighOfNeigh>=i))
                            break;
                        intB oppHalfButterflies = numW[neighOfNeigh] - 1;
                        if (neigh < pvtThresh)
                            numB[neigh] += oppHalfButterflies;
                        else
                            __sync_fetch_and_add(&opCnt[neigh], oppHalfButterflies);
                    }
                }

                for (auto x : hop2Neighs)
                    numW[x] = 0;
                hop2Neighs.clear();
            }

        }
    
        for (intV i=0; i<pvtThresh; i++)
            __sync_fetch_and_add(&opCnt[i], numB[i]);
    }
}

/*****************************************************************************
OVERLOADED FUNCTION DEFINITION
Count butterflies per vertex during coarse-grained decomposition decomposition
Focus only on one side
Inputs:
    1. G -> graph object (vertices reordered by degree)
    2. vertices -> list of vertices to count from (from U side)
Outputs:
    1. opCnt -> per vertex butterfly counts
Arguments:
    1. wedgeCnt -> 2D array used to count wedges during vertex counting/peeling
******************************************************************************/
void count_per_vertex (Graph &G, std::vector<intV> &vertices, std::vector<intB> &opCnt, std::vector<std::vector<intV>> &wedgeCnt)
{
    if (opCnt.size() != G.numT)
        opCnt.resize(G.numT);
    #pragma omp parallel for
    for (intV i=0; i<vertices.size(); i++)
        opCnt[vertices[i]] = 0;

    intV pvtThresh = std::min(G.numT, (intV)100000); //thread private count array for top vertices, reduce coherency traffic
    intV uBS = ((vertices.size()-1)/NUM_THREADS + 1); uBS = (uBS > 5) ? 5 : uBS;
    intV vBS = 5;
    #pragma omp parallel 
    {
        size_t tid = omp_get_thread_num();
        std::vector<intB> numB (pvtThresh, 0); 
        std::vector<intV> &numW = wedgeCnt[tid];
        std::vector<intV> hop2Neighs (4096);
        hop2Neighs.clear();
        #pragma omp for schedule(dynamic, uBS)
        for (intV i=0; i<vertices.size(); i++)
        {
            int u = vertices[i]; 
            if(!G.is_deleted(u)) 
            {
                //count wedges
                intV deg = 0;
                std::vector<intV>& neighList = G.get_neigh(u, deg);
                for (intV j=0; j<deg; j++)
                {
                    intV neigh = neighList[j];
                    intV neighDeg = 0;
                    std::vector<intV>& neighOfNeighList = G.get_neigh(neigh, neighDeg);
                    for (intV k=0; k<neighDeg; k++)
                    {
                        intV neighOfNeigh = neighOfNeighList[k];
                        if(G.is_deleted(neighOfNeigh)) continue;
                        if ((neighOfNeigh>=neigh)||(neighOfNeigh>=u))
                            break;
                        if (numW[neighOfNeigh]==0)
                            hop2Neighs.push_back(neighOfNeigh);
                        numW[neighOfNeigh]++;
                    }
                }


                //add to the butterfly count of self and 2-hop neighbors
                for (auto x : hop2Neighs)
                {
                    intB sameHalfButterflies = choose2<intB, intV>(numW[x]);
                    if (u < pvtThresh)
                        numB[u] += sameHalfButterflies;
                    else
                        __sync_fetch_and_add(&opCnt[u], sameHalfButterflies);
                    if (x < pvtThresh)
                        numB[x] += sameHalfButterflies;
                    else
                        __sync_fetch_and_add(&opCnt[x], sameHalfButterflies);
                    numW[x] = 0;
                }

                hop2Neighs.clear();

            }

        }

        #pragma omp for schedule(dynamic, vBS)
        for (intV i=0; i<G.numV; i++)
        {
            int v = G.vLabels[i]; 
            //count wedges
            intV deg = 0;
            std::vector<intV>& neighList = G.get_neigh(v, deg);
            for (intV j=0; j<deg; j++)
            {
                intV neigh = neighList[j];
                if(G.is_deleted(neigh)) continue;
                intV neighDeg = 0;
                std::vector<intV>& neighOfNeighList = G.get_neigh(neigh, neighDeg);
                for (intV k=0; k<neighDeg; k++)
                {
                    intV neighOfNeigh = neighOfNeighList[k];
                    if ((neighOfNeigh>=neigh)||(neighOfNeigh>=v))
                        break;
                    if (numW[neighOfNeigh]==0)
                        hop2Neighs.push_back(neighOfNeigh);
                    numW[neighOfNeigh]++;
                }
            }


            //add to butterflies of neighboring vertices (opposite half)
            for (intV j=0; j<deg; j++)
            {
                intV neigh = neighList[j];
                if(G.is_deleted(neigh)) continue;
                intV neighDeg = 0;
                std::vector<intV>& neighOfNeighList = G.get_neigh(neigh, neighDeg);
                for (intV k=0; k<neighDeg; k++)
                {
                    intV neighOfNeigh = neighOfNeighList[k];
                    if ((neighOfNeigh>=neigh)||(neighOfNeigh>=v))
                        break;
                    intB oppHalfButterflies = numW[neighOfNeigh] - 1;
                    if (neigh < pvtThresh)
                        numB[neigh] += oppHalfButterflies;
                    else
                        __sync_fetch_and_add(&opCnt[neigh], oppHalfButterflies);
                }
            }

            for (auto x : hop2Neighs)
                numW[x] = 0;
            hop2Neighs.clear();



        }
    
        if (vertices.size() < (pvtThresh<<2))
        {
            for (intV i=0; i<vertices.size(); i++)
            {
                if (vertices[i] >= pvtThresh) continue;
                if (numB[vertices[i]] > 0)
                    __sync_fetch_and_add(&opCnt[vertices[i]], numB[vertices[i]]);
            }
        }
        else
        {
            for (intV i=0; i<pvtThresh; i++)
            {
                if (numB[i] > 0)
                    __sync_fetch_and_add(&opCnt[i], numB[i]);
            }
        }
    }
}



/*****************************************************************************
OVERLOADED FUNCTION DEFINITION
SEQUENTIAL
Count butterflies per vertex, using a CSR MATRIX, used for fine-grained decomposition
Inputs:
    1. partG -> csr matrix (only edges incident on partition vertices)
    2. G -> graph object
    3. partVertices -> list of vertices belonging to this partition
Outputs:
    1. opCnt -> per vertex butterfly counts
Arguments:
    1. wedgeCnt -> 2D array used to count wedges during vertex counting/peeling
******************************************************************************/
void count_per_vertex (myCSR &partG, Graph &G, std::vector<intV> &partVertices, std::vector<intB> &opCnt, std::vector<intV> &wedgeCnt)
{
    if (opCnt.size() != partG.numT)
        opCnt.resize(partG.numT);
    for (auto x : partVertices)
        opCnt[x] = 0;

    std::vector<intV> &numW = wedgeCnt;
    std::vector<intV> hop2Neighs;
    hop2Neighs.reserve(4096);
    for (auto u : partVertices)
    {
        if (G.is_deleted(u)) continue;
        for (intV j=partG.VI[u]; j<partG.VI[u] + partG.deg[u]; j++)
        {
            intV neigh = partG.EI[j];
            for (intV k=partG.VI[neigh]; k<partG.VI[neigh]+partG.deg[neigh]; k++)
            {
                intV neighOfNeigh = partG.EI[k];
                if (G.is_deleted(neighOfNeigh)) continue;
                if ((neighOfNeigh>=neigh) || (neighOfNeigh>=u)) 
                    break;
                if (numW[neighOfNeigh]==0)
                    hop2Neighs.push_back(neighOfNeigh);
                numW[neighOfNeigh]++;
            }
        }
        for (auto x : hop2Neighs)
        {
            intB sameHalfButterflies = choose2<intB, intV>(numW[x]);
            opCnt[u] += sameHalfButterflies;
            opCnt[x] += sameHalfButterflies;
            numW[x] = 0;
        }
        hop2Neighs.clear();
    }

    for (auto v : G.vLabels)
    {
        for (intV j=partG.VI[v]; j<partG.VI[v]+partG.deg[v]; j++)
        {
            intV neigh = partG.EI[j];
            if (G.is_deleted(neigh)) continue;
            for (intV k=partG.VI[neigh]; k<partG.VI[neigh]+partG.deg[neigh]; k++)
            {
                intV neighOfNeigh = partG.EI[k];
                if((neighOfNeigh>=neigh) || (neighOfNeigh>=v))
                    break;
                if (numW[neighOfNeigh]==0)
                    hop2Neighs.push_back(neighOfNeigh);
                numW[neighOfNeigh]++;
            }
        }
        for (intV j=partG.VI[v]; j<partG.VI[v]+partG.deg[v]; j++)
        {
            intV neigh = partG.EI[j];
            if (G.is_deleted(neigh)) continue;
            for (intV k=partG.VI[neigh]; k<partG.VI[neigh]+partG.deg[neigh]; k++)
            {
                intV neighOfNeigh = partG.EI[k];
                if((neighOfNeigh>=neigh) || (neighOfNeigh>=v))
                    break;
                int oppHalfButterflies = numW[neighOfNeigh] - 1;
                opCnt[neigh] += oppHalfButterflies;
            }
        }
        for (auto x : hop2Neighs)
            numW[x] = 0;
        hop2Neighs.clear();
    }
}





//hash function to hash std::pair for unordered map
struct hash_pair { 
    template <class T1, class T2> 
    size_t operator()(const std::pair<T1, T2>& p) const
    { 
        auto hash1 = std::hash<T1>{}(p.first); 
        auto hash2 = std::hash<T2>{}(p.second); 
        return hash1 ^ hash2; 
    } 
};

//index edges for per-edge computation
//index generation is serial for now
void label_edges (Graph &G)
{
    std::unordered_map<std::pair<intV, intV>, intE, hash_pair> labels;
    labels.reserve(G.numE);
    intE currLabel = 0;
    intV numT = G.numT;
    for (intV i=0; i<numT; i++)
    {
        intV deg = 0;
        std::vector<intV>& neighList = G.get_neigh(i, deg);
        for (intV j=0; j<deg; j++)
        {
            intV neigh = neighList[j];
            if (neigh < i)
            {
                std::pair<intV, intV> key = std::make_pair(neigh, i);
                labels[key] = currLabel++; 
            }
            else
                break;
        } 
    }  
    #pragma omp parallel for schedule(dynamic, NUM_THREADS) 
    for (intV i=0; i<numT; i++)
    {
        intV deg = 0;
        std::vector<intV>& neighList = G.get_neigh(i, deg);
        G.eId[i].resize(deg);
        for (intV j=0; j<deg; j++)
        {
            intV neigh = neighList[j];
            std::pair<intV, intV> key = std::make_pair(std::min(i, neigh), std::max(i, neigh));
            G.eId[i][j] = labels[key]; 
        }
    } 
}

void count_per_edge (Graph &G, std::vector<intB> &opCnt)
{
}
