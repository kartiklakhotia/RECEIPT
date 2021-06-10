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
    std::unordered_map<uint64_t, intE> labels;
    labels.reserve(2*G.numE);
    intE currLabel = 0;
    for (intV i=0; i<G.numT; i++)
    {
        intV deg = 0;
        std::vector<intV>& neighList = G.get_neigh(i, deg);
        for (intV j=0; j<deg; j++)
        {
            intV neigh = neighList[j];
            if (neigh < i)
            {
                uint64_t key = ((uint64_t)i)*((uint64_t)G.numT) + ((uint64_t)neigh);
                labels[key] = currLabel++; 
            }
            else
                break;
        } 
    }  
    
    //printf("labels created. max label = %u\n", currLabel);

    G.eId.resize(G.numT);
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, NUM_THREADS) 
    for (intV i=0; i<G.numT; i++)
    {
        intV deg = 0;
        std::vector<intV>& neighList = G.get_neigh(i, deg);
        G.eId[i].resize(deg);
        for (intV j=0; j<deg; j++)
        {
            intV neigh = neighList[j];
            uint64_t key = ((uint64_t)std::max(i, neigh))*((uint64_t)G.numT) + ((uint64_t)std::min(i,neigh));
            G.eId[i][j] = labels[key]; 
        }
    } 
}

//map edge ID to linked vertices
void edgeId_to_vertices (Graph &G, std::vector<std::pair<intV, intV>> &eIdToV)
{
    if (eIdToV.size() < G.numE) eIdToV.resize(G.numE);
    #pragma omp parallel for num_threads(NUM_THREADS) schedule (dynamic, NUM_THREADS)
    for (intV i=0; i<G.numT; i++)
    {
        for (intV j=0; j<G.deg[i]; j++)
        {
            intV neigh = G.adj[i][j];
            if (neigh >= i) break;
            intE eId = G.eId[i][j];
            eIdToV[eId] = std::make_pair(i, neigh);
            assert(neigh < G.numT);
        } 
    }  
}

/*****************************************************************************
Count butterflies per edge
Inputs:
    1. G -> graph object
Outputs:
    1. opCnt -> per edge butterfly counts
Arguments:
    1. wedgeCnt -> 2D array used to accumulate wedges during counting
******************************************************************************/
void count_per_edge (Graph &G, std::vector<intE> &opCnt, std::vector<std::vector<intV>> &wedgeCnt)
{
    if (opCnt.size() != G.numE)
        opCnt.resize(G.numE);
    #pragma omp parallel for num_threads(NUM_THREADS) 
    for (intV i=0; i<G.numE; i++)
        opCnt[i] = 0;

    #pragma omp parallel num_threads(NUM_THREADS) 
    {
        size_t tid = omp_get_thread_num();
        std::vector<intV> &numW = wedgeCnt[tid];
        std::vector<intV> hop2Neighs (std::max(G.numU, G.numV));
        intV numHop2Neighs = 0;
        #pragma omp for schedule(dynamic, 6)
        for (intV i=0; i<G.numT; i++)
        {
            std::vector<intV> &neighList = G.adj[i];
            //count wedges
            for (intV j=0; j<G.deg[i]; j++)
            {
                intV neigh = neighList[j];
                std::vector<intV>& neighOfNeighList = G.adj[neigh];
                for (intV k=0; k<G.deg[neigh]; k++)
                {
                    intV neighOfNeigh = neighOfNeighList[k];
                    if ((neighOfNeigh>=neigh)||(neighOfNeigh>=i))
                        break;
                    if (numW[neighOfNeigh]==0)
                        hop2Neighs[numHop2Neighs++] = neighOfNeigh;
                    numW[neighOfNeigh]++;
                }
            }


            for (intV j=0; j<G.deg[i]; j++)
            {
                intE e1 = G.eId[i][j];
                intV neigh = neighList[j];
                std::vector<intV>& neighOfNeighList = G.adj[neigh];
                intB locSum = 0;
                for (intV k=0; k<G.deg[neigh]; k++)
                {
                    intE e2 = G.eId[neigh][k];
                    intV neighOfNeigh = neighOfNeighList[k];
                    if ((neighOfNeigh>=neigh)||(neighOfNeigh>=i))
                        break;
                    if (numW[neighOfNeigh] < 2) continue; 
                    intB butterflies = numW[neighOfNeigh] - 1;
                    locSum += butterflies;
                    __sync_fetch_and_add(&opCnt[e2], butterflies);
                }
                __sync_fetch_and_add(&opCnt[e1], locSum);
            }

            for (intV j=0; j<numHop2Neighs; j++)
                numW[hop2Neighs[j]] = 0;
            numHop2Neighs = 0;
           

        }
    
    }
}


/*****************************************************************************
Estimate the size of BE-Index
Inputs:
    1. G -> graph object
Outputs:
    1.Prints the size of BE-Index to std terminal
Arguments:
    1. wedgeCnt -> 2D array used to accumulate wedges during counting
******************************************************************************/
void estimate_BE_Index_size (Graph &G, std::vector<std::vector<intV>> &wedgeCnt)
{
    intB numBlooms = 0;
    intB edgesBE = 0;


    //find number of blooms (total and per-vertex)
    #pragma omp parallel num_threads(NUM_THREADS) 
    {
        size_t tid = omp_get_thread_num();
        std::vector<intV> &numW = wedgeCnt[tid];
        std::vector<intV> hop2Neighs (std::max(G.numU, G.numV));
        intV hop2NeighsPtr = 0;
        #pragma omp parallel for schedule (dynamic, 5) reduction (+:numBlooms, edgesBE)
        for (intV i=0; i<G.numT; i++)
        {
            std::vector<intV> &neighList = G.adj[i];
            //count wedges
            for (intV j=0; j<G.deg[i]; j++)
            {
                intV neigh = neighList[j];
                std::vector<intV>& neighOfNeighList = G.adj[neigh];
                for (intV k=0; k<G.deg[neigh]; k++)
                {
                    intV neighOfNeigh = neighOfNeighList[k];
                    if ((neighOfNeigh>=neigh)||(neighOfNeigh>=i))
                        break;
                    if (numW[neighOfNeigh]==0)
                        hop2Neighs[hop2NeighsPtr++] = neighOfNeigh;
                    numW[neighOfNeigh]++;
                }
            }
            for (intV j=0; j<G.deg[i]; j++)
            {
                intE e1 = G.eId[i][j];
                intV neigh = neighList[j];
                for (intV k=0; k<G.deg[neigh]; k++)
                {
                    intV neighOfNeigh = G.adj[neigh][k];
                    if ((neighOfNeigh >= neigh) || (neighOfNeigh >= i)) break;
                    edgesBE += 2;
                }
            } 
            for (intV j=0; j<hop2NeighsPtr; j++)
            {
                numBlooms++;
                numW[hop2Neighs[j]] = 0;
            }
            hop2NeighsPtr = 0;
        }
    }
    std::cout << "number of blooms = " << numBlooms << ", ";
    std::cout << "number of edges in BE index = " << edgesBE << std::endl;
    std::cout << "size in bytes = " << edgesBE*16 + numBlooms*8 << std::endl;
}



/*****************************************************************************
Compute space efficient BE-Index
Inputs:
    1. G -> graph object
Outputs:
    1. opCnt -> per edge butterfly counts
    2. BEG -> BE-Index as a bipartite graph
Arguments:
    1. wedgeCnt -> 2D array used to accumulate wedges during counting
******************************************************************************/
void count_and_create_BE_Index (Graph &G, std::vector<intE> &opCnt, BEGraphLoMem &BEG, std::vector<std::vector<intV>> &wedgeCnt)
{

    double start    = omp_get_wtime();

    if (opCnt.size() < G.numE)
        opCnt.resize(G.numE);
    parallel_init(opCnt, (intE)0);

    std::vector<intE> perVertexBloomEdges (G.numT); 
    std::vector<intV> perVertexBlooms (G.numT); 
    std::vector<intE> perEdgeBlooms (G.numE); parallel_init(perEdgeBlooms, (intE)0);

    double end      = omp_get_wtime();
    MEM_ALLOC_TIME  += end-start;

    //count butterflies and find number of blooms (total and per-vertex)
    #pragma omp parallel num_threads (NUM_THREADS)
    {
        size_t tid = omp_get_thread_num();
        std::vector<intV> &numW = wedgeCnt[tid];
        std::vector<intV> hop2Neighs (std::max(G.numU, G.numV));
        intV hop2NeighsPtr = 0;
        #pragma omp for schedule (dynamic, 5)
        for (intV i=0; i<G.numT; i++)
        {
            //count wedges
            intE vBloomEdges = 0;
            intV vBloomCnt = 0;
            for (intV j=0; j<G.deg[i]; j++)
            {
                intV neigh = G.adj[i][j];
                for (intV k=0; k<G.deg[neigh]; k++)
                {
                    intV neighOfNeigh = G.adj[neigh][k];
                    if ((neighOfNeigh>=neigh)||(neighOfNeigh>=i))
                        break;
                    if (numW[neighOfNeigh]==0)
                        hop2Neighs[hop2NeighsPtr++] = neighOfNeigh;
                    numW[neighOfNeigh]++;
                }
            }
            for (intV j=0; j<G.deg[i]; j++)
            {
                intE e1 = G.eId[i][j];
                intE e1ButterflyLocSum = 0; //butterflies of e1
                intE e1BloomLocSum = 0; //number of bloom connections of e1
                intV neigh = G.adj[i][j];
                for (intV k=0; k<G.deg[neigh]; k++)
                {
                    intV neighOfNeigh = G.adj[neigh][k];
                    if ((neighOfNeigh >= neigh) || (neighOfNeigh >= i)) break;
                    if (numW[neighOfNeigh] < 2) continue;

                    intE e2 = G.eId[neigh][k];
                    vBloomEdges += 1;

                    intE butterflies = numW[neighOfNeigh] - 1;
                    __sync_fetch_and_add(&opCnt[e2], butterflies);
                    __sync_fetch_and_add(&perEdgeBlooms[e2], 1);


                    

                    e1ButterflyLocSum += butterflies;
                    e1BloomLocSum += 1;
                }
                __sync_fetch_and_add(&opCnt[e1], e1ButterflyLocSum);
                __sync_fetch_and_add(&perEdgeBlooms[e1], e1BloomLocSum);
            } 
            for (intV j=0; j<hop2NeighsPtr; j++)
            {
                if (numW[hop2Neighs[j]] >= 2) 
                    vBloomCnt++;
                numW[hop2Neighs[j]] = 0;
            }
            perVertexBlooms[i] = vBloomCnt;
            perVertexBloomEdges[i] += vBloomEdges;
            hop2NeighsPtr = 0;
        }
    }


    std::vector<intB> vertexBloomOffset;
    std::vector<intB> vertexEdgeOffset;

    parallel_prefix_sum(vertexEdgeOffset, perVertexBloomEdges);
    free_vec(perVertexBloomEdges);

    parallel_prefix_sum(vertexBloomOffset, perVertexBlooms);
    free_vec(perVertexBlooms);
    

    //initialize BE Graph
    BEG.numU = G.numE;
    BEG.numV = vertexBloomOffset[G.numT];
    BEG.numE = 2*vertexEdgeOffset[G.numT];


    start   = omp_get_wtime();
    
    BEG.edgeEI.resize(BEG.numE); parallel_init(BEG.edgeEI, std::make_pair<intB, intE>(0,0)); 
    BEG.bloomEI.resize(BEG.numE/2); parallel_init(BEG.bloomEI, std::make_pair<intE, intE>(0, 0));
    BEG.bloomVI.resize(BEG.numV+1); BEG.bloomVI[0] = 0; parallel_init(BEG.bloomVI, (intB)0); 
    BEG.bloomDegree.resize(BEG.numV); parallel_init(BEG.bloomDegree, (intE)0);

    end     = omp_get_wtime();
    MEM_ALLOC_TIME += end-start;

    parallel_prefix_sum(BEG.edgeVI, perEdgeBlooms); 
    
    parallel_init(perEdgeBlooms, (intE)0); 
    parallel_init(BEG.bloomDegree, (intE)0);
    
    //fill adjacencies of edges in BEgraph
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        size_t tid = omp_get_thread_num();
        std::vector<intV> &numW = wedgeCnt[tid];
        std::vector<intV> hop2Neighs (std::max(G.numU, G.numV));
        intV hop2NeighsPtr = 0;
        #pragma omp for schedule (dynamic, 5)
        for (intV i=0; i<G.numT; i++)
        {
            for (intV j=0; j<G.deg[i]; j++)
            {
                intV neigh = G.adj[i][j];
                for (intV k=0; k<G.deg[neigh]; k++)
                {
                    intV neighOfNeigh = G.adj[neigh][k];
                    if ((neighOfNeigh>=neigh)||(neighOfNeigh>=i))
                        break;
                    if (numW[neighOfNeigh]==0)
                        hop2Neighs[hop2NeighsPtr++] = neighOfNeigh;
                    numW[neighOfNeigh]++;
                }
            }
            //assign indices to blooms and compute degrees
            intB bloomIdBase = vertexBloomOffset[i];
            intV bloomIdOff = 1; //bloom ID = bloomIDBase + bloomIDOff - 1
            BEG.bloomVI[bloomIdBase] = vertexEdgeOffset[i];
            for (intV j=0; j<hop2NeighsPtr; j++)
            {
                intV neighOfNeigh = hop2Neighs[j];
                if (numW[neighOfNeigh] >= 2) 
                {
                    intB bloomId = bloomIdBase + (intB)bloomIdOff - 1;
                    assert(bloomId < vertexBloomOffset[i+1]);
                    
                    BEG.bloomVI[bloomId+1] = BEG.bloomVI[bloomId] + numW[neighOfNeigh];

                    //numW now represents the bloomIndex (and not the # wedges)
                    numW[neighOfNeigh] = bloomIdOff++;
                }
                else
                    numW[neighOfNeigh] = 0;
            }

            for (intV j=0; j<G.deg[i]; j++)
            {
                intE e1 = G.eId[i][j];
                intV neigh = G.adj[i][j];
                for (intV k=0; k<G.deg[neigh]; k++)
                {
                    intV neighOfNeigh = G.adj[neigh][k];
                    if ((neighOfNeigh >= neigh) || (neighOfNeigh >= i)) break;
                    if (numW[neighOfNeigh] == 0) continue;

                    intE e2 = G.eId[neigh][k];

                    intB bloomId = bloomIdBase + (intB)numW[neighOfNeigh] - 1;
                    intB BEedgeId = BEG.bloomVI[bloomId] + BEG.bloomDegree[bloomId]; 

                    BEG.bloomEI[BEedgeId] = std::make_pair(e1, e2);

                    BEG.bloomDegree[bloomId] += 1;

                    BEedgeId = BEG.edgeVI[e1] + __sync_fetch_and_add(&perEdgeBlooms[e1], 1);
                    BEG.edgeEI[BEedgeId] = std::make_pair(bloomId, e2);
            
                    BEedgeId = BEG.edgeVI[e2] + __sync_fetch_and_add(&perEdgeBlooms[e2], 1);
                    BEG.edgeEI[BEedgeId] = std::make_pair(bloomId, e1);
                }
            } 
            for (intV j=0; j<hop2NeighsPtr; j++)
                numW[hop2Neighs[j]] = 0;
            hop2NeighsPtr = 0;
        }
    }
    BEG.edgeDegree.swap(perEdgeBlooms);
    
#ifdef DEBUG
    std::cout << "number of blooms = " << BEG.numV << ", ";
    std::cout << "number of edges in BE index = " << BEG.numE << std::endl;
    std::cout << "size in bytes = " << BEG.numE*16 + (BEG.numU+BEG.numV)*12 << std::endl;
#endif
}




/*****************************************************************************
Compute space efficient BE-Index for a subgraph (for progressive compression peeling)
Inputs:
    1. G -> graph object
    2. isEdgeInSG -> boolean vector indicating if the edge is in subgraph
    3. isEdgeComputed -> boolean vector indicating if the edge has a higher wing number than subgraph
Outputs:
    1. opCnt -> per edge butterfly counts
    2. BEG -> BE-Index as a bipartite graph
Arguments:
    1. wedgeCnt -> 2D array used to accumulate wedges during counting
******************************************************************************/
void sg_count_and_create_BE_Index (Graph &G, std::vector<bool> &isEdgeInSG, std::vector<bool> &isEdgeComputed, std::vector<intE> &opCnt, BEGraphLoMem &BEG, std::vector<std::vector<intV>> &wedgeCnt)
{
    double start, end;

    
    std::vector<std::pair<intV, intE>> edgePair (std::max(G.numU, G.numV));
    for (intV i=0; i<G.numT; i++)
    {
        if (G.deg[i]==0) continue;
        for (intV j=0; j<G.deg[i]; j++)
            edgePair[j] = std::make_pair(G.adj[i][j], G.eId[i][j]);
        std::sort(edgePair.begin(), edgePair.begin()+G.deg[i]);
        for (intV j=0; j<G.deg[i]; j++)
        {
            G.adj[i][j] = edgePair[j].first;
            G.eId[i][j] = edgePair[j].second;
        } 
    }

    if (opCnt.size() < G.numE)
    {
        opCnt.resize(G.numE);
        for (intE i=0; i<G.numE; i++)
            opCnt[i] = 0;
    }

    std::vector<intE> perEdgeBlooms (G.numE, 0);

    //count butterflies and find number of blooms (total and per-vertex)
    std::vector<intV> &numW = wedgeCnt[0];
    std::vector<intV> hop2Neighs (std::max(G.numU, G.numV));
    std::vector<intV> bloomExists (G.numT, 0);
    intV hop2NeighsPtr = 0;

    intV bloomId = 0;
    intB vertexBaseBloomId = 0;
    BEG.bloomVI.push_back(0);
    for (intV i=0; i<G.numT; i++)
    {
        vertexBaseBloomId += bloomId;
        bloomId = 0;
        //count wedges
        for (intV j=0; j<G.deg[i]; j++)
        {
            intV neigh = G.adj[i][j];
            assert(neigh < G.numT);
            for (intV k=0; k<G.deg[neigh]; k++)
            {
                intV neighOfNeigh = G.adj[neigh][k];
                assert(neighOfNeigh < G.numT);
                if ((neighOfNeigh>=neigh)||(neighOfNeigh>=i))
                    break;

                assert(hop2NeighsPtr<std::max(G.numU, G.numV));
                if (numW[neighOfNeigh]==0)
                    hop2Neighs[hop2NeighsPtr++] = neighOfNeigh;
                numW[neighOfNeigh]++;
            }
        }
        for (intV j=0; j<G.deg[i]; j++)
        {
            intE e1 = G.eId[i][j]; assert(e1 < G.numE);
            assert(isEdgeInSG[e1]);
            intB e1ButterflyLocSum = 0; //butterflies of e1
            intB e1BloomLocSum = 0; //number of bloom connections of e1

            intV neigh = G.adj[i][j];


            for (intV k=0; k<G.deg[neigh]; k++)
            {
                intV neighOfNeigh = G.adj[neigh][k];
                intE e2 = G.eId[neigh][k];  assert(isEdgeInSG[e2] && e2<G.numE);
                if ((neighOfNeigh >= neigh) || (neighOfNeigh >= i)) break;
                if (numW[neighOfNeigh] < 2) continue;
                if (isEdgeComputed[e1] && isEdgeComputed[e2]) continue;

                assert(isEdgeInSG[e2]);
                intB butterflies = numW[neighOfNeigh] - 1;

                if ((!isEdgeComputed[e2]) || (!isEdgeComputed[e1])) bloomExists[neighOfNeigh]++;

                if (!isEdgeComputed[e2])
                {
                    opCnt[e2] += butterflies;
                    perEdgeBlooms[e2] += 1;
                }
                if (!isEdgeComputed[e1])
                {
                    e1ButterflyLocSum += butterflies;
                    e1BloomLocSum += 1;
                }
            }
            if (!isEdgeComputed[e1])
            {
                opCnt[e1] += e1ButterflyLocSum;
                perEdgeBlooms[e1] += e1BloomLocSum;
            }
        } 


        for (intV j=0; j<hop2NeighsPtr; j++)
        {
            intV neighOfNeigh = hop2Neighs[j];
            if (bloomExists[neighOfNeigh] > 0)
            {
                assert(numW[neighOfNeigh]>=2);
                intE bloomDeg = bloomExists[neighOfNeigh];
                BEG.bloomDegree.push_back(0);
                BEG.bloomVI.push_back(BEG.bloomVI.back() + bloomDeg);
                BEG.bloomWdgCnt.push_back(numW[neighOfNeigh]);
                numW[neighOfNeigh] = bloomId++; 
            }
            else    
                numW[neighOfNeigh] = 0;
        }
        for (intB j=BEG.bloomVI[vertexBaseBloomId]; j<BEG.bloomVI[vertexBaseBloomId+bloomId]; j++)
            BEG.bloomEI.push_back(std::make_pair<intE, intE>(0, 0));

        for (intV j=0; j<G.deg[i]; j++)
        {
            intE e1 = G.eId[i][j];
            intV neigh = G.adj[i][j];

            for (intV k=0; k<G.deg[neigh]; k++)
            {
                intV neighOfNeigh = G.adj[neigh][k];
                intE e2 = G.eId[neigh][k];
                if ((neighOfNeigh >= neigh) || (neighOfNeigh >= i)) break;
                if (bloomExists[neighOfNeigh] == 0) continue;
                if (isEdgeComputed[e1] && isEdgeComputed[e2]) continue;

                intB bId = vertexBaseBloomId + numW[neighOfNeigh]; 
                assert(bId < BEG.bloomVI.size()-1);

                if (!isEdgeComputed[e2] || !isEdgeComputed[e1])
                {
                    assert(BEG.bloomVI[bId]+BEG.bloomDegree[bId] < BEG.bloomEI.size());
                    BEG.bloomEI[BEG.bloomVI[bId]+BEG.bloomDegree[bId]++] = std::make_pair(e2, e1);
                }

                assert(BEG.bloomDegree[bId] <= bloomExists[neighOfNeigh]);
            }
        } 

        for (intV j=0; j<hop2NeighsPtr; j++)
        {
            intV neighOfNeigh = hop2Neighs[j];
            numW[neighOfNeigh] = 0;
            bloomExists[neighOfNeigh] = 0;
        }
        hop2NeighsPtr = 0; 
    }

    //initialize BE Graph
    BEG.numU = G.numE;
    BEG.numV = vertexBaseBloomId + bloomId; assert(vertexBaseBloomId + bloomId < BEG.bloomVI.size());
    BEG.numE = 2*BEG.bloomVI[BEG.numV]; assert(BEG.numE <= 2*BEG.bloomEI.size());
    BEG.edgeDegree.swap(perEdgeBlooms);

    start   = omp_get_wtime();

    BEG.edgeEI.resize(BEG.numE); parallel_init(BEG.edgeEI, std::make_pair<intB, intE>(0, 0));
    BEG.edgeVI.resize(G.numE+1); BEG.edgeVI[0] = 0;

    end     = omp_get_wtime();
    MEM_ALLOC_TIME += end-start;


    for (intE i=0; i<G.numE; i++) {BEG.edgeVI[i+1] = BEG.edgeVI[i]+BEG.edgeDegree[i]; BEG.edgeDegree[i] = 0;}
    
    //fill adjacencies of edges in BEgraph
    for (auto i=0; i<BEG.numV; i++)
    {
        for (auto j=BEG.bloomVI[i]; j<BEG.bloomVI[i+1]; j++)
        {
            intE e1 = BEG.bloomEI[j].first;
            intE e2 = BEG.bloomEI[j].second;
            if (!isEdgeComputed[e1])
            {
                assert(BEG.edgeVI[e1]+BEG.edgeDegree[e1] < BEG.numE);
                BEG.edgeEI[BEG.edgeVI[e1]+BEG.edgeDegree[e1]++] = std::make_pair(i, e2);
            }
            if (!isEdgeComputed[e2])
            {
                assert(BEG.edgeVI[e2]+BEG.edgeDegree[e2] < BEG.numE);
                BEG.edgeEI[BEG.edgeVI[e2]+BEG.edgeDegree[e2]++] = std::make_pair(i,e1);
            }
        }
    }
    
#ifdef DEBUG
    std::cout << "number of blooms = " << BEG.numV << std::endl;
    std::cout << "number of edges in BE index = " << BEG.numE << std::endl;
    std::cout << "size in bytes = " << BEG.numE*20 + (BEG.numU+BEG.numV)*8 << std::endl;
#endif
}



