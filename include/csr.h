/*******************************************************************************
    DATA structures to store projection graph
    Functions to create projection graph
    Use CSR format to avoid individual memory allocations for each adj. list
    Functions in this file are SEQUENTIAL as each partition is processed by 
    a single thread
*******************************************************************************/
#ifndef MY_CSR
#define MY_CSR

#include "graph.h"

/*********************************************************
Basic CSR structure.
A thread will instantiate it once and use
it for all the partitions that it processes.
*********************************************************/
typedef struct myCSR
{
    intV numU;
    intV numV;
    intV numT;
    intE numE;
    std::vector<intE> VI;
    std::vector<intV> deg; 
    std::vector<intV> EI;
    std::vector<intB> EW;
} myCSR;




/*********************************************************
Bloom Edge Graph Space Efficient
*********************************************************/
typedef struct BEGraphLoMem
{
    intE numU; // # edges of G
    intB numV; // # blooms
    intB numE; // # edges in BE Index

    std::vector<intB> edgeVI; //offsets for U
    std::vector<intB> bloomVI; //offsets for V
    std::vector<intE> edgeDegree;
    std::vector<intE> bloomDegree;
    std::vector<intE> bloomWdgCnt;
    std::vector<std::pair<intB, intE>> edgeEI; //edges from U to V (bloom ID, neighbor edge ID)
    std::vector<std::pair<intE, intE>> bloomEI; //edges to V to U (transpose of edgeEI)

} BEGraphLoMem;




/******************************************************
create CSR from undeleted vertices
assuming only U vertices are deleted
WILL NOT MODIFY THE GRAPH OBJECT
Arguments:
1. G -> graph object
2. partId -> partitionId
3. partVertices -> list of vertices in the partition 
4. vtxToPart -> map from vertex to partition IDs
5. oG -> output CSR
******************************************************/
void create_part_csr(Graph &G, int partId, std::vector<intV> &partVertices, std::vector<int> &vtxToPart, myCSR &oG)
{

    oG.numU = G.numU;
    oG.numV = G.numV;
    oG.numT = G.numT;

    //create offset array
    if (oG.VI.size() != G.numT+1)
        oG.VI.resize(G.numT+1);

    //reset the degrees
    if (oG.deg.size() != G.numT)
        oG.deg.resize(G.numT, 0);
    else
    {
        for (intV i=0; i<G.numT; i++)
            oG.deg[i] = 0;
    }

    //compute number of edges
    intE numEdges = 0;
    for (auto x : partVertices)
        numEdges += G.deg[x];
    numEdges = 2*numEdges; //undirected edges
    oG.numE = numEdges;

    //allocate space for edges
    if (oG.EI.size() != numEdges)
    {
        oG.EI.clear();
        oG.EI.resize(numEdges);
    }

    //compute degrees and prefix sum for offsets and insert edges
    //for 'U' vertices
    //compute degrees
    for(intV i=0; i<G.numU; i++)
    {
        intV v = G.uLabels[i];
        if (vtxToPart[v]!=partId) //vertices not in this partition
            continue;
        else
        {
            std::vector<intV> &neighList = G.get_neigh(v, oG.deg[v]);
            for (intV j=0; j<oG.deg[v]; j++)
                oG.deg[neighList[j]]++; //increase degree of the neighbor
        }
    }

    //compute offsets
    oG.VI[0] = 0;
    for (intV i=0; i<G.numT; i++)
    {
        oG.VI[i+1] = oG.deg[i] + oG.VI[i];
        oG.deg[i] = 0;
    } 

    //Insert edges (undirected)
    //edges will be sorted by default
    for (intV i=0; i<G.numU; i++)
    {
        intV u = G.uLabels[i];
        if (vtxToPart[u]!=partId) continue;
        std::vector<intV> &neighList = G.get_neigh(u, oG.deg[u]);
        for (intV j=0; j<oG.deg[u]; j++)
        {
            intV v = neighList[j];
            oG.EI[oG.VI[u] + j] = v; 
            oG.EI[oG.VI[v] + oG.deg[v]] = u;
            oG.deg[v]++;
        } 
    }
}


void delete_edges_csr(Graph &G, myCSR &oG)
{
    for (intV i=0; i<G.numU; i++)
    {
        intV u = G.uLabels[i];
        if(G.is_deleted(u)) oG.deg[u] = 0;
    }
    for (intV i=0; i<G.numV; i++)
    {
        intV v = G.vLabels[i];
        intV start = oG.VI[v];
        intV end =  oG.VI[v+1];
        while(start != end)
        {
            if (G.is_deleted(oG.EI[start]))
            {
                end--;
                std::swap(oG.EI[start], oG.EI[end]);
            }
            else
                start++;
        }
        oG.deg[v] = end - oG.VI[v];
        std::sort(oG.EI.begin() + oG.VI[v], oG.EI.begin() + oG.VI[v] + oG.deg[v]);
    }
}

void restore_edges_csr(myCSR &oG)
{
    for (intV i=0; i<oG.numT; i++)
    {
        oG.deg[i] = oG.VI[i+1] - oG.VI[i];
        std::sort(oG.EI.begin() + oG.VI[i], oG.EI.begin() + oG.VI[i] + oG.deg[i]);
    }
}



template <typename T1, typename T2>
class valCompareBloomAdj
{
    const std::vector<T2> &value_vector;

    public:
    valCompareBloomAdj(const std::vector<T2> &val_vec):
        value_vector(val_vec) {}

    bool operator() (const T1 &i1, const T1 &i2) const
    {
        return value_vector[i1.first] < value_vector[i2.first];
    }
};

template<typename ptrType>
inline ptrType insert_part_blooms(unsigned locQWrPtr, const unsigned locQSize, ptrType &globQWrPtr, std::array<intB, locBuffSizeSmall> &locPartBlooms, std::array<intE, locBuffSizeSmall> &locBloomDegree, std::array<intE, locBuffSizeSmall> &locBloomWdgCnt, std::vector<intB> &partBlooms, BEGraphLoMem &partBEG)
{
    if (locQWrPtr >= locQSize)
    {
        intE tempIdx = __sync_fetch_and_add(&globQWrPtr, locQSize);
        for (intE bufIdx = 0; bufIdx < locQSize; bufIdx++)
        {
            intB partBloomId = bufIdx + tempIdx;
            assert(partBloomId < partBEG.numV);
            partBlooms[partBloomId] = locPartBlooms[bufIdx];
            partBEG.bloomDegree[partBloomId] = locBloomDegree[bufIdx];
            partBEG.bloomWdgCnt[partBloomId] = locBloomWdgCnt[bufIdx];
        }
        locQWrPtr = 0;
    }
    return locQWrPtr;
}


/******************************************************
construct BE-Index for all edge partitions in parallel
Inputs:
    1. BEG -> BE-Index for the entire graph
    2. numParts -> number of partitions
    3. partEdges -> 2D array containing edges for each partition
    4. edgeToPart -> mapping from edge index to partition index
    5. edgeToPartEId -> mapping from edge index in graph G to edge index in its partition
    2. isEdgeInSG -> boolean vector indicating if the edge is in subgraph
    3. isEdgeComputed -> boolean vector indicating if the edge has a higher wing number than subgraph
Outputs:
    1. partBEG -> array of BE-Indices for each partition
    2. partBlooms -> 2D array to store blooms for each partition
    3. partWork -> work required to peel each partition
******************************************************/
void construct_part_BEG(BEGraphLoMem &BEG, int numParts, std::vector<std::vector<intE>> &partEdges, std::vector<std::vector<intB>> &partBlooms, std::vector<int> &edgeToPart, std::vector<intE> &edgeToPartEId, std::vector<BEGraphLoMem> &partBEG, std::vector<intB> &partWork)
{
    double start, end;

    free_vec(BEG.edgeVI);
    free_vec(BEG.edgeEI);
    free_vec(BEG.edgeDegree);
    parallel_init(BEG.bloomDegree, (intE)0);

    std::vector<intB> partBloomPtr (numParts, 0);
    

    valCompareBloomAdj<std::pair<intE, intE>, int> adjCompObj (edgeToPart); 

    intB work = 0;
    #pragma omp parallel num_threads (NUM_THREADS)
    {
        size_t tid = omp_get_thread_num();
        std::vector<std::array<intB, locBuffSizeSmall>> locPartBlooms (numParts);
        std::vector<std::array<intE, locBuffSizeSmall>> locBloomDegree (numParts);
        std::vector<std::array<intE, locBuffSizeSmall>> locBloomWdgCnt (numParts);
        std::vector<unsigned> locPartBloomPtr (numParts, 0);
        //sort edges of each bloom on min part ID (eId.first, eId.second)
        //find # blooms in each partition
        #pragma omp for schedule (dynamic, 5)
        for (intB i=0; i<BEG.numV; i++)
        {
            for (intB j=BEG.bloomVI[i]; j<BEG.bloomVI[i+1]; j++)
            {
                std::pair<intE, intE> &ep = BEG.bloomEI[j];
                if (edgeToPart[ep.first] > edgeToPart[ep.second])
                    std::swap(ep.first, ep.second);
            }
            std::sort(BEG.bloomEI.begin() + BEG.bloomVI[i], BEG.bloomEI.begin()+BEG.bloomVI[i+1], adjCompObj);

            intE bloomPartDeg = 0;
            int prevPart = numParts + 1;
            for (intB j=BEG.bloomVI[i]; j<BEG.bloomVI[i+1]; j++)
            {
                intE e1 = BEG.bloomEI[j].first;
                intE e2 = BEG.bloomEI[j].second;

                int currPart = edgeToPart[e1];

                if (j > BEG.bloomVI[i]) assert (currPart >= prevPart);

                if (currPart == prevPart) bloomPartDeg++; else bloomPartDeg = 1;
                if (bloomPartDeg==2) locPartBloomPtr[currPart]++;

                if (edgeToPart[e2] == currPart)
                {
                    bloomPartDeg++;
                    if (bloomPartDeg==2) locPartBloomPtr[currPart]++;
                }

                prevPart = currPart;
            }
        }
        for (int i=0; i<numParts; i++)
        {
            __sync_fetch_and_add(&partBloomPtr[i], locPartBloomPtr[i]);
            locPartBloomPtr[i] = 0;
        }
        #pragma omp barrier
        
        #pragma omp single
        start   = omp_get_wtime();
        
        #pragma omp for schedule (dynamic, 1)
        for (int i=0; i<numParts; i++) 
        {
            partBlooms[i].resize(partBloomPtr[i], 0); partBloomPtr[i] = 0; 
            partBEG[i].numV = partBlooms[i].size(); 
            partBEG[i].numU = partEdges[i].size(); 
            partBEG[i].bloomDegree.resize(partBEG[i].numV, 0);
            partBEG[i].bloomWdgCnt.resize(partBEG[i].numV, 0); 
            partBEG[i].edgeDegree.resize(partBEG[i].numU, 0); 
        }
        #pragma omp barrier
    
        #pragma omp single
        {
            end = omp_get_wtime();
            MEM_ALLOC_TIME  += end-start;
        }

        //add blooms for each partition into corresponding partBlooms vector;
        //add their degrees and wedge counts as well
        #pragma omp for schedule (dynamic, 5)
        for (intB i=0; i<BEG.numV; i++)
        {
            intE bloomPartDeg = 0;
            intE bloomPairDeg = 0;
            bool recordBloom = false;
            int prevPart = numParts + 1; 
            intE bloomWdgCnt = 0;    
            //traverse edges of each bloom in global bloom structure
            for (intB j=BEG.bloomVI[i]; j<BEG.bloomVI[i+1]; j++)
            {
                intE e1 = BEG.bloomEI[j].first;
                intE e2 = BEG.bloomEI[j].second;

                int currPart = edgeToPart[e1];
                if (j > BEG.bloomVI[i]) assert (currPart >= prevPart);

                if (currPart == prevPart) 
                {
                    //if same bloom, continue increasing the degree
                    bloomPartDeg++; 
                    bloomPairDeg++;
                }
                else 
                {
                    //if bloom changes and current bloom is marked for recording
                    if (recordBloom)
                    {
                        assert(bloomPartDeg >= 2);
                        locBloomDegree[prevPart][locPartBloomPtr[prevPart]++] = bloomPairDeg;
                        locPartBloomPtr[prevPart] = insert_part_blooms(locPartBloomPtr[prevPart], locBuffSizeSmall, partBloomPtr[prevPart],
                                                           locPartBlooms[prevPart], locBloomDegree[prevPart], locBloomWdgCnt[prevPart],
                                                           partBlooms[prevPart], partBEG[prevPart]); 
                    }
                    recordBloom = false;
                    //re-initialize 
                    bloomPartDeg = 1; //edges in the partition connected to this bloom
                    bloomPairDeg = 1; //edge pairs (with e1 in this part) connected to this bloom
                    bloomWdgCnt = BEG.bloomVI[i+1]-j; //wedges with both edges in current or higher partition
                }

                //add a bloom only if it has degree >= 2
                if (bloomPartDeg==2) 
                {
                    recordBloom = true;
                    locPartBlooms[currPart][locPartBloomPtr[currPart]] = i;
                    locBloomWdgCnt[currPart][locPartBloomPtr[currPart]] = bloomWdgCnt;
                }

                if (edgeToPart[e2] == currPart)
                {
                    bloomPartDeg++;
                    if (bloomPartDeg==2) 
                    {
                        recordBloom = true;
                        locPartBlooms[currPart][locPartBloomPtr[currPart]] = i;
                        locBloomWdgCnt[currPart][locPartBloomPtr[currPart]] = bloomWdgCnt;
                    }
                }
                prevPart = currPart;
            }
            //record the bloom for the last partition in its adjacency
            if (recordBloom)
            {
                locBloomDegree[prevPart][locPartBloomPtr[prevPart]++] = bloomPairDeg;
                locPartBloomPtr[prevPart] = insert_part_blooms(locPartBloomPtr[prevPart], locBuffSizeSmall, partBloomPtr[prevPart],
                                                   locPartBlooms[prevPart], locBloomDegree[prevPart], locBloomWdgCnt[prevPart],
                                                   partBlooms[prevPart], partBEG[prevPart]); 
            }
        }
        for (int i=0; i<numParts; i++)
        {
            if (locPartBloomPtr[i] > 0)
            {
                locPartBloomPtr[i] = insert_part_blooms(locPartBloomPtr[i], locPartBloomPtr[i], partBloomPtr[i], locPartBlooms[i],
                                                     locBloomDegree[i], locBloomWdgCnt[i], partBlooms[i], partBEG[i]); 

            }
        }
         


        #pragma omp barrier

        #pragma omp for schedule (dynamic, 1)
        for (int i=0; i<numParts; i++)
            serial_prefix_sum(partBEG[i].bloomVI, partBEG[i].bloomDegree);

        #pragma omp single
        start   = omp_get_wtime();

        #pragma omp for schedule (dynamic, 1)
        for (int i=0; i<numParts; i++)
        {
            partBEG[i].bloomEI.resize(partBEG[i].bloomVI.back(), std::make_pair<intE, intE>(0,0));
            partBloomPtr[i] = 0;
        }

        #pragma omp barrier

        #pragma omp single
        {
            end = omp_get_wtime();
            MEM_ALLOC_TIME  += end-start;
        }

        #pragma omp barrier


        //for each partition
        for (int i=0; i<numParts; i++)
        {
            #pragma omp barrier
            //for each bloom in that partition
            #pragma omp for schedule (dynamic, 5) reduction (+:work)
            for (intB j=0; j<partBEG[i].numV; j++)
            {
                //probe only those edge pairs(wedges) that are in this partition
                intB bloomId = partBlooms[i][j];
                intE maxDeg = BEG.bloomVI[bloomId+1] - BEG.bloomVI[bloomId];
                intE partBloomDeg = 0;
                intE bloomWork = 0;
                while(BEG.bloomDegree[bloomId] < maxDeg) 
                {
                    std::pair<intE, intE> &ep = BEG.bloomEI[BEG.bloomVI[bloomId]+BEG.bloomDegree[bloomId]];
                    int currPart = edgeToPart[ep.first];

                    //remaining wedges belong to higher partition
                    if (currPart > i) break; 
                    BEG.bloomDegree[bloomId]++;

                    //this bloom didn't get included in "currPart"
                    //skip those wedges
                    if (currPart < i) continue; 

                    intE partE1, partE2;
                    partE1 = edgeToPartEId[ep.first]; __sync_fetch_and_add(&partBEG[i].edgeDegree[partE1], 1); 
                    bloomWork++;
                    if (edgeToPart[ep.second]==i)
                    {
                        bloomWork++;
                        partE2 = edgeToPartEId[ep.second];
                        __sync_fetch_and_add(&partBEG[i].edgeDegree[partE2], 1);
                    }
                    else  partE2 = partBEG[i].numU+1; //other edges in diff. partition. Don't process later

                    assert(partBloomDeg < partBEG[i].bloomDegree[j]);
                    partBEG[i].bloomEI[partBEG[i].bloomVI[j] + partBloomDeg++] = std::make_pair(partE1, partE2);
                }
                work += partBloomDeg*(partBloomDeg-1);
            }    
            #pragma omp barrier
            #pragma omp single
            {
                partWork[i] = work;
                work = 0;
            }
        }

        #pragma omp barrier

        #pragma omp single
        {
            free_vec(BEG.bloomEI); 
            free_vec(BEG.bloomDegree); 
            free_vec(BEG.bloomWdgCnt);
            free_vec(BEG.bloomVI);
        }

        #pragma omp for schedule (dynamic, 1)
        for (int i=0; i<numParts; i++)
            serial_prefix_sum(partBEG[i].edgeVI, partBEG[i].edgeDegree);

        #pragma omp single
        start   = omp_get_wtime();

        #pragma omp for
        for (int i=0; i<numParts; i++)
        {
            partBEG[i].numE = partBEG[i].edgeVI.back(); 
            partBEG[i].edgeEI.resize(partBEG[i].numE, std::make_pair<intB, intE>(0, 0));
        }

        #pragma omp barrier

        #pragma omp single
        {
            end = omp_get_wtime();
            MEM_ALLOC_TIME  += end-start;
        }

        #pragma omp barrier
        
        
        //transpose partBEG[i].bloomEI to create partBEG[i].edgeEI
        for (int i=0; i<numParts; i++)
        {
            #pragma omp for
            for (intE j=0; j<partBEG[i].numU; j++) partBEG[i].edgeDegree[j] = 0;
            #pragma omp barrier
            #pragma omp for
            for (intB j=0; j<partBEG[i].numV; j++)
            {
                for (intB k=partBEG[i].bloomVI[j]; k<partBEG[i].bloomVI[j+1]; k++)
                {
                    intE e1 = partBEG[i].bloomEI[k].first;
                    intE e2 = partBEG[i].bloomEI[k].second;
                    intE e1Idx = __sync_fetch_and_add(&partBEG[i].edgeDegree[e1], 1);
                    partBEG[i].edgeEI[partBEG[i].edgeVI[e1] + e1Idx] = std::make_pair(j, e2);
                    if (e2 > partBEG[i].numU) continue;
                    intE e2Idx = __sync_fetch_and_add(&partBEG[i].edgeDegree[e2], 1);
                    partBEG[i].edgeEI[partBEG[i].edgeVI[e2] + e2Idx] = std::make_pair(j, e1);
                }
            } 
        }
    }
#ifdef DEBUG
    printf("part BEGs constructed\n");
#endif
}



#endif
