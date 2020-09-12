#include "peel.h"
#include "csr.h"
#include "kheap.h"

/*****************************************************************************
Re-count and generate updates for 2-hop neighbors of deleted vertices
Used for fine-grained peeling within a partition
SERIAL
Put in update list only if count + non native support differs from current supp
Inputs:
    1. partG -> csr matrix for edges incident on current partition
    1. G-> graph object
    2. labels -> vertices to be peeled but not deleted yet
    3. activeList -> vertices peeled in this round
    4. isActive -> boolean vector mapping a vertex ID to its active status
    5. currSupport -> current support of vertices
    6. nonNativeSupport -> support contrib from vertices of other partitions
    7. wedgeCnt -> 2D array for threads to store wedges while counting
Outputs:
    1. updateVertexList -> list of vertices whose support values are updated
    2. updateValueList -> corresponding values by which support should be reduced
******************************************************************************/
intV return_updates_by_counting_part_ser(myCSR &partG, Graph &G, std::vector<intV> &labels, std::vector<intV> &activeList, std::vector<uint8_t> &isActive, std::vector<intB> &currSupport, std::vector<intB> &nonNativeSupport, std::vector<intV> &updateVertexList, std::vector<intB> &updateValueList, std::vector<intV> &wedgeCnt)
{
    std::vector<intB> bcnt;
    count_per_vertex (partG, G, labels, bcnt, wedgeCnt);

    std::vector<uint8_t> differs(labels.size());
    for (intV i=0; i<labels.size(); i++)
    {
        auto v = labels[i];
        differs[i] = (((bcnt[v] + nonNativeSupport[v]) != currSupport[v]) && (!G.is_deleted(v)) && (!isActive[v])) ? 1 : 0; 
    }
    intV numUpdates = sequential_compact<intV, intV>(labels, differs, updateVertexList);
    if (updateValueList.size() < numUpdates)
    {
        updateValueList.clear();
        updateValueList.resize(numUpdates);
    }
    for (intV i=0; i<numUpdates; i++)
    {
        auto v = updateVertexList[i];
        updateValueList[i] = currSupport[v]-bcnt[v]-nonNativeSupport[v];
    }
    return numUpdates;
}


/*****************************************************************************
Peel active vertices and generate updates for 2-hop neighbors of deleted vertices
Used for fine-grained peeling within a partition
SERIAL
Inputs:
    1. partG -> csr matrix for edges incident on current partition
    1. G-> graph object
    2. labels -> vertices to be peeled but not deleted yet
    3. activeList -> vertices peeled in this round
    4. numActiveVertices -> number of active vertices
    4. isActive -> boolean vector mapping a vertex ID to its active status
    5. currSupport -> current support of vertices
    6. wedgeCnt -> 2D array for threads to store wedges while counting
Outputs:
    1. updateVertexList -> list of vertices whose support values are updated
    2. updateValueList -> corresponding values by which support should be reduced
Other args(they must be initialized to all "falses/zeros"):
    //can be replaced by sparseAdditiveSet from ligra
    1. isUpdated -> boolean vector that maps vertices to their "support updated" status
                    in the current peeling round
    2. peelCnts ->  store the running count of #butterflies deleted for vertices during peeling
    3. hop2Neighs -> vector to store 2-hop neighbors of active vertices. 
                    Passed as argument to avoid repeated allocation/deallocation
******************************************************************************/
intV return_updates_by_peeling_part_ser(myCSR &partG, Graph &G, std::vector<intV> &labels, std::vector<intV> &activeList, intV numActiveVertices, std::vector<uint8_t> &isActive, std::vector<intB> &currSupport, std::vector<intV> &updateVertexList, std::vector<intB> &updateValueList, std::vector<intV> &wedgeCnt, std::vector<uint8_t> &isUpdated, std::vector<intB> &peelCnts, std::vector<intV> &hop2Neighs)
{
    intV numUpdates = 0;
    std::vector<intV> &numW = wedgeCnt; 
    for (intV i=0; i<numActiveVertices; i++)
    {
        intV delV = activeList[i];
        for (intV j=partG.VI[delV]; j<partG.VI[delV] + partG.deg[delV]; j++)
        {
            intV neigh = partG.EI[j];
            for (intV k=partG.VI[neigh]; k<partG.VI[neigh] + partG.deg[neigh]; k++)
            {
                intV neighOfNeigh = partG.EI[k];
                if(isActive[neighOfNeigh] || G.is_deleted(neighOfNeigh))
                    continue;
                if (numW[neighOfNeigh]==0)
                    hop2Neighs.push_back(neighOfNeigh);
                numW[neighOfNeigh] = numW[neighOfNeigh] + 1;
            }
        }
        for (auto x:hop2Neighs)
        {
            if (numW[x] >= 2) 
            {
                intB butterflies = choose2<intB, intV>(numW[x]);
                //Can use same "isUpdated" array across threads
                //No partition touches vertices of other partitions
                if (!isUpdated[x])
                {
                    isUpdated[x] = 1;
                    if (numUpdates+1 > updateVertexList.size())
                        updateVertexList.push_back(x);
                    else
                        updateVertexList[numUpdates] = x;
                    numUpdates++;
                }
                peelCnts[x] += butterflies;
            }
            numW[x] = 0;
        }
        hop2Neighs.clear();
    }
    if (updateValueList.size() < numUpdates)
    {
        updateValueList.clear();
        updateValueList.resize(numUpdates);
    }
    for (intV i=0; i<numUpdates; i++)
    {
        intV vId = updateVertexList[i];
        updateValueList[i] = peelCnts[vId];
        isUpdated[vId] = 0;
        peelCnts[vId] = 0;
    }
    return numUpdates;
}

/*****************************************************************************
Update the deleted status of vertices peeled in current round
SERIAL
Inputs:
    1. G-> graph object
    2. activeList -> List of vertices peeled
    3. numActiveVertices -> number of active vertices
Outputs:
    1. isActive -> array that maps vertex IDs to their "active" status
******************************************************************************/
void delete_active_vertices_part_ser(Graph &G, std::vector<intV> &activeList, intV numActiveVertices, std::vector<uint8_t> &isActive)
{
    for (intV i=0; i<numActiveVertices; i++)
    {
        isActive[activeList[i]] = false;
        G.delete_vertex(activeList[i]);
    }
}

/*****************************************************************************
Peel the active vertices and generated count updates to their 2-hop neighbors
Choose either re-counting or peeling for update generation
This function is specific for peeling within a partition where other vertices
are ignored
SERIAL
Inputs:
    1. partG -> csr matrix for edges incident on current partition
    1. G-> graph object
    2. labels -> vertices to be peeled but not deleted yet
    3. activeList -> vertices peeled in this round
    4. numActiveVertices -> number of active vertices
    4. isActive -> boolean vector mapping a vertex ID to its active status
    5. supp -> support vector
    6. nonNativeSupp -> support from vertices belonging to higher partitions
    7. countComplexity -> work required to do a re-count
    8. peelWork -> per-vertex work for peeling
    9. wedgeCnt -> 2D array for threads to store wedges while counting
Outputs:
    1. updateVertexList -> list of vertices whose support values are updated
    2. updateValueList -> corresponding values by which support should be reduced
Other args(they must be initialized to all "falses/zeros"):
    //can be replaced by sparseAdditiveSet from ligra
    1. isUpdated -> boolean vector that maps vertices to their "support updated" status
                    in the current peeling round
    2. peelCnts ->  store the running count of #butterflies deleted for vertices during peeling
    3. hop2Neighs -> vector to store 2-hop neighbors of active vertices. 
                    Passed as argument to avoid repeated allocation/deallocation
******************************************************************************/
intV update_count_part_ser(myCSR &partG, Graph &G, std::vector<intV> &labels, std::vector<intV> &activeList, intV numActiveVertices, std::vector<uint8_t> &isActive, std::vector<intB> &supp, std::vector<intB> &nonNativeSupp, std::vector<intV> &updateVertexList, std::vector<intB> &updateSupportVal, intB countComplexity, std::vector<intE> &peelWork, std::vector<intV> &wedgeCnt, std::vector<uint8_t> &isUpdated, std::vector<intB> &peelCnts, std::vector<intV> &hop2Neighs)
{
    intB peelComplexity = 0;
    for (intV i=0; i<numActiveVertices; i++)
        peelComplexity += peelWork[activeList[i]];


    bool dontPeel = (countComplexity < peelComplexity);
    //dontPeel = false;
    if (dontPeel)
    {
        //printf("counting, cc=%lld, pc=%lld\n", countComplexity, peelComplexity);
        delete_active_vertices_part_ser(G, activeList, numActiveVertices, isActive);
        intV numUpdates = return_updates_by_counting_part_ser(partG, G, labels, activeList, isActive, supp, nonNativeSupp, updateVertexList, updateSupportVal, wedgeCnt);
        return numUpdates;
    }
    else
    {
        //printf("peeling, cc=%lld, pc=%lld\n", countComplexity, peelComplexity);
        intV numUpdates = return_updates_by_peeling_part_ser(partG, G, labels, activeList, numActiveVertices, isActive, supp, updateVertexList, updateSupportVal, wedgeCnt, isUpdated, peelCnts, hop2Neighs);
        delete_active_vertices_part_ser(G, activeList, numActiveVertices, isActive);
        return numUpdates;
    }
     
}





/*****************************************************************************
Peel all levels of a given partition using priority queue instead of bucketing
SERIAL
Arguments:
    1. partG -> csr matrix for edges incident on current partition
    1. G-> graph object (potentially with edges to other partitions deleted)
    2. countComplexity -> work required to re-count butterflies for this partition
    3. peelWork, countWork -> work per vertex to peel/count
    2. partId -> partition ID
    3. partVertices -> array of vertices in that partition
    4. vtxToIdx -> mapping of vertex IDs to their position in the partVertices array
                   needed for bucketing as bucketing requires contiguous IDs starting from 0
    4. tipVal -> (ip/op) support value of vertices
                 initialized with the initial value of all vertices when this partitions starts peeling
    4. lo, hi -> range of support values for this partition
    7. wedgeCnt -> 2D array for threads to store wedges while counting
    6. isActive -> boolean vector mapping a vertex ID to its active status
    8. isUpdated -> boolean vector indicates if the support of a vertex was updated
    9. peelCnts -> running count of #butterflies deleted for vertices during peeling
******************************************************************************/
void peel_partition_ser_q (myCSR &partG, Graph &G, intB countComplexity, std::vector<intE> &peelWork, std::vector<intE> &countWork, int partId, std::vector<intV> &partVertices, std::vector<intV>& vtxToIdx, std::vector<intB> &tipVal, std::vector<intB> &nonNativeSupp, intB lo, intB hi, std::vector<intV> &wedgeCnt, std::vector<uint8_t> &isActive, std::vector<uint8_t> &isUpdated, std::vector<intB> &peelCnts)
{ 
    intV nu = partVertices.size();


    intV finished = 0;

    int rounds = 0;
    intV maxPeeled = 0;
    intV avgPeeled = 0;

    std::vector<intV> updateVertexList;
    std::vector<intB> updateValueList; 
    std::vector<intV> activeList;
    activeList.reserve(nu);
    std::vector<intV> hop2Neighs;

    //init priority queue
    KHeap<intV, intB> queue(partVertices.size()); 
    for (intV i=0; i<partVertices.size(); i++)
        queue.update(i, tipVal[partVertices[i]]); 
    //printf("queue initialized\n");



    intB edgeDelThresh = ((intB)G.numE)*((intB)std::log2((double)G.numT)); //if more work than this is done, can clean up CSR
    intB peelWorkDone = 0;

    while((finished+1 < nu) && (!queue.empty()))
    {
        //subgraph size reduction
        if (peelWorkDone > edgeDelThresh)
        {
            delete_edges_csr(G, partG); 
            peelWorkDone = 0;
        }

        //find the min. k value 
        std::pair<intV, intB> kv = queue.top();
        intV v = partVertices[kv.first];
        intB k = kv.second;


        //find all vertices with tipvalue = k
        while(tipVal[v]==k)
        {
            activeList.push_back(partVertices[queue.pop()]);
            if (queue.empty()) break;
            kv = queue.top();
            v = partVertices[kv.first];
        }
        maxPeeled = std::max(maxPeeled, (intV)activeList.size());
        avgPeeled += activeList.size();
        if (queue.empty()) break;
        if (k>=(hi-1))
        {
            while(!queue.empty())
                tipVal[partVertices[queue.pop()]] = k;
        }
        if (queue.empty()) break;
        finished += activeList.size();

        for (auto x : activeList)
        {
            isActive[x] = 1;
            peelWorkDone += peelWork[x];
        }

        intV numUpdates = 0;
        if (k>0)
            numUpdates = update_count_part_ser(partG, G, partVertices, activeList, activeList.size(), isActive, tipVal, nonNativeSupp, updateVertexList, updateValueList, countComplexity, peelWork, wedgeCnt, isUpdated, peelCnts, hop2Neighs);

    
        activeList.clear();


        for (intV i=0; i<numUpdates; i++)
        {
            v = updateVertexList[i];
            if (G.is_deleted(v)) continue;
            tipVal[v] = std::max(k, tipVal[v] - updateValueList[i]);
            queue.update(vtxToIdx[v], tipVal[v]);
        }
        rounds++;
    }
    //printf("partition id = %d, # rounds = %d, max vertices peeled in a round = %u, avg peeled per round = %u\n", partId, rounds, maxPeeled, avgPeeled/rounds);
}
