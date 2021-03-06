#include "count.h"



//buffers/arrays for histogramming edges/edge workload for wing decomposition
std::vector<std::array<intB, locBuffSizeLarge>> thdBloomBuff;
std::vector<std::array<intE, locBuffSizeLarge>> thdEdgeBuff;
std::vector<std::vector<intE>> histCountPerThread;
std::vector<intE> histCountGlobal;
std::vector<intE> histAccGlobal;

std::vector<std::vector<intB>> histWorkPerThread;
std::vector<intB> histWorkGlobal;
std::vector<intB> histWorkAccGlobal;


//for static load balancing
std::vector<intB> workBloomSchedule;
std::vector<intB> accWorkBloomSchedule;
std::vector<intB> partBloomStart;



/*****************************************************************************
Re-count and generate updates for 2-hop neighbors of deleted vertices
Inputs:
    1. G-> graph object
    2. labels -> vertices not deleted yet
    3. activeList -> vertices peeled in this round
    4. isActive -> boolean vector mapping a vertex ID to its active status
    5. currSupport -> current support of vertices
    5. nonNativeSupport -> support from other vertices not included in G
    6. wedgeCnt -> 2D array for threads to store wedges while counting
Outputs:
    1. updateVertexList -> list of vertices whose support values are updated
    2. updateValueList -> corresponding values by which support should be reduced
******************************************************************************/
void return_updates_by_counting(Graph &G, std::vector<intV> &labels, std::vector<intV> &activeList, std::vector<uint8_t> &isActive, std::vector<intB> &currSupport, std::vector<intB> &nonNativeSupport, std::vector<intV> &updateVertexList, std::vector<intB> &updateValueList, std::vector<std::vector<intV>> &wedgeCnt)
{
    std::vector<intB> bcnt;
    count_per_vertex (G, labels, bcnt, wedgeCnt);

    std::vector<uint8_t> differs(labels.size());
    #pragma omp parallel for 
    for (intV i=0; i<labels.size(); i++)
    {
        auto v = labels[i];
        differs[i] = (((bcnt[v] + nonNativeSupport[v]) != currSupport[v]) && (!G.is_deleted(v)) && (!isActive[v])) ? 1 : 0; 
    }
    parallel_compact<intV, intV>(labels, differs, updateVertexList);
    updateValueList.resize(updateVertexList.size());
    #pragma omp parallel for 
    for (intV i=0; i<updateVertexList.size(); i++)
    {
        auto v = updateVertexList[i];
        updateValueList[i] = currSupport[v]-bcnt[v]-nonNativeSupport[v];
    }
}



/*****************************************************************************
Peel active vertices and generate updates for 2-hop neighbors of deleted vertices
Inputs:
    1. G-> graph object
    2. labels -> vertices to be peeled but not deleted yet
    3. activeList -> vertices peeled in this round
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
******************************************************************************/
void return_updates_by_peeling(Graph &G, std::vector<intV> &labels, std::vector<intV> &activeList, std::vector<uint8_t> &isActive, std::vector<intB> &currSupport, std::vector<intV> &updateVertexList, std::vector<intB> &updateValueList, std::vector<std::vector<intV>> &wedgeCnt, std::vector<uint8_t> &isUpdated, std::vector<intB> &peelCnts)
{
    std::vector<intV> updatesPerThread (NUM_THREADS, 0);
    std::vector<intV> offset (NUM_THREADS+1, 0);
    intV numActiveVertices = activeList.size();
    int numActiveThreads = std::min((unsigned int)(numActiveVertices>>1) + 1, NUM_THREADS);
    intV BS = ((numActiveVertices-1)/numActiveThreads + 1); BS = (BS > 5) ? 5 : BS;
    #pragma omp parallel num_threads(numActiveThreads) 
    {
        size_t tid = omp_get_thread_num();
        std::vector<intV> tmpVertexList;
        std::vector<intV> &numW = wedgeCnt[tid]; 
        std::vector<intV> hop2Neighs;
        hop2Neighs.reserve(8096);
        #pragma omp for schedule(dynamic, BS) 
        for (intV i=0; i<numActiveVertices; i++)
        {
            intV delV = activeList[i];
            intV deg;
            std::vector<intV> &neighList = G.get_neigh(delV, deg);
            for (intV j=0; j<deg; j++)
            {
                intV neigh = neighList[j];
                intV neighDeg;
                std::vector<intV> &neighOfNeighList = G.get_neigh(neigh, neighDeg);
                for (intV k=0; k<neighDeg; k++)
                {
                    intV neighOfNeigh = neighOfNeighList[k];
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
                    if (__sync_bool_compare_and_swap(&isUpdated[x], 0, 1))
                        tmpVertexList.push_back(x);
                    __sync_fetch_and_add(&peelCnts[x], butterflies); 
                }
                numW[x] = 0;
            }
            hop2Neighs.clear();
        }
        updatesPerThread[tid] = tmpVertexList.size(); 
        #pragma omp barrier
        #pragma omp single
        {
            serial_prefix_sum(offset, updatesPerThread); 
            updateVertexList.clear(); updateValueList.clear();
            updateVertexList.resize(offset[NUM_THREADS]);
            updateValueList.resize(offset[NUM_THREADS]);
        }
        #pragma omp barrier
        for (intV i=0; i<tmpVertexList.size(); i++)
        {
            intV vId = tmpVertexList[i];
            updateVertexList[offset[tid]+i] = vId;
            updateValueList[offset[tid]+i] = peelCnts[vId];
        }    
        #pragma omp barrier
        #pragma omp for
        for (intV i=0; i<offset[NUM_THREADS]; i++)
        {
            intV vId = updateVertexList[i];
            isUpdated[vId] = 0;
            peelCnts[vId] = 0;
        }
    }
}


/*****************************************************************************
Update the deleted status of vertices peeled in current round
Inputs:
    1. G-> graph object
    2. activeList -> List of vertices peeled
Outputs:
    1. isActive -> array that maps vertex IDs to their "active" status
******************************************************************************/
void delete_active_vertices(Graph &G, std::vector<intV> &activeList, std::vector<uint8_t> &isActive)
{
    intV numActiveVertices = activeList.size();
    #pragma omp parallel for 
    for (intV i=0; i<numActiveVertices; i++)
    {
        isActive[activeList[i]] = false;
        G.delete_vertex(activeList[i]);
    }
}


/*****************************************************************************
Peel the active vertices and generated count updates to their 2-hop neighbors
Choose either re-counting or peeling for update generation
Inputs:
    1. G-> graph object
    2. labels -> vertices to be peeled but not deleted yet
    3. activeList -> vertices peeled in this round
    4. isActive -> boolean vector mapping a vertex ID to its active status
    5. supp -> current support of vertices
    5. nonNativeSupport -> support from other vertices not included in G
    6. countComplexity -> work required to do a re-count
    7. peelWork -> per-vertex work for peeling
    6. wedgeCnt -> 2D array for threads to store wedges while counting
Outputs:
    1. updateVertexList -> list of vertices whose support values are updated
    2. updateValueList -> corresponding values by which support should be reduced
Other args(they must be initialized to all "falses/zeros"):
    //can be replaced by sparseAdditiveSet from ligra
    1. isUpdated -> boolean vector that maps vertices to their "support updated" status
                    in the current peeling round
    2. peelCnts ->  store the running count of #butterflies deleted for vertices during peeling
******************************************************************************/
int update_count (Graph &G, std::vector<intV> &labels, std::vector<intV> &activeList, std::vector<uint8_t> &isActive, std::vector<intB> &supp, std::vector<intB> &nonNativeSupport, std::vector<intV> &updateVertexList, std::vector<intB> &updateSupportVal, intB countComplexity, std::vector<intE> &peelWork, std::vector<std::vector<intV>> &wedgeCnt, std::vector<uint8_t> &isUpdated, std::vector<intB> &peelCnts)
{
    intB peelComplexity = 0;
    #pragma omp parallel for reduction(+:peelComplexity)
    for (intV i=0; i<activeList.size(); i++)
        peelComplexity += peelWork[activeList[i]];


    bool dontPeel = (countComplexity < peelComplexity);
    //dontPeel = false;
    if (dontPeel)
    {
        delete_active_vertices(G, activeList, isActive);
        return_updates_by_counting(G, labels, activeList, isActive, supp, nonNativeSupport, updateVertexList, updateSupportVal, wedgeCnt);
        return 0;
    }
    else
    {
        return_updates_by_peeling(G, labels, activeList, isActive, supp, updateVertexList, updateSupportVal, wedgeCnt, isUpdated, peelCnts);
        delete_active_vertices(G, activeList, isActive);
        return 1;
    }
     
}


/*****************************************************************************
Construct (in parallel) a list of vertices whose support lies between 'lo' and 'hi'
Inputs:
    1. G-> graph object
    2. candidates -> vector of potential vertices that can be activated
    3. lo, hi -> range of support to be activated
    4. supp -> current support of vertices
Outputs:
    1. activeList -> list of active vertices
    2. isActive -> boolean vector mapping a vertex ID to its active status
******************************************************************************/
void construct_active_list (Graph &G, std::vector<intV> &candidates, intB lo, intB hi, std::vector<intV> &activeList, std::vector<uint8_t> &isActive, std::vector<intB> &supp)
{
    #pragma omp parallel for 
    for (int i=0; i<candidates.size(); i++)
    {
        auto v = candidates[i];
        if((supp[v]<hi) && (supp[v]>=lo) && (!G.is_deleted(v)))
        {
            supp[v] = lo;
            isActive[v] = 1;
        }
    }
    parallel_compact_kv<intV, intV>(candidates, isActive, activeList);
}


/*****************************************************************************
Peel vertices whose support is in the given range and update support of other vertices
Arguments:
    1. G-> graph object
    2. vertices -> candidate (remaining) vertices on the side being peeled
    3. lo, hi -> range of support values to be peeled
    4. isActive -> boolean vector mapping a vertex ID to its "active" status
    5. supp -> support vector
    5. nonNativeSupport -> support from other vertices not included in G
    6. countComplexity -> work required to do a re-count
    7. peelWork -> per-vertex work for peeling
    8. wedgeCnt -> 2D array for threads to store wedges while counting
    9. isUpdated -> boolean vector that maps vertices to their "support updated" status
                    in the current peeling round
    10. peelCnts ->  store the running count of #butterflies deleted for vertices during peeling
******************************************************************************/
intV peel_range (Graph &G, std::vector<intV> &vertices, intB lo, intB hi, std::vector<uint8_t> &isActive, std::vector<intB> &supp, std::vector<intB> &nonNativeSupport, intB countComplexity, std::vector<intE> &peelWork, std::vector<std::vector<intV>> &wedgeCnt, std::vector<uint8_t> &isUpdated, std::vector<intB> &peelCnts) 
{
    intV numDeleted = 0;
    std::vector<intV> activeList;
    for (auto x : vertices)
    {
        assert((supp[x] >= lo) || G.is_deleted(x));
    }
    construct_active_list(G, vertices, lo, hi, activeList, isActive, supp);
    numDeleted += activeList.size();

    //iteratively delete all vertices with tip values in this range//
    ////////////////////////////////////////////////////////////////
    std::vector<intV> updateVertexList;
    std::vector<intB> updateSupportVal;

    intB edgeDelThresh = ((intB)G.numE*((intB)std::log2(double(G.numV))));
    intB peelWorkDone = 0;

    intV numRounds = 0;
    intV numPeeled = 0;
    while(activeList.size() > 0)
    {
        //for (auto x:activeList)
        //    printf("deleting %u with support %llu\n", x, supp[x]);
        numPeeled += update_count(G, vertices, activeList, isActive, supp, nonNativeSupport, updateVertexList, updateSupportVal, countComplexity, peelWork, wedgeCnt, isUpdated, peelCnts);
        intV numUpdates = updateVertexList.size();
        #pragma omp parallel for 
        for (intV i=0; i<numUpdates; i++)
        {
            intV v = updateVertexList[i];
            intB updateVal = std::min(updateSupportVal[i], supp[v]-lo);
            supp[v] -= updateVal;
        }
        activeList.clear();
        construct_active_list(G, updateVertexList, lo, hi, activeList, isActive, supp);
        numDeleted += activeList.size();
        numRounds++;
        updateVertexList.clear();
        updateSupportVal.clear();
    }
    //printf("number of rounds required = %d, peeled = %d, counted = %d\n", numRounds, numPeeled, numRounds-numPeeled);
    return numDeleted;
}


/*****************************************************************************
Remove deleted vertices from the candidate list and return peeling complexity 
of remaining vertices
Arguments:
    1. G -> graph object
    2. vertices -> current vertex list (will be updated)
    3. peelComplexity -> peeling work required for each vertex (vector)
    4. keep -> helper boolean vector to be used in parallel compaction
******************************************************************************/
intB remove_deleted_vertices(Graph &G, std::vector<intV> &vertices, std::vector<intE> &peelComplexity, std::vector<uint8_t>& keep)
{
    keep.resize(vertices.size());
    intB remPeelComplexity = 0;
    #pragma omp parallel for reduction (+:remPeelComplexity) 
    for (intV i=0; i<vertices.size(); i++)
    {
        keep[i] = (G.is_deleted(vertices[i])) ? 0 : 1;
        remPeelComplexity += (keep[i]) ? peelComplexity[vertices[i]] : 0;
    }
    parallel_compact_in_place<intV, intV>(vertices, keep);
    return remPeelComplexity;
}

/*****************************************************************************
//overloaded function defintion
//cleans "vertices" vector and also creates a list of deleted vertices
//Additional argument - "delVertices" vector
******************************************************************************/
intB remove_deleted_vertices(Graph &G, std::vector<intV> &vertices, std::vector<intE> &peelComplexity, std::vector<uint8_t>& keep, std::vector<intV> &delVertices)
{
    keep.resize(vertices.size());
    intB remPeelComplexity = 0;
    #pragma omp parallel for reduction (+:remPeelComplexity) 
    for (intV i=0; i<vertices.size(); i++)
    {
        keep[i] = (G.is_deleted(vertices[i])) ? 0 : 1;
        remPeelComplexity += (keep[i]) ? peelComplexity[vertices[i]] : 0;
    }
    parallel_compact_in_place<intV, intV>(vertices, keep, delVertices);
    return remPeelComplexity;
}



/*****************************************************************************
2-approximate tip-decomposition. Peel a range of support values that doubles
every round
Arguments:
    1. G -> graph object
    2. tipVal -> half approximation of tip values of vertices
                 must be initialized with the per-vertex butterfly counts
    3. peelSide -> 0 implies peeling vertices in U, 1 means V  
    4. wedgeCnt -> 2D helper array for threads to store wedges
******************************************************************************/
/*
void approx_tip_decomposition(Graph &G, std::vector<intB> &tipVal, int peelSide, std::vector<std::vector<intV>> &wedgeCnt)
{
    std::vector<intV> vertices;
    G.get_labels(vertices, peelSide);
    std::vector<uint8_t> keep;
    //std::vector<uint8_t> keep(vertices.size());
    printf("number of vertices to peel = %u\n", vertices.size());
    //printf("vertices are");
    //print_list_horizontal(vertices);
    
    std::vector<intE> countWork;
    std::vector<intE> peelWork;
    printf("estimating workloads\n");
    intB totalCountComplexity = estimate_total_workload(G, countWork, peelWork);
    
    std::vector<uint8_t> isActive (G.numT);
    std::vector<uint8_t> isUpdated(G.numT);
    std::vector<intB> peelCnts(G.numT);
    std::vector<intB> nonNativeSupport(G.numT);
    #pragma omp parallel for
    for (intV i=0; i<G.numT; i++)
    {
        isActive[i] = 0;
        isUpdated[i] = 0;
        peelCnts[i] = 0;
        nonNativeSupport[i] = 0;
    }

    intB lo = 0;
    intB range = 1;
    intV numDeleted = 0;
    intV targetDeletion = (peelSide) ? G.numV : G.numU;
    printf("starting decopmosition\n");
    while(numDeleted < targetDeletion)
    {
        numDeleted = numDeleted + peel_range(G, vertices, lo, lo+range, isActive, tipVal, nonNativeSupport, totalCountComplexity, peelWork, wedgeCnt, isUpdated, peelCnts);
        //update range
        lo = lo + range;
        range = range*2;

        intB remPeelComplexity = remove_deleted_vertices(G, vertices, peelWork, keep);
    }
}
*/


/*****************************************************************************
Find the target range to create partition with desired peeling complexity
Arguments:
    1. vertices -> candidate vertices
    2. tipVal -> support value of vertices
    3. targetPeelComplexity -> desired amount of work required to peel the partition
    4. lowerBound -> lowest tip value
    5. peelWork -> work required to peel the vertices
******************************************************************************/
std::tuple<intB, intV, intV> find_range (std::vector<intV> &vertices, std::vector<intB>&tipVal, intB targetPeelComplexity, intB lowerBound, std::vector<intE> &peelWork)
{
    parallel_sort_kv_increasing<intV, intB>(vertices, tipVal);  //sort vertices on their current support

    //compute workload for each bucket - map, prefix sum, scatter
    //find bucket id for each vertex using map and prefix sum
    //scatter with atomic add to compute workload for the buckets
    std::vector<uint8_t> suppIsUniq(vertices.size()); 
    suppIsUniq[suppIsUniq.size()-1] = 1;
    #pragma omp parallel for 
    for (intV i=0; i<vertices.size()-1; i++)
        suppIsUniq[i] = (tipVal[vertices[i]]==tipVal[vertices[i+1]]) ? 0 : 1;

    std::vector<intV> wrOffset;
    parallel_prefix_sum(wrOffset, suppIsUniq);
    intV numUniqSuppVals = wrOffset.back(); //last element in the offset vector
    std::vector<intB> workPerSuppVal(numUniqSuppVals); //work to peel all vertices in a given bucket
    std::vector<intB> suppVal(numUniqSuppVals); //support value corresponding to the individual buckets
    #pragma omp parallel 
    {
        #pragma omp for
        for (intV i=0; i<numUniqSuppVals; i++)
            workPerSuppVal[i] = 0;
        #pragma omp barrier
        #pragma omp for
        for (intV i=0; i<vertices.size(); i++)
        {
            intV v = vertices[i];
            intB work = peelWork[v];
            suppVal[wrOffset[i]] = tipVal[v];
            __sync_fetch_and_add(&workPerSuppVal[wrOffset[i]], work);
        }
    }


    //none of the vertices with support < lo should've survived
    assert(suppVal[0] >= lowerBound);


    //prefix sum to compute work required to peel all vertices till a particular bucket 
    parallel_prefix_sum_inclusive(workPerSuppVal, workPerSuppVal);


    //find the first bucket with work just lower than the target value
    intV tgtBktId = std::lower_bound(workPerSuppVal.begin(), workPerSuppVal.end(), targetPeelComplexity) - workPerSuppVal.begin();

    intB hi = std::max(suppVal[tgtBktId], suppVal[0]+1); //hi should be greater than the support of the first bucket to ensure non-zero vertex peeling

    return std::make_tuple(hi, tgtBktId, numUniqSuppVals);
}


/*****************************************************************************
Coarse-grained decomposition with (targetted) equal workload partitions.
Arguments:
    1. G -> graph object
    2. tipVal -> (output) support of vertices when their partition begins peeling
                 must be initialized with the per-vertex butterfly counts
    3. peelSide -> 0 implies peeling vertices in U, 1 means V  
    4. wedgeCnt -> 2D helper array for threads to store wedges
    5. numParts -> number of partitions to create; final partitions may be smaller
    6. partTipVals -> output vector containing support ranges of the partitions
    7. partVertices -> 2D array to store vertices for each partition
    8. partPeelWork -> work done to peel the entire partition (considering no re-counting)
******************************************************************************/
int create_balanced_partitions(Graph &G, std::vector<intB> &tipVal, int peelSide, std::vector<std::vector<intV>> &wedgeCnt, int numParts, std::vector<std::pair<intB, intB>> &partTipVals, std::vector<std::vector<intV>> &partVertices, std::vector<intB> &partPeelWork)
{
    std::vector<intV> vertices;
    G.get_labels(vertices, peelSide);
    intV targetDeletion = vertices.size();

    std::vector<uint8_t> keep;
    std::vector<uint8_t> isActive(G.numT);

    std::vector<uint8_t> isUpdated(G.numT);
    std::vector<intB> peelCnts(G.numT);
    std::vector<intB> nonNativeSupport(G.numT);

    #pragma omp parallel for
    for (intV i=0; i<G.numT; i++)
    {
        isActive[i] = 0;
        isUpdated[i] = 0;
        peelCnts[i] = 0;
        nonNativeSupport[i] = 0;
    }
     
    std::vector<intE> countWork; //work required per vertex to count
    std::vector<intE> peelWork; //work required per vertex to peel, 2-hop neighborhood size
    
    intB totalCountComplexity = estimate_total_workload(G, countWork, peelWork);

    partTipVals.resize(numParts);
    partPeelWork.resize(numParts);
    std::vector<intV> verticesPerPart (numParts);

    std::vector<intB> partTipValInit; //initial values of vertices when their corresponding partitions starts peeling
    parallel_vec_copy(partTipValInit, tipVal);
     

    intB totalPeelComplexity = 0;
    #pragma omp parallel for reduction(+:totalPeelComplexity) 
    for (intV i=0; i<vertices.size(); i++)
        totalPeelComplexity += peelWork[vertices[i]];
    intB avgPeelComplexityRequired = totalPeelComplexity/numParts;
    printf("total peel complexity = %lld, count complexity = %lld\n", totalPeelComplexity, totalCountComplexity);

    intB remPeelComplexity = totalPeelComplexity;
    intB lo = 0;

    int numPartsCreated = 0;
    int numPartsPerThread = numParts/NUM_THREADS;
    intV numDeleted = 0;
    
    //if lot of work done, remove deleted edges to speedup further processing
    intB edgeDelThresh = ((intB)G.numE)*((intB)std::log2((double)G.numV));
    intB peelWorkDone = 0;
    //helps in adapting the targetWorkComplexity if the partitions become too heavy
    double scaleFactor = 1.0; 

    //till there is something to peel or only last partition remains
    while((remPeelComplexity > 0) && (numPartsCreated < numParts-1) && (numDeleted < targetDeletion)) 
    {
        if (peelWorkDone > edgeDelThresh)
        {
            G.delete_edges();
            peelWorkDone = 0;
        }
        double bktPeelStart = omp_get_wtime();
        intB targetPeelComplexity = (intB)((scaleFactor*(double)remPeelComplexity)/(numParts-numPartsCreated)); //figure out target complexity to cover 
        intB desiredPeelComplexity = remPeelComplexity/(numParts-numPartsCreated);
        intB hi; intV tgtBktId, numUniqSuppVals;
        std::tie(hi, tgtBktId, numUniqSuppVals) = find_range(vertices, tipVal, targetPeelComplexity, lo, peelWork);

        //peel the range
        verticesPerPart[numPartsCreated] = peel_range(G, vertices, lo, hi, isActive, tipVal, nonNativeSupport, totalCountComplexity, peelWork, wedgeCnt, isUpdated, peelCnts);

        //logistics, track # deleted vertices, record range of the partition
        numDeleted += verticesPerPart[numPartsCreated];
        partTipVals[numPartsCreated] = std::make_pair(lo, hi); 

        intB prevRemPeelComplexity = remPeelComplexity;
        std::vector<intV> delVertices;
        remPeelComplexity = remove_deleted_vertices(G, vertices, peelWork, keep, delVertices);
        partPeelWork[numPartsCreated] = prevRemPeelComplexity - remPeelComplexity;
        peelWorkDone += partPeelWork[numPartsCreated];

        double bktPeelEnd = omp_get_wtime();
#ifdef DEBUG
        printf("partition id = %d, time taken = %lf, vertices deleted = %u, range from %lld to %lld, desired complexity = %lld, target complexity = %lld, actual work done = %lld\n", numPartsCreated, (bktPeelEnd-bktPeelStart)*1000, verticesPerPart[numPartsCreated], lo, hi, desiredPeelComplexity, targetPeelComplexity, partPeelWork[numPartsCreated]);
#endif


        //adapt, if too much work in this bucket, make targets smaller for the next partition
        scaleFactor = std::min(((double)targetPeelComplexity)/((double)partPeelWork[numPartsCreated]), 1.0);

        partVertices.push_back(delVertices);
        numPartsCreated++;
        parallel_vec_elems_copy(partTipValInit, tipVal, vertices);


        //prep for next partition creation
        lo = hi;
        
    }
    
    intV remVertices = vertices.size();
    intB maxRemSupp = 0;
    //put anything remaining in the last partition
    if (remVertices > 0)
    {
        partPeelWork[numPartsCreated] = remPeelComplexity;
        partVertices.push_back(vertices);
        #pragma omp parallel for reduction(max:maxRemSupp) 
        for (intV i=0; i<remVertices; i++)
        {
            maxRemSupp = std::max(tipVal[vertices[i]], maxRemSupp);
            tipVal[vertices[i]] = lo;
        }
        partTipVals[numPartsCreated++] = std::make_pair(lo, maxRemSupp+1); 
    }
    tipVal.swap(partTipValInit);

    #pragma omp parallel for
    for (intV i=0; i<G.numU; i++)
        G.restore_vertex(G.uLabels[i]);
    G.restore_edges();
    G.sort_adj();

    return numPartsCreated;
} 


/*****************************************************************************
Print Coarse-grained decomposition details into a binary file
Arguments:
    1. G -> graph object
    2. tipVal -> (output) support of vertices when their partition begins peeling
                 must be initialized with the per-vertex butterfly counts
    5. numParts -> number of partitions to create; final partitions may be smaller
    6. partTipVals -> output vector containing support ranges of the partitions
    7. partVertices -> 2D array to store vertices for each partition
    8. partPeelWork -> work done to peel the entire partition (considering no re-counting)
******************************************************************************/
void print_partitioning_details (std::string &filename, Graph &G, std::vector<intB> &tipVal, int numParts, std::vector<std::pair<intB, intB>> &partTipVals, std::vector<std::vector<intV>> &partVertices, std::vector<intB> &partPeelWork)
{ 
    std::vector<intV> vOut;
    std::vector<int> pOut;
    std::vector<intB> tOut;
    std::vector<intB> pRangeLo;
    std::vector<intB> pRangeHi;
    intV numVOut = 0;
    for (int i=0; i<numParts; i++)
    {
        pRangeLo.push_back(partTipVals[i].first);
        pRangeHi.push_back(partTipVals[i].second);
        for (intV j=0; j<partVertices[i].size(); j++)
        {
            numVOut++;
            vOut.push_back(partVertices[i][j]);
            pOut.push_back(i);
            tOut.push_back(tipVal[partVertices[i][j]]);    
        }
    }
    assert(G.numU==numVOut);
    
    FILE* fpart = fopen("part_details.bin", "w");
    fwrite(&numParts, sizeof(int), 1, fpart);
    fwrite(&partPeelWork[0], sizeof(intB), numParts, fpart);
    fwrite(&pRangeLo[0], sizeof(intB), numParts, fpart);
    fwrite(&pRangeHi[0], sizeof(intB), numParts, fpart); 
    fwrite(&vOut[0], sizeof(intV), numVOut, fpart);
    fwrite(&pOut[0], sizeof(int), numVOut, fpart);
    fwrite(&tOut[0], sizeof(intB), numVOut, fpart);
     
    fclose (fpart);
}
     
/*****************************************************************************
Read Coarse-grained decomposition details from a binary file
Arguments:
    1. G -> graph object
    2. tipVal -> (output) support of vertices when their partition begins peeling
                 must be initialized with the per-vertex butterfly counts
    5. numParts -> number of partitions to create; final partitions may be smaller
    6. partTipVals -> output vector containing support ranges of the partitions
    7. partVertices -> 2D array to store vertices for each partition
    8. partPeelWork -> work done to peel the entire partition (considering no re-counting)
******************************************************************************/
void read_partitioning_details (std::string &filename, Graph &G, std::vector<intB> &tipVal, int &numParts, std::vector<std::pair<intB, intB>> &partTipVals, std::vector<std::vector<intV>> &partVertices, std::vector<intB> &partPeelWork)
{ 
    FILE* fcd = fopen(filename.c_str(), "rb");
    if (fcd==NULL)
    {
        fputs("file error\n", stderr);
        exit(EXIT_FAILURE);
    }
    printf("file opened\n");
    int np;
    fread(&np, sizeof(int), 1, fcd);
    numParts = np;
    printf("number of partitions = %d\n", numParts);
    partPeelWork.resize(numParts);
    partTipVals.resize(numParts);
    partVertices.resize(numParts);
    fread(&partPeelWork[0], sizeof(intB), numParts, fcd);
    printf("read peel work\n");
    for (int i=0; i<numParts; i++)
        fread(&partTipVals[i].first, sizeof(intB), 1, fcd);
    for (int i=0; i<numParts; i++)
        fread(&partTipVals[i].second, sizeof(intB), 1, fcd);
    printf("read partition ranges\n");
    std::vector<intV> vIn (G.numU);
    std::vector<intV> pIn (G.numU);
    std::vector<intB> tIn (G.numU);
    intV bytesRead = fread(&vIn[0], sizeof(intV), G.numU, fcd);
    printf("number of vertices read = %u\n", bytesRead);
    assert(bytesRead==G.numU);
    bytesRead = fread(&pIn[0], sizeof(int), G.numU, fcd);
    assert(bytesRead==G.numU);
    printf("read partition map\n");
    bytesRead = fread(&tIn[0], sizeof(intB), G.numU, fcd);
    assert(bytesRead==G.numU);
    printf("read tipvals\n");
    for (intV i=0; i<G.numU; i++)
    {
        intV v = vIn[i];
        tipVal[v] = tIn[i];
        partVertices[pIn[i]].push_back(v);
    }
}





/*****************************************************************************
Compute upper bound on the maximum wing number
Inputs:
    1. eIds -> edge indices sorted on current support 
    2. tipVal -> vector of current support of edges
    3. nEdgesRem -> number of not yet peeled edges in eIds vector
Outputs:
    returns an upper bound on max wing number
******************************************************************************/
intE find_upper_bound_wing(std::vector<intE> &eIds, std::vector<intE> &tipVal, intE nEdgesRem)
{
    parallel_unstable_sort_kv_increasing(eIds, tipVal);
    intE ub = 0;
    if (nEdgesRem > 10*NUM_THREADS)
    {
        intE BS = (nEdgesRem-1)/NUM_THREADS + 1;
        #pragma omp parallel num_threads(NUM_THREADS) reduction (max:ub)
        {
            unsigned tid = omp_get_thread_num();
            intE start = tid*BS;
            intE end = std::min(nEdgesRem, start+BS);
            for (intE i = end-1; i>=start; i--)
            {
                intE currSupp = tipVal[eIds[i]];
                intE numEdgesWithHigherSupp = nEdgesRem - i;
                if (numEdgesWithHigherSupp >= currSupp)
                {
                    ub = std::max(ub, currSupp);
                    break;
                }
                else
                    ub = std::max(std::min(currSupp, numEdgesWithHigherSupp), ub);
            }
        }
    }
    else
    {
        for (intE i=nEdgesRem-1; i>=0; i--)
        {
            intE currSupp = tipVal[eIds[i]];
            intE numEdgesWithHigherSupp = nEdgesRem - i;
            if (numEdgesWithHigherSupp >= currSupp)
            {
                ub = std::max(ub, currSupp);
                break;
            }
            else
                ub = std::max(std::min(currSupp, numEdgesWithHigherSupp), ub);
        }
    }
    return ub;
}


/*****************************************************************************
Re-compute upper bound on the maximum wing number and populate histograms
for range determination
Inputs:
    1. eIds -> edge indices sorted on current support 
    2. tipVal -> vector of current support of edges
    3. nEdgesRem -> number of not yet peeled edges in eIds vector
    4. minTipVal -> lower bound based on edges peeled so far
    5. currUb -> previous upper bound
Outputs:
    returns an upper bound on max wing number
******************************************************************************/
intE update_upper_bound_wing(std::vector<intE> &eIds, intE nEdgesRem, std::vector<intE> &tipVal, intE minTipVal, intE currUb)
{
    intE range = currUb - minTipVal + 1;
    if (histCountGlobal.size() < range)
        histCountGlobal.resize(range);
    if (histAccGlobal.size() < range)
        histAccGlobal.resize(range);
    if (histCountPerThread.size() < NUM_THREADS)
        histCountPerThread.resize(NUM_THREADS);

    intE edgesPerThread = (nEdgesRem-1)/NUM_THREADS + 1;
    intE BS = (range-1)/NUM_THREADS + 1;
    intE newUb = minTipVal;
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        intE tid = omp_get_thread_num();

        #pragma omp for
        for (intE i=0; i<range; i++)
            histCountGlobal[i] = 0;

        std::vector<intE> &locHistCount = histCountPerThread[tid];
        if (locHistCount.size() < range)
            locHistCount.resize(range);
        for (intE i=0; i<range; i++)
            locHistCount[i] = 0;

        #pragma omp for
        for (intE i=0; i<nEdgesRem; i++)
        {
            intE val = std::min(tipVal[eIds[i]], currUb) - minTipVal;
            assert(val < range);
            //reverse for suffix sum
            locHistCount[range-val-1]++; 
        }

        intE ptr = rand()%range;
        for (intE i=0; i<range; i++)
        {
            intE idx = (ptr+i)%range;
            __sync_fetch_and_add(&histCountGlobal[idx], locHistCount[idx]);
        }

        #pragma omp barrier
        //PREFIX SUM
        intE start = BS*tid;
        intE end = std::min((intE)(start+BS), range);
        

        if (range > NUM_THREADS*10)
        {
            histAccGlobal[start] = histCountGlobal[start];
    
            for (intE i=start+1; i<end; i++)
                histAccGlobal[i] = histAccGlobal[i-1] + histCountGlobal[i];
    
            #pragma omp barrier
            #pragma omp single
            {
                for (size_t i=1; i<NUM_THREADS; i++)
                {
                    intE prevEnd = BS*i; if (prevEnd >= range) continue; 
                    intE tend = std::min(prevEnd + BS, range);
                    histAccGlobal[tend-1] += histAccGlobal[prevEnd-1];
                }
            }
            #pragma omp barrier
    
            if (tid > 0)
            {
                intB blockScan = histAccGlobal[start-1];
                for (intE i=start; i<end-1; i++)
                    histAccGlobal[i] += blockScan;
            }
            intE locMax = 0;
            if (end > start)
            {
                for (intE i=start; i<end; i++)
                {
                    intE supp = (range - i - 1)  + minTipVal;
                    if (histAccGlobal[i] >= supp)
                    {
                        locMax = supp;
                        break;
                    }
                }
                #pragma omp critical
                {
                    if (locMax > newUb)
                        newUb = locMax;
                }
            }
        }
        else
        {
            #pragma omp single
            {
                histAccGlobal[0] = histCountGlobal[0];
                for (intE i=1; i<range; i++)
                    histAccGlobal[i] = histAccGlobal[i-1] + histCountGlobal[i];
                for (intE i=0; i<range; i++)
                {
                    intE supp = (range - i - 1) + minTipVal;
                    if (histAccGlobal[i] >= supp)
                    {
                        newUb = supp;
                        break;
                    } 
                }
                assert(histAccGlobal[range-1]==nEdgesRem);
            }
        }
    }
    return newUb;
}




/*****************************************************************************
Compute upper bound for the range of a partition
Inputs:
    1. eIds -> edge indices sorted on current support 
    2. tipVal -> vector of current support of edges
    3. nPartsRem -> number of partitions remaining to be created
    4. nEdgesRem -> number of not yet peeled edges in eIds vector
    5. scaling -> scaling factor to apply
    6. tipMin -> lower bound based on partition's wing number range
    7. tipMax -> recently updated upper bound
    8. oldMax -> previous upper bound
Outputs:
    1. range upper bound for the partition
    2. estimated work value for the partition based on current edge support
******************************************************************************/
std::tuple<intE, intB> find_upper_bound_part(std::vector<intE> &eIds, std::vector<intE> &tipVal, intE nPartsRem, intE nEdgesRem, double scaling, intE tipMin, intE tipMax, intE oldMax)
{
    intE range = tipMax - tipMin + 1;
    intE oldRange = oldMax - tipMin + 1;

    std::vector<intB> &workPerSupp = histWorkGlobal; if (workPerSupp.size() < range) workPerSupp.resize(range);
    std::vector<intB> &accWork = histWorkAccGlobal; if (accWork.size() < range) accWork.resize(range);


    intE BS = (range-1)/NUM_THREADS + 1;

    intE newMaxCount = histCountGlobal[oldMax-tipMax];
    #pragma omp parallel num_threads (NUM_THREADS)
    {
        unsigned tid = omp_get_thread_num();
        //count edges with higher support than new max into the bin of new max value
        //histCountGlobal[i] is the no. of edges with support oldMax - (tipMin + i)
        #pragma omp for reduction (+:newMaxCount)
        for (intE i=0; i<oldMax-tipMax; i++)
            newMaxCount += histCountGlobal[i];
        #pragma omp single
        {
            histCountGlobal[oldMax-tipMax] = newMaxCount;
        }

        #pragma omp for
        for (intE i=0; i<range; i++)
        {
            intB val = i + tipMin;
            intE countIdx = oldMax - val; 
            intB edgeCnt = histCountGlobal[countIdx];
            workPerSupp[i] = edgeCnt*val; 
        }

        //PREFIX SUM counts to compute write offsets for each support value
        if (range < 10*NUM_THREADS)
        {
            #pragma omp single
            {
                accWork[0] = workPerSupp[0];
                for (intE i=1; i<range; i++) accWork[i] = accWork[i-1]+workPerSupp[i];
            }
        }
        else
        {
            intE start = BS*tid;
            intE end = std::min((intE)(start+BS), range);
        
            if (start < range) accWork[start] = workPerSupp[start];

            for (intE i=start+1; i<end; i++)
                accWork[i] = accWork[i-1] + workPerSupp[i];

            #pragma omp barrier
            #pragma omp single
            {
                for (size_t i=1; i<NUM_THREADS; i++)
                {
                    intE prevEnd = BS*i; if (prevEnd >= range) continue; 
                    intE tend = std::min(prevEnd + BS, range);
                    accWork[tend-1] += accWork[prevEnd-1];
                }
            }
            #pragma omp barrier

            if (tid > 0)
            {
                intB blockScan = accWork[start-1];
                for (intE i=start; i<end-1; i++)
                    accWork[i] += blockScan;
            }
        }
    }

    
    //dynamic average with scaling
    intB tgtWorkVal = (long long int)(double(accWork[range-1]/nPartsRem)*scaling);
    //find smallest support value at which work is greater than average
    intE partUB = (std::lower_bound(accWork.begin(), accWork.begin()+range, tgtWorkVal) - accWork.begin()) + tipMin + 1; 
    tgtWorkVal = accWork[std::min(partUB-tipMin-1, range-1)];
    return std::make_tuple(partUB, tgtWorkVal);
             
}


//compute scaling factor
double compute_scale(std::vector<intE> &partEdges, std::vector<intE> &initSupp, intE maxSupp, intB tgtWork)
{
    intB actualWork = 0;
    #pragma omp parallel for num_threads(NUM_THREADS) reduction (+:actualWork)
    for (intE i=0; i<partEdges.size(); i++)
        actualWork += std::min(initSupp[partEdges[i]], maxSupp);
    if (actualWork == 0) return 1.0;
    assert(actualWork >= tgtWork);
    double scaling = ((double)tgtWork)/((double)actualWork);
    return scaling;
}



//find active edges for the first peeling iteration of a partition
void find_active_edges(std::vector<intE> &eIds, std::vector<intE> &tipVal, std::vector<uint8_t> &isActive, intE nEdgesRem, intE kLo, intE kHi, std::vector<intE> &activeEdges, intE &activeEdgePtr)
{
    if (thdEdgeBuff.size() < NUM_THREADS) thdEdgeBuff.resize(NUM_THREADS);
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        size_t tid = omp_get_thread_num();
        std::array<intE, locBuffSizeLarge> &locBuff = thdEdgeBuff[tid]; unsigned locBuffPtr = 0;
        #pragma omp for
        for (intE i=0; i<nEdgesRem; i++)
        {
            intE e = eIds[i];
            assert(tipVal[e] >= kLo);
            if (tipVal[e] < kHi)
            {
                locBuff[locBuffPtr++] = e;
                locBuffPtr = updateGlobalQueue(locBuffPtr, locBuffSizeLarge, activeEdgePtr, locBuff, activeEdges);   
                isActive[e] = true;
            }
        }
        if (locBuffPtr > 0)
            locBuffPtr = updateGlobalQueue(locBuffPtr, locBuffPtr, activeEdgePtr, locBuff, activeEdges);
    }
    
} 





/*****************************************************************************
Update support of edges in a peeling iteration
Inputs:
    1. BEG -> BE-Index
    2. tipVal -> vector of support of edges
    3. kLo, kHi -> partition range
    4. activeEdges, activeEdgePtr, activeEdgeStartOffset -> set of edges to Peel
    5. isActive -> boolean array to indicate if an edge is active
    6. isPeeled -> boolean array to indicate if an edge is already peeled
Outputs:
    1. updated edge supports
    2. updated list of active edges 
    3. returns a pointer to indicate the newly added active edges in activeEdges[] array 
Arguments:
    1. bloomUpdates -> vector to accumulate updates at blooms 
    2. activeBlooms -> array to store blooms with non-zero updates
******************************************************************************/
intE update_edge_supp(BEGraphLoMem& BEG, std::vector<intE> &tipVal, intE kLo, intE kHi, std::vector<intE> &activeEdges, intE activeEdgePtr, intE activeEdgeStartOffset, std::vector<intE> &bloomUpdates, std::vector<intB> &activeBlooms, std::vector<uint8_t> &isActive, std::vector<uint8_t> &isPeeled)
{
    intE prevActiveEdgePtr = activeEdgePtr; 
    intB activeBloomPtr = 0; 

    if (thdBloomBuff.size() < NUM_THREADS) thdBloomBuff.resize(NUM_THREADS);
    if (thdEdgeBuff.size() < NUM_THREADS) thdEdgeBuff.resize(NUM_THREADS);

    unsigned numBloomParts = NUM_THREADS*50; 
    if (partBloomStart.size() < numBloomParts+1)
        partBloomStart.resize(numBloomParts+1);


    if (workBloomSchedule.size() == 0)
    {
        workBloomSchedule.resize(BEG.numV);
        accWorkBloomSchedule.resize(BEG.numV + 1);
    }


    #pragma omp parallel num_threads(NUM_THREADS) 
    {
        size_t tid = omp_get_thread_num();

        std::array<intB, locBuffSizeLarge> &locBloomBuff = thdBloomBuff[tid]; unsigned locBloomBuffPtr = 0;
        std::array<intE, locBuffSizeLarge> &locEdgeBuff = thdEdgeBuff[tid]; unsigned locEdgeBuffPtr = 0;

        //Explore active edges and activate blooms
        #pragma omp for schedule (dynamic)
        for (intE i=activeEdgeStartOffset; i<prevActiveEdgePtr; i++)
        {
            intE e = activeEdges[i];
            assert(!isPeeled[e]);
            assert(isActive[e]);
            intE NeI = BEG.edgeDegree[e];
            for (intE j=0; j<NeI; j++)
            {
                intB belink = BEG.edgeVI[e]+j;
                intB bloomId = BEG.edgeEI[belink].first;
                intE neighEdgeId = BEG.edgeEI[belink].second;
                if (isPeeled[neighEdgeId] || (BEG.bloomDegree[bloomId]<2)) continue;
                if (isActive[neighEdgeId] && (neighEdgeId>e)) continue;

                intE updateVal = BEG.bloomDegree[bloomId]-1;
                //update neighbor edge
                intE prevTipVal = tipVal[neighEdgeId];
                if (prevTipVal >= kHi) 
                {
                    prevTipVal = __sync_fetch_and_sub(&tipVal[neighEdgeId], updateVal);
                    if ((prevTipVal < kHi + updateVal) && (prevTipVal >= kHi))
                    {
                        locEdgeBuff[locEdgeBuffPtr++] = neighEdgeId;
                        locEdgeBuffPtr = updateGlobalQueue(locEdgeBuffPtr, locBuffSizeLarge, activeEdgePtr, locEdgeBuff, activeEdges); 
                    }
                }

                //update bloom
                intE numDels = __sync_fetch_and_add(&bloomUpdates[bloomId], (intE)1);
                if (numDels==0)
                {
                    locBloomBuff[locBloomBuffPtr++] = bloomId;
                    locBloomBuffPtr = updateGlobalQueue(locBloomBuffPtr, locBuffSizeLarge, activeBloomPtr, locBloomBuff, activeBlooms);
                } 
            } 
        }
        if (locBloomBuffPtr > 0)
            locBloomBuffPtr = updateGlobalQueue(locBloomBuffPtr, locBloomBuffPtr, activeBloomPtr, locBloomBuff, activeBlooms);

        #pragma omp barrier

        #pragma omp for
        for (intE i=activeEdgeStartOffset; i<prevActiveEdgePtr; i++)
        {
            intE e = activeEdges[i];
            isActive[e] = false;
            isPeeled[e] = true; 
        }


        //LOAD BALANCING
        #pragma omp for
        for (intB i=0; i<activeBloomPtr; i++)
            workBloomSchedule[i] = BEG.bloomDegree[activeBlooms[i]];

        //compute prefix scan
        int bloomsPerThd = (activeBloomPtr-1)/NUM_THREADS + 1;
        if (tid==0)
            accWorkBloomSchedule[0] = 0;
        #pragma omp barrier
        if (bloomsPerThd < 10)
        {
            #pragma omp single
            {
                for (intB i=0; i<activeBloomPtr; i++)
                    accWorkBloomSchedule[i+1] =  accWorkBloomSchedule[i] + workBloomSchedule[i];
            }
        } 
        else
        {
            intB startBloomIdx = bloomsPerThd*tid+1;
            intB endBloomIdx = std::min(startBloomIdx + bloomsPerThd, activeBloomPtr+1);
            accWorkBloomSchedule[startBloomIdx] = workBloomSchedule[startBloomIdx-1];
            for (intB i=startBloomIdx+1; i<endBloomIdx; i++)
                accWorkBloomSchedule[i] = accWorkBloomSchedule[i-1] + workBloomSchedule[i-1]; 
            #pragma omp barrier
            #pragma omp single
            {
                for (size_t i=1; i<NUM_THREADS; i++)
                {
                    intB prevEnd = bloomsPerThd*i + 1; if (prevEnd > activeBloomPtr) continue;
                    intB tend = std::min(prevEnd + bloomsPerThd, activeBloomPtr+1);
                    accWorkBloomSchedule[tend-1] += accWorkBloomSchedule[prevEnd-1];
                }
                partBloomStart[0] = 0;
            }
            #pragma omp barrier
            if (tid>0)
            {
                intB blockScan = accWorkBloomSchedule[startBloomIdx-1];
                for (intB i=startBloomIdx; i<endBloomIdx-1; i++)
                    accWorkBloomSchedule[i] += blockScan;
            }
        }

        #pragma omp barrier

        intB workPerPart = (accWorkBloomSchedule[activeBloomPtr]-1)/numBloomParts + 1;

        //find task offsets
        #pragma omp for
        for (intB i=0; i<numBloomParts; i++)
        {
            intB ptrOff = std::lower_bound(accWorkBloomSchedule.begin(), accWorkBloomSchedule.begin()+activeBloomPtr+1, workPerPart*(i+1)) - accWorkBloomSchedule.begin();
            partBloomStart[i+1] = std::min(ptrOff, activeBloomPtr); 
        }

        #pragma omp barrier
        #pragma omp for
        for (intB i=0; i<numBloomParts; i++)
        {
            assert(partBloomStart[i+1] >= partBloomStart[i]);
        }




        #pragma omp barrier

        //explore active blooms and update edge supports
        #pragma omp for schedule (dynamic,5)
        for (intB i=0; i<activeBloomPtr; i++)
        {
            intB bloomId = activeBlooms[i];
            intE numDels = bloomUpdates[bloomId];
            bloomUpdates[bloomId] = 0;

            intB baseIndex = BEG.bloomVI[bloomId];
            for (intE j=0; j<BEG.bloomDegree[bloomId]; j++) 
            {
                intE e1Id = BEG.bloomEI[baseIndex + j].first;
                intE e2Id = BEG.bloomEI[baseIndex + j].second;

                if (isPeeled[e1Id] || isPeeled[e2Id])
                {
                    std::swap(BEG.bloomEI[baseIndex+j], BEG.bloomEI[baseIndex+BEG.bloomDegree[bloomId]-1]);
                    j--;
                    BEG.bloomDegree[bloomId]--;
                    continue;
                }

                intE prevTipVal = tipVal[e1Id];
                if (prevTipVal >= kHi) 
                {
                    prevTipVal = __sync_fetch_and_sub(&tipVal[e1Id], numDels);
                    if ((prevTipVal < kHi + numDels) && (prevTipVal >= kHi))
                    {
                        locEdgeBuff[locEdgeBuffPtr++] = e1Id;
                        locEdgeBuffPtr = updateGlobalQueue(locEdgeBuffPtr, locBuffSizeLarge, activeEdgePtr, locEdgeBuff, activeEdges); 
                    }
                    //else if (prevTipVal < kHi) __sync_fetch_and_add(&tipVal[e1Id], numDels);
                }

                prevTipVal = tipVal[e2Id];
                if (prevTipVal >= kHi) 
                {
                    prevTipVal = __sync_fetch_and_sub(&tipVal[e2Id], numDels);
                    if ((prevTipVal < kHi + numDels) && (prevTipVal >= kHi))
                    {
                        locEdgeBuff[locEdgeBuffPtr++] = e2Id;
                        locEdgeBuffPtr = updateGlobalQueue(locEdgeBuffPtr, locBuffSizeLarge, activeEdgePtr, locEdgeBuff, activeEdges); 
                    }
                    //else if (prevTipVal < kHi) __sync_fetch_and_add(&tipVal[e2Id], numDels);
                }
            }
        }
        if (locEdgeBuffPtr > 0)
            locEdgeBuffPtr = updateGlobalQueue(locEdgeBuffPtr, locEdgeBuffPtr, activeEdgePtr, locEdgeBuff, activeEdges);

        #pragma omp barrier

        #pragma omp for
        for (intE i=prevActiveEdgePtr; i<activeEdgePtr; i++)
            isActive[activeEdges[i]] = true;
    }
    return activeEdgePtr;
} 
