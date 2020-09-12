#include "count.h"

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
    5. upperBound -> highest possible tip value
    6. peelWork -> work required to peel the vertices
******************************************************************************/
std::tuple<intB, intV, intV> find_range (std::vector<intV> &vertices, std::vector<intB>&tipVal, intB targetPeelComplexity, intB lowerBound, intB upperBound, std::vector<intE> &peelWork)
{
    parallel_sort_kv_increasing<intV, intB>(vertices, tipVal);  //sort vertices on their current support
    //printf("sorted, ");

    //compute workload for each bucket - map, prefix sum, scatter
    //find bucket id for each vertex using map and prefix sum
    //scatter with atomic add to compute workload for the buckets
    std::vector<uint8_t> suppIsUniq(vertices.size()); 
    suppIsUniq[suppIsUniq.size()-1] = 1;
    #pragma omp parallel for 
    for (intV i=0; i<vertices.size()-1; i++)
        suppIsUniq[i] = (std::min(tipVal[vertices[i]], upperBound)==std::min(tipVal[vertices[i+1]], upperBound)) ? 0 : 1;
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
            suppVal[wrOffset[i]] = std::min(tipVal[v], upperBound);
            __sync_fetch_and_add(&workPerSuppVal[wrOffset[i]], work);
        }
    }


    //none of the vertices with support < lo should've survived
    assert(suppVal[0] >= lowerBound);


    //prefix sum to compute work required to peel all vertices till a particular bucket 
    parallel_prefix_sum_inclusive(workPerSuppVal, workPerSuppVal);


    //find the first bucket with work just lower than the target value
    intV tgtBktId = std::lower_bound(workPerSuppVal.begin(), workPerSuppVal.end(), targetPeelComplexity) - workPerSuppVal.begin();

    intB hi = std::max(std::min(suppVal[tgtBktId], upperBound), suppVal[0]+1); //hi should be greater than the support of the first bucket to ensure non-zero vertex peeling

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
        std::tie(hi, tgtBktId, numUniqSuppVals) = find_range(vertices, tipVal, targetPeelComplexity, lo, (intB)(((intB)G.numV)*((intB)G.numV>>1)), peelWork);

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
