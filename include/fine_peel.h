#include "fine_peel_funcs.h"

/*****************************************************************************
Fine-grained decomposition of all partitions 
Arguments:
    1. G-> graph object
    2. partId -> partition ID
    3. partVertices -> array of vertices in that partition
    4. partTipVals -> range of support values for vertices in that partition
    5. partPeelComplexity -> work required to peel each partition
    6. tipVal -> (ip/op) support value of vertices
                 initialized with the initial support of vertices when their corresponding partition starts peeling
    7. wedgeCnt -> 2D array for threads to store wedges while counting
******************************************************************************/
void process_partitions (Graph &G, std::vector<std::vector<intV>> &partVertices, std::vector<std::pair<intB, intB>> &partTipVals, std::vector<intB> &partPeelComplexity, std::vector<intB> &tipVal, std::vector<std::vector<intV>> &wedgeCnt)
{
    int numParts = partVertices.size();

    std::vector<intV> vtxToIdx(G.numT); //vertex to id in the partVertices[i] list
    std::vector<int> vtxToPart(G.numT);

    std::vector<intE> countWork;
    std::vector<intE> peelWork;
    intB totalCountComplexity = estimate_total_workload(G, countWork, peelWork);


    std::vector<uint8_t> isActive(G.numT);
    std::vector<uint8_t> isUpdated(G.numT);
    std::vector<intB> peelCnts(G.numT);
    //butterfly count per vertex from higher partitions 
    std::vector<intB> nonNativeButterflyCount (G.numT); 
    #pragma omp parallel for 
    for (intV i=0; i<G.numT; i++)
    {
        isActive[i] = 0;
        isUpdated[i] = 0;
        peelCnts[i] = 0;
        nonNativeButterflyCount[i] = 0;
        //countWorkCpy[i] = 0;
        //peelWorkCpy[i] = 0;
    }

    //if greater than a threshold, process the partition in parallel
    intB avgPeelComplexity = parallel_reduce<intB, intB>(partPeelComplexity)/numParts; 
    intB parProcThresh = 5*avgPeelComplexity;

    //process the partitions in decreasing order of complexity for load balance
    std::vector<int> partOrder (numParts);
    for (int i=0; i<numParts; i++)
        partOrder[i] = i;
    serial_sort_kv(partOrder, partPeelComplexity);

    //mapping from vertex to partitions
    //mapping from vertex to ID in partitions (used for bucketing)
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i=0; i<numParts; i++)
    {
        for (intV j=0; j<partVertices[i].size(); j++)
        {
            vtxToPart[partVertices[i][j]] = i;
            vtxToIdx[partVertices[i][j]] = j;
        }
    }


    #pragma omp parallel for
    for (intV i=0; i<G.numU; i++)
        G.delete_vertex(G.uLabels[i]);



    for (int i=0; i<numParts; i++)
    {
        intB newLo = ((intB)G.numU)*((intB)G.numU);
        intB newHi = 0;
        #pragma omp parallel for reduction (min:newLo) reduction (max:newHi)
        for (intV j=0; j<partVertices[i].size(); j++)
        {
                newLo = std::min(newLo, tipVal[partVertices[i][j]]); 
                newHi = std::max(newHi, tipVal[partVertices[i][j]]+1);
        }
        partTipVals[i].first = std::max(newLo, partTipVals[i].first);
        partTipVals[i].second = std::min(newHi, partTipVals[i].second);
        
        if (partPeelComplexity[i] > parProcThresh)
        {
            double fdInitTime = omp_get_wtime();
            #pragma omp parallel for
            for (intV j=0; j<partVertices[i].size(); j++)
                G.restore_vertex(partVertices[i][j]);
            G.delete_edges();
            //compute non-native counts for each partition
            count_per_vertex(G, partVertices[i], nonNativeButterflyCount, wedgeCnt);
            #pragma omp parallel for
            for (intV j=0; j<partVertices[i].size(); j++)
                nonNativeButterflyCount[partVertices[i][j]] = tipVal[partVertices[i][j]] - nonNativeButterflyCount[partVertices[i][j]];
            #pragma omp parallel for
            for (intV j=0; j<partVertices[i].size(); j++)
                G.delete_vertex(partVertices[i][j]);
            G.restore_edges();
            double fdEndTime = omp_get_wtime();
        }
    }

    #pragma omp parallel for
    for (intV i=0; i<G.numU; i++)
        G.restore_vertex(G.uLabels[i]);
    G.sort_adj();


    #pragma omp parallel num_threads(NUM_THREADS)
    {
        unsigned int tid = omp_get_thread_num();
        myCSR partG;

        //thread local versions of arrays that will be frequently updated
        std::vector<uint8_t> tlIsUpdated (isUpdated.size(), false);
        std::vector<uint8_t> tlIsActive (isActive.size(), false);
        std::vector<intB> tlTipVal (G.numT);
        std::vector<intB> tlPeelCnts (G.numT);

        #pragma omp for schedule(dynamic, 1)
        for (int i=0; i<numParts; i++)
        {

            int partId = partOrder[i];

            for (auto x : partVertices[partId])
            {
                tlTipVal[x] = tipVal[x];
                tlPeelCnts[x] = peelCnts[x];
            }


            double partStart = omp_get_wtime();


            create_part_csr(G, partId, partVertices[partId], vtxToPart, partG); 
                

            double graphCopyEnd = omp_get_wtime();
#ifdef DEBUG                
            printf("partition id = %d, time to compute csr = %lf\n", partId, (graphCopyEnd-partStart)*1000);
#endif
            std::vector<intE> peelWorkCpy;
            std::vector<intE> countWorkCpy;
            intB totalCountComplexityCpy = estimate_total_workload_part(partG, countWorkCpy, peelWorkCpy);
            intB totalPeelComplexity = 0;
            for (auto x : partVertices[partId])
                totalPeelComplexity += peelWorkCpy[x];

            //compute support coming from higher partitions
            //first compute support from within the same partition
            if ((partPeelComplexity[partId] <= parProcThresh) && (totalPeelComplexity > totalCountComplexity))
            {
                count_per_vertex(partG, G, partVertices[partId], nonNativeButterflyCount, wedgeCnt[tid]);
                //then subtract from net support
                for (intV j=0; j<partVertices[partId].size(); j++)
                    nonNativeButterflyCount[partVertices[partId][j]] = tlTipVal[partVertices[partId][j]] - nonNativeButterflyCount[partVertices[partId][j]];
                double countEnd = omp_get_wtime();
                printf("partition id = %d, time to compute non-native counts = %lf\n", partId, (countEnd-graphCopyEnd)*1000);
            }

            //peel partition serially
            if (partTipVals[partId].first + 1 < partTipVals[partId].second) 
                peel_partition_ser_q(partG, G, totalCountComplexityCpy, peelWorkCpy, countWorkCpy, partId, partVertices[partId], vtxToIdx, tlTipVal, nonNativeButterflyCount, partTipVals[partId].first, partTipVals[partId].second, wedgeCnt[tid], tlIsActive, tlIsUpdated, tlPeelCnts); 
            else
            {
                for (intV j=0; j<partVertices[partId].size(); j++)
                    tlTipVal[partVertices[partId][j]] = partTipVals[partId].first;
            }


            for (intV j=0; j<partVertices[partId].size(); j++)
                tipVal[partVertices[partId][j]] = tlTipVal[partVertices[partId][j]];

            double partEnd = omp_get_wtime();
#ifdef DEBUG    
            printf("partition id = %d, time taken = %lf\n", partId, (partEnd-partStart)*1000);
#endif


        } 
    }


    #pragma omp parallel for 
    for (intV i=0; i<G.numU; i++)
        G.delete_vertex(G.uLabels[i]);

}






/*****************************************************************************
Fine-grained wing decomposition of an edge partition
Arguments:
    1. BEG -> BE-Index of the partition 
    2. kLo, kHi -> wing number range of a partition
    3. partEIdToEId -> mapping from edge index in partition to edge index in G
    4. activeBlooms, bloomUpdates -> arrays to store blooms with non-zero updates during processing
    4. supp -> support vector
    5. isPeeled -> edges already peeled
    6. initSupp -> initial support vector
    7. outSupp -> output wing numbers
    8. isLastPart -> boolean flag to indicate if it is the last partition
******************************************************************************/
void fine_peel_part_wing(BEGraphLoMem &BEG, intE kLo, intE kHi, std::vector<intE> &partEIdToEId, std::vector<intB> &activeBlooms, std::vector<intE> &supp, std::vector<intE> &bloomUpdates, std::vector<bool> &isPeeled, std::vector<intE> &initSupp, std::vector<intE> &outSupp, bool isLastPart)
{
    
    //std::vector<intE> supp (BEG.numU);
    if (supp.size() < BEG.numU) supp.resize (BEG.numU);
    KHeap<intE, intE> queue (BEG.numU);
    for (intE i=0; i<BEG.numU; i++)
    {
        supp[i] = initSupp[partEIdToEId[i]];
        queue.update(i, supp[i]);
    }
    //std::vector<bool> isPeeled (BEG.numU+2, false);
    if (isPeeled.size() < BEG.numU+2) isPeeled.resize(BEG.numU+2);
    parallel_init(isPeeled, false);

    //std::vector<intE> bloomUpdates(BEG.numV, 0);
    if (bloomUpdates.size() < BEG.numV) bloomUpdates.resize(BEG.numV);
    parallel_init(bloomUpdates, (intE)0);
    

    intB activeBloomPtr = 0;
    intE numEdgesRem = BEG.numU;

    while(!queue.empty())
    {
        
        std::pair<intE, intE> kv = queue.top();
        intE e = kv.first; intE k = kv.second;
        assert(k >= kLo);
        if (k==0 || k>=kHi-1)
        {
            queue.pop();
            numEdgesRem--;
            isPeeled[e] = true;
            outSupp[partEIdToEId[e]] = std::min(k, kHi-1);
            continue;
        }
        if (isLastPart)
        {
            if (k >= numEdgesRem)
            {
                outSupp[partEIdToEId[e]] = k;
                while(!queue.empty())
                {
                    kv = queue.top();
                    outSupp[partEIdToEId[kv.first]] = k;
                    queue.pop();
                    numEdgesRem--;
                }
                break;
            }
        }
        intE nk = k;
        while(nk==k)
        {
            queue.pop();
            isPeeled[e] = true;
            outSupp[partEIdToEId[e]] = k;
            for (intB i=BEG.edgeVI[e]; i<BEG.edgeVI[e+1]; i++)
            {
                intB bloomId = BEG.edgeEI[i].first;
                intE neighEId = BEG.edgeEI[i].second;
                if (isPeeled[neighEId] || (BEG.bloomWdgCnt[bloomId]<2)) continue;

                intE updateVal = BEG.bloomWdgCnt[bloomId] - 1; 
                if (neighEId < BEG.numU)
                {
                    supp[neighEId] = std::max(supp[neighEId]-updateVal, k);
                    queue.update(neighEId, supp[neighEId]);
                }

                if (bloomUpdates[bloomId]==0) activeBlooms[activeBloomPtr++] = bloomId;
                bloomUpdates[bloomId]++;
            }
            if (queue.empty()) break;
            kv = queue.top();
            e = kv.first;
            nk = kv.second;
        }
        if (queue.empty()) break;
        for (intB i=0; i<activeBloomPtr; i++)
        {
            intB bloomId = activeBlooms[i];
            intE numDels = bloomUpdates[bloomId];
            bloomUpdates[bloomId] = 0;

            assert(BEG.bloomWdgCnt[bloomId] >= numDels);
            BEG.bloomWdgCnt[bloomId] -= numDels;
            intB baseEId = BEG.bloomVI[bloomId];
            assert(baseEId + BEG.bloomDegree[bloomId] <= BEG.bloomVI[bloomId+1]);
            for (intE j=0; j<BEG.bloomDegree[bloomId]; j++)
            {
                intE e1 = BEG.bloomEI[baseEId+j].first; assert(e1 < BEG.numU);
                intE e2 = BEG.bloomEI[baseEId+j].second; assert (e2 < BEG.numU+2);
                if (isPeeled[e1] || isPeeled[e2])
                {
                    std::swap(BEG.bloomEI[baseEId+j], BEG.bloomEI[baseEId + BEG.bloomDegree[bloomId] - 1]);
                    j--;
                    BEG.bloomDegree[bloomId]--;
                    continue;
                }

                assert(supp[e1] >= numDels);
                supp[e1] = std::max(k, supp[e1]-numDels);
                queue.update(e1, supp[e1]);
                if (e2 > BEG.numU) continue;

                assert(supp[e2] >= numDels);
                supp[e2] = std::max(k, supp[e2]-numDels);
                queue.update(e2, supp[e2]); 
            }
        }
        activeBloomPtr = 0;
    }

}
