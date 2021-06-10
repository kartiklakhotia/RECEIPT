#define DEBUG 
#undef DEBUG
#include "count.h"
#include "kheap.h"

size_t numSG = 50; 

int main(int argc, char** argv)
{
    Graph G;
    std::string graphFile;
    std::string opCntFile;
    std::string cdFile;
    bool ipExists = false;
    bool opExists = false;
    bool helpReq = false;
    bool readBin = false;
    int peelSide = 0;
    omp_set_num_threads(1);
    for (int i = 1; i < argc; i++)
    {  
        if (i + 1 != argc)
        {
            if (strcmp(argv[i], "-i") == 0) // input graph filename
            {                 
                graphFile = std::string(argv[i+1]);   // The next value in the array is your value
                ipExists = true;
                i++;    // Move to the next flag
            }
            if (strcmp(argv[i], "-o") == 0) // output file in which counts are dumped
            {
                opCntFile = std::string(argv[i+1]);
                opExists = true;
                i++; 
            }
        }
        if (strcmp(argv[i], "-h") == 0)
        {
            helpReq = true;
            break;
        }
    }
    if (helpReq)
    {
        printf("command to run is \n             ./decomposePCWing -i <inputFile> -o <outputFile> \n\n");
        return 0;
    }
    if ((!ipExists))
    {
        printf("ERROR: correct command is \n             ./decomposePCWing -i <inputFile> -o <outputFile> \n\n");
        return -1;
    }
    printf("reading graph file\n");
    G.read_graph(graphFile, peelSide);
    printf("graph file read\n");
    printf("# edges = %u, U  = %u, V = %u\n", G.numE, G.numU, G.numV); 

    double start = omp_get_wtime();

    printf("start computation with %d threads\n", omp_get_max_threads());

    printf("sorting on degree\n");
    std::vector<intV> labelsToVertex;
    G.sort_deg(labelsToVertex);
    std::vector<intV> vertexToLabels;
    invertMap(labelsToVertex, vertexToLabels);

    double sortDone = omp_get_wtime();

    printf("reordering the graph\n");
    G.reorder_in_place(vertexToLabels); 
    double reOrderDone = omp_get_wtime();


    printf("labeling the edges\n");
    label_edges(G);
    double labelingDone = omp_get_wtime();

    printf("counting butterflies\n");
    std::vector<intE> butterflyCnt (G.numE);
    std::vector<std::vector<intV>> wedgeCnt (NUM_THREADS, std::vector<intV> (G.numT, 0));
    count_per_edge(G, butterflyCnt, wedgeCnt);
    double countDone = omp_get_wtime();
    intB totCnt = parallel_reduce<intB, intE>(butterflyCnt);
    printf("total butterflies = %lld\n", totCnt/4);
    std::vector<intE> tipVal;
    tipVal.swap(butterflyCnt);
    
    double stop = omp_get_wtime();
    printf("TIME: sort = %lf, reordering = %lf, counting = %lf, rearranging = %lf, total = %lf\n", (sortDone-start)*1000, (reOrderDone-sortDone)*1000, (countDone-labelingDone)*1000, (stop-countDone)*1000, (stop-start-(labelingDone-reOrderDone))*1000);


    printf("beginning decomposition\n");
    //edge ID to vertices (u,v)
    std::vector<std::pair<intV, intV>> eIdToV;
    edgeId_to_vertices(G, eIdToV);


    //sort edges based on support
    std::vector<intE> eIds (G.numE);
    std::iota(eIds.begin(), eIds.end(), 0);
    serial_sort_kv(eIds, tipVal);
    for (intE i=1; i<eIds.size(); i++)
        assert(tipVal[eIds[i]] <= tipVal[eIds[i-1]]);

    //old edge ID to subgraph edge ID mapping
    std::vector<intE> oldIdToSGId (G.numE);
    for (intE i=0; i < eIds.size(); i++) oldIdToSGId[eIds[i]] = i;
    

    //Find Upper bound on max wing number
    intE ub = 0;
    for (intE i=0; i<G.numE; i++)
    {
        intE k = tipVal[eIds[i]];
        intE numEdgesWithHigherSupport = i+1;
        if (numEdgesWithHigherSupport >= k)
        {
            ub = k;
            break;
        }
    }
    printf("max upper bound = %u\n", ub);

    //sort adjacencies according to tip value of edges
    for (intV i=0; i<G.numT; i++)
    {
        serial_sort_kv(G.eId[i], tipVal);
        for (intV j=0; j<G.deg[i]; j++)
        {
            if (j > 0) assert (tipVal[G.eId[i][j]] <= tipVal[G.eId[i][j-1]]);
            std::pair<intV, intV> st = eIdToV[G.eId[i][j]];
            assert(st.first==i || st.second==i);
            if (st.first==i) G.adj[i][j] = st.second;
            else G.adj[i][j] = st.first;        
        }
        G.deg[i] = 0;
    }

    intE range = (ub)/numSG + 1;

    //helper data structures
    std::vector<bool> isComputed (G.numE, false);
    std::vector<bool> isProcessed(G.numE, false);
    std::vector<bool> isEdgeInSG (G.numE, false);
    std::vector<intE> uncomputedEdgesInSG (G.numE);
    std::vector<intE> sgBcnt (G.numE, 0);


    intE numEdgesInSG = 0;
    intE prevEdgesInSG = 0;

    intE maxT = 0;
    intE numEdgesPeeled = 0;


    for (int sgId = numSG-1; sgId >=0; sgId--)
    {

        intE kLo = range*sgId; kLo = std::min(kLo, ub);
        intE kHi = range*(sgId+1); kHi = std::min(kHi, ub);

#ifdef DEBUG
        printf("processing sg, kLo=%lld, kHi=%lld\n", kLo, kHi);
#endif

        //create subgraph
        if (numEdgesInSG < G.numE)
        {
            while(tipVal[eIds[numEdgesInSG]] >= kLo)
            {
                assert(eIds[numEdgesInSG] < G.numE);
                isEdgeInSG[eIds[numEdgesInSG]] = true;
                std::pair<intV, intV> st = eIdToV[eIds[numEdgesInSG]];
                intV s = st.first; intV t = st.second;
                assert(s<G.numT && t<G.numT);
                assert(G.eId[s][G.deg[s]] == eIds[numEdgesInSG]);
                assert(G.eId[t][G.deg[t]] == eIds[numEdgesInSG]);
                G.deg[s]++; G.deg[t]++;
                numEdgesInSG++;
                if (numEdgesInSG >= G.numE) break;
            }
        }
        
        //count butterflies and create BE index
        BEGraphLoMem BESG;
        sg_count_and_create_BE_Index(G, isEdgeInSG, isComputed, sgBcnt, BESG, wedgeCnt); 

#ifdef DEBUG
        printf("computed be index for %d\n", sgId);
#endif

         
        //filter edges with bitruss number lower than the threshold (kLo)
        std::vector<intE> bloomUpdates(BESG.numV, 0);
        std::vector<intB> activeBlooms;
        intE numEdgesBelowLo = 0; intE uncomputedPtr = 0;
        if (kLo > 0)
        {
            for (size_t i=prevEdgesInSG; i<numEdgesInSG; i++)
            {
                if (sgBcnt[eIds[i]] < kLo)
                    uncomputedEdgesInSG[numEdgesBelowLo++] = eIds[i];
            }
            

            intE prevUncomputedPtr = uncomputedPtr;
            uncomputedPtr = numEdgesBelowLo;
            while(numEdgesBelowLo > 0)
            {
                numEdgesBelowLo = 0;
                for (intE i=prevUncomputedPtr; i<uncomputedPtr; i++)
                {
                    intE e = uncomputedEdgesInSG[i];
                    for (intE i=0; i<BESG.edgeDegree[e]; i++)
                    {
                        intB bloomId = BESG.edgeEI[BESG.edgeVI[e]+i].first;
                        intE neighEId = BESG.edgeEI[BESG.edgeVI[e]+i].second;

                        if (BESG.bloomWdgCnt[bloomId] < 2) continue;
                        if (isProcessed[neighEId]) continue;
                        intE updateVal = BESG.bloomWdgCnt[bloomId]-1;
                        if (!isComputed[neighEId])
                        {
                            intE prevBcnt = sgBcnt[neighEId];
                            sgBcnt[neighEId] = std::max(prevBcnt-updateVal, kLo-1);
                            if ((sgBcnt[neighEId] < kLo) && (prevBcnt >= kLo))
                                uncomputedEdgesInSG[uncomputedPtr + numEdgesBelowLo++] = neighEId;
                        }
                        
                        if (bloomUpdates[bloomId]==0) activeBlooms.push_back(bloomId);
                        bloomUpdates[bloomId]++; 
                    }
                    isProcessed[e] = true;
                }
                for (auto bloomId : activeBlooms)
                {
                    intE numDels = bloomUpdates[bloomId];
                    bloomUpdates[bloomId] = 0;
                    BESG.bloomWdgCnt[bloomId] -= numDels;
                    intB baseEId = BESG.bloomVI[bloomId];   
                    for (intE i=0; i<BESG.bloomDegree[bloomId]; i++)
                    {
                        intE e1 = BESG.bloomEI[baseEId+i].first; 
                        intE e2 = BESG.bloomEI[baseEId+i].second; 

                        if (isProcessed[e1] || isProcessed[e2])
                        {
                            std::swap(BESG.bloomEI[baseEId+i], BESG.bloomEI[baseEId+BESG.bloomDegree[bloomId]-1]);
                            i--;
                            BESG.bloomDegree[bloomId]--;
                            continue;
                        }

                        if (!isComputed[e1])
                        {
                            intE prevBcnt = sgBcnt[e1];
                            sgBcnt[e1] = std::max(prevBcnt-numDels, kLo-1);
                            if ((sgBcnt[e1] < kLo) && (prevBcnt >= kLo))
                                uncomputedEdgesInSG[uncomputedPtr + numEdgesBelowLo++] = e1;
                        }

                        if (!isComputed[e2])
                        {
                            intE prevBcnt = sgBcnt[e2];
                            sgBcnt[e2] = std::max(prevBcnt-numDels, kLo-1);
                            if ((sgBcnt[e2] < kLo) && (prevBcnt >= kLo))
                                uncomputedEdgesInSG[uncomputedPtr + numEdgesBelowLo++] = e2;
                        }
                    } 
                }
                activeBlooms.clear();
                prevUncomputedPtr = uncomputedPtr;
                uncomputedPtr += numEdgesBelowLo;
            }
        }
        
        
        //compute bitruss numbers of remaining edges 
        intV finished = 0;  

        assert(numEdgesInSG >= prevEdgesInSG+uncomputedPtr);

        intE shiftPtr = prevEdgesInSG;
        for (intE i=prevEdgesInSG; i<numEdgesInSG; i++)
        {
            if (sgBcnt[eIds[i]] >= kLo)
            {
                oldIdToSGId[eIds[i]] = shiftPtr-prevEdgesInSG;
                eIds[shiftPtr++] = eIds[i];
            }
        }
        assert(shiftPtr==numEdgesInSG-uncomputedPtr);
        for (intE i=shiftPtr; i<numEdgesInSG; i++)
        {
            eIds[i] = uncomputedEdgesInSG[i-shiftPtr];
            assert(sgBcnt[eIds[i]] < kLo);
        }


        KHeap<intE, intE> queue(shiftPtr-prevEdgesInSG); 
        for (intE i=prevEdgesInSG; i<shiftPtr; i++)
        {
            assert(eIds[i] < G.numE); assert(sgBcnt[eIds[i]] >= kLo);
            assert(oldIdToSGId[eIds[i]]==i-prevEdgesInSG);
            queue.update(i-prevEdgesInSG, sgBcnt[eIds[i]]);
        }

#ifdef DEBUG
        printf("beginning peeling\n");
#endif

        while(!queue.empty())
        {
            std::pair<intE, intE> kv = queue.top();
            intE e = eIds[prevEdgesInSG + kv.first];
            intE k = kv.second;
            assert(e<G.numE && k>=kLo);
            if (k==0)
            {
                numEdgesPeeled++; queue.pop();
                isProcessed[e] = true; continue; 
            }
            intE nk = k;
            while(nk == k)
            {
                queue.pop();
                assert(!isProcessed[e]);
                isProcessed[e] = true;
                numEdgesPeeled++; 
                assert(numEdgesPeeled <= G.numE);

                for (intE i=0; i<BESG.edgeDegree[e]; i++)
                {
                    intB bloomId = BESG.edgeEI[BESG.edgeVI[e]+i].first;
                    intE neighEId = BESG.edgeEI[BESG.edgeVI[e]+i].second;

                    if (isProcessed[neighEId] || (BESG.bloomWdgCnt[bloomId]<2)) continue;

                    intE updateVal = BESG.bloomWdgCnt[bloomId] - 1;
                    if (!isComputed[neighEId])
                    {
                        sgBcnt[neighEId] = std::max(sgBcnt[neighEId]-updateVal, k);
                        queue.update(oldIdToSGId[neighEId], sgBcnt[neighEId]);
                    }
                    
                    if (bloomUpdates[bloomId]==0) activeBlooms.push_back(bloomId);
                    bloomUpdates[bloomId]++; 
                }
                if (queue.empty()) break;
                kv = queue.top();
                e = eIds[prevEdgesInSG+kv.first];
                nk = kv.second;
            }
            for (auto bloomId:activeBlooms)
            {
                intE numDels = bloomUpdates[bloomId];
                bloomUpdates[bloomId] = 0;
                
                BESG.bloomWdgCnt[bloomId] -= numDels; 
                intB baseEId = BESG.bloomVI[bloomId];
                for (intE i=0; i<BESG.bloomDegree[bloomId]; i++)
                {
                    intE e1 = BESG.bloomEI[baseEId+i].first;
                    intE e2 = BESG.bloomEI[baseEId+i].second;
                    if (isProcessed[e1] || isProcessed[e2])
                    {
                        std::swap(BESG.bloomEI[baseEId+i], BESG.bloomEI[baseEId+BESG.bloomDegree[bloomId]-1]);
                        i--;
                        BESG.bloomDegree[bloomId]--;
                        continue;
                    }
                    if (!isComputed[e1])
                    {
                        sgBcnt[e1] = std::max(k, sgBcnt[e1]-numDels);
                        queue.update(oldIdToSGId[e1], sgBcnt[e1]);
                    }
                    if (!isComputed[e2])
                    {
                        sgBcnt[e2] = std::max(k, sgBcnt[e2]-numDels);
                        queue.update(oldIdToSGId[e2], sgBcnt[e2]);
                    }
                }
            }
            activeBlooms.clear();
            maxT = std::max(maxT, k);
        }
#ifdef DEBUG
        printf("peeled\n");
#endif
        
         

        //reset helpers
        for (intE i=prevEdgesInSG; i<shiftPtr; i++)
        {
            assert(sgBcnt[eIds[i]] >= kLo);
            tipVal[eIds[i]] = sgBcnt[eIds[i]];
            isComputed[eIds[i]] = true;
            isProcessed[eIds[i]] = false;
        }
        for (intE i=shiftPtr; i<numEdgesInSG; i++)
        {
            assert(sgBcnt[eIds[i]] < kLo);
            isProcessed[eIds[i]] = false;
            sgBcnt[eIds[i]] = 0;
        }
        prevEdgesInSG = shiftPtr;

        free_vec(BESG.bloomEI); free_vec(BESG.edgeEI);
        free_vec(BESG.bloomVI); free_vec(BESG.edgeVI);
        free_vec(BESG.bloomDegree); free_vec(BESG.edgeDegree);
        free_vec(BESG.bloomWdgCnt);

    }
    
    printf("\n");
    printf("maximum wing number = %lld\n", maxT);
    double decEnd = omp_get_wtime();
    printf("TIME to decompose = %lf\n", (decEnd-stop)*1000);

    if (opExists==true)
    {
        FILE* fp = fopen(opCntFile.c_str(), "w");
        printf("printing exact wing numbers for edges\n");
            
        for (intV i=0; i<G.numE; i++)
            fprintf(fp, "%u, %u, %lld \n", eIdToV[i].first, eIdToV[i].second, tipVal[i]);

        fclose(fp);
    }
    
    

    return 0;
}
