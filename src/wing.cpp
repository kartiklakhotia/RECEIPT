#define DEBUG 
#undef DEBUG

#define NDEBUG
#undef NDEBUG


#include "fine_peel.h"

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
    int numParts = 150;
    int peelSide = 0;
    for (int i = 1; i < argc; i++)
    {  
        if (i + 1 != argc)
        {
            if (strcmp(argv[i], "-t") == 0) // number of threads
            {                 
                NUM_THREADS = (unsigned int)atoi(argv[i + 1]);    // The next value in the array is your value
                omp_set_num_threads(NUM_THREADS); 
                i++;    // Move to the next flag
            }
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
            if (strcmp(argv[i], "-p") == 0)
            {
                numParts = (int)atoi(argv[i+1]);
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
        printf("command to run is \n             ./decomposeParWing -i <inputFile> -o <outputFile> -t <# threads> -p <# partitions to create> \n\n");
        return 0;
    }
    if ((!ipExists))
    {
        printf("ERROR: correct command is \n             ./decomposeParWing -i <inputFile> -o <outputFile> -t <# threads> -p <# partitions to create> \n\n");
        return -1;
    }
    printf("reading graph file\n");
    G.read_graph(graphFile, peelSide);
    printf("graph file read\n");
    printf("# edges = %u, U  = %u, V = %u\n", G.numE, G.numU, G.numV); 

    double start = omp_get_wtime();

    printf("threads=%d, partitions=%d\n", omp_get_max_threads(), numParts);

#ifdef DEBUG
    printf("sorting on degree\n");
#endif
    std::vector<intV> labelsToVertex;
    G.sort_deg(labelsToVertex);
    std::vector<intV> vertexToLabels;
    invertMap(labelsToVertex, vertexToLabels);

    double sortDone = omp_get_wtime();

#ifdef DEBUG
    printf("reordering the graph\n");
#endif
    G.reorder_in_place(vertexToLabels); 
    double reOrderDone = omp_get_wtime();


    printf("labeling the edges\n");
    label_edges(G);
    double labelingDone = omp_get_wtime();

    printf("counting butterflies\n");
    std::vector<intE> butterflyCnt (G.numE);
    std::vector<std::vector<intV>> wedgeCnt (NUM_THREADS, std::vector<intV> (G.numT));
    for (size_t i=0; i<NUM_THREADS; i++)
        parallel_init(wedgeCnt[i], (intV)0);

    BEGraphLoMem BEG;
    count_and_create_BE_Index(G, butterflyCnt, BEG, wedgeCnt);
    for (size_t i=0; i<NUM_THREADS; i++)
        free_vec(wedgeCnt[i]);

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
    std::vector<int> edgeToPart (G.numE);
    std::vector<intE> edgeToPartEId (G.numE);

    //list of active (+previously deleted) edges
    std::vector<intE> activeEdges (G.numE); intE activeEdgePtr = 0; intE prevActiveEdgePtr = 0;
    std::vector<intB> activeBlooms (BEG.numV); std::vector<intE> bloomUpdates (BEG.numV);

    std::vector<std::vector<intE>> partEdges (numParts);
    std::vector<std::vector<intB>> partBlooms (numParts);
    std::vector<std::pair<intE, intE>> partRange (numParts);

    //edges remaining to be partitioned
    intE remEdges = G.numE; 

    //sort edges based on support
    std::vector<intE> eIds (G.numE);
    std::vector<intE> tempEIds (G.numE);

    std::vector<uint8_t> isActive (G.numE);
    std::vector<uint8_t> isPeeled (G.numE);
    auto f = [&] (intE i) {return (!isPeeled[i]);};

    #pragma omp parallel for num_threads(NUM_THREADS) 
    for (intE i=0; i<G.numE; i++)
    {
        eIds[i] = i;
        isActive[i] = false;
        isPeeled[i] = false;
    }

    intE kLo = 0; intE kHi = 0;
    intE nEdgesRem = G.numE;

    intE ub = find_upper_bound_wing(eIds, tipVal, nEdgesRem);

    //partition edges into different tip number ranges
    std::vector<intE> tipValPartInit (G.numE);

    intB tgtWork = 0;
    double scaling = 1.0;

    printf("COARSE PEEL\n");
    double ubTime = 0;


    double initDone = omp_get_wtime();
    for (int i=0; i<numParts; i++)
    {

        #pragma omp parallel for num_threads(NUM_THREADS) 
        for (intE j=0; j<nEdgesRem; j++)
        {
            assert(!isPeeled[eIds[j]]);
            tipValPartInit[eIds[j]] = tipVal[eIds[j]];
        }


        intE partStartDelPtr = activeEdgePtr;

        int nPartsRem = numParts-i;
        kLo = kHi;

        double ubTimeStart = omp_get_wtime();
        intE oldUb = ub;
        ub = update_upper_bound_wing(eIds, nEdgesRem, tipVal, kLo, ub);

        std::tie(kHi, tgtWork) = find_upper_bound_part(eIds, tipVal, nPartsRem, nEdgesRem, scaling, kLo, ub, oldUb);
        partRange[i] = std::make_pair(kLo, kHi);
        double ubTimeEnd = omp_get_wtime();
        ubTime += (ubTimeEnd-ubTimeStart);
#ifdef DEBUG
        printf("partition = %u, range = [%u, %u), tgt work = %lld\n", i, kLo, kHi, tgtWork);
#endif

        int peelRound = 0;

        if (nPartsRem==1)
        {
            #pragma omp parallel for num_threads(NUM_THREADS) 
            for (intE i=0; i<nEdgesRem; i++)
            {
                activeEdges[i + partStartDelPtr] = eIds[i]; 
                isPeeled[eIds[i]] = true;
            }
            activeEdgePtr += nEdgesRem;
        }
        else
        {
            prevActiveEdgePtr = activeEdgePtr;
            find_active_edges(eIds, tipVal, isActive, nEdgesRem, kLo, kHi, activeEdges, activeEdgePtr);
            while (activeEdgePtr > prevActiveEdgePtr)
            {
                intE activeEdgeStartOffset = prevActiveEdgePtr;
                prevActiveEdgePtr = activeEdgePtr;
                activeEdgePtr = update_edge_supp(BEG, tipVal, kLo, kHi, activeEdges, activeEdgePtr, activeEdgeStartOffset, bloomUpdates, activeBlooms, isActive, isPeeled);
            }
        }
     
#ifdef DEBUG
        printf("num edges deleted = %u\n", activeEdgePtr-partStartDelPtr);
#endif
        partEdges[i].resize(activeEdgePtr-partStartDelPtr);

        #pragma omp parallel num_threads(NUM_THREADS) 
        {
            #pragma omp for
            for (intE j=partStartDelPtr; j<activeEdgePtr; j++)
            {
                assert(j < G.numE);
                partEdges[i][j-partStartDelPtr] = activeEdges[j];
                assert(activeEdges[j] < G.numE);
                edgeToPart[activeEdges[j]] = i;
                edgeToPartEId[activeEdges[j]] = j-partStartDelPtr;
            }
            
        }

        scaling = compute_scale(partEdges[i], tipValPartInit, ub, tgtWork);        
#ifdef DEBUG
        printf("partition=%d, tgtWork=%lld, scaling=%lf\n", i, tgtWork, scaling);
#endif
        assert(scaling <= 1.0);


        nEdgesRem = parallel_compact_func_in_place(eIds, f, tempEIds, nEdgesRem);
        if (nEdgesRem == 0)
        {
            numParts = i+1;
            break;
        }
    }
    free_vec(isActive);
    free_vec(isPeeled);
    free_vec(activeEdges);
    free_vec(eIds);



    double coarseDone = omp_get_wtime();

    //create bloom edge graphs for each partition
    printf("constructing BEGs for individual partitions\n");
    std::vector<BEGraphLoMem> partBEG (numParts); 
    std::vector<intB> partWork (numParts);
    construct_part_BEG(BEG, numParts, partEdges, partBlooms, edgeToPart, edgeToPartEId, partBEG, partWork);

    clock_t clockStart = clock();
    double partBEGDone = omp_get_wtime();

    std::vector<int> partIds (numParts);
    std::iota(partIds.begin(), partIds.end(), 0);
    serial_sort_kv(partIds, partWork);

#ifdef DEBUG
    printf("actual partitions = %d, ", numParts);
#endif

    printf("FINE PEEL\n");
    
    //peel partitions
    int partQPtr = 0;

    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        size_t tid = omp_get_thread_num();
        int localTaskId = __sync_fetch_and_add(&partQPtr, 1);
        thread_local std::vector<intE> tlSupp, tlBloomUpdates;
        thread_local std::vector<bool> tlIsPeeled;
        while(localTaskId < numParts)
        {
            int localPartId = partIds[localTaskId];
            bool isLastPart = (localPartId == numParts-1);
            fine_peel_part_wing(partBEG[localPartId], partRange[localPartId].first, partRange[localPartId].second, partEdges[localPartId], partBlooms[localPartId], tlSupp, tlBloomUpdates, tlIsPeeled, tipValPartInit, tipVal, isLastPart);
            localTaskId = __sync_fetch_and_add(&partQPtr, 1);
            if (localTaskId >= numParts) break;
        }
    }
    

    double fineDone = omp_get_wtime();
    clock_t clockEnd = clock();
    printf("finished decomposition\n");


    intE maxT = 0;
    #pragma omp parallel for num_threads(NUM_THREADS) reduction (max:maxT)
    for (intE i=0; i<G.numE; i++)
        maxT = std::max(tipVal[i], maxT); 
    printf("maximum wing number = %lld\n", maxT);

    printf("TIME: decompose init = %lf, coarse = %lf, part BEG construction = %lf, fine = %lf,  total = %lf\n", (initDone-stop)*1000, (coarseDone-initDone)*1000, (partBEGDone-coarseDone)*1000, (fineDone-partBEGDone)*1000, (fineDone-start)*1000);

    printf("\n\n\n\n");

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
