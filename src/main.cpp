#define DEBUG 
#undef DEBUG
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
            if (strcmp(argv[i], "-s") == 0)
            {
                peelSide = (int)atoi(argv[i+1]);
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
        printf("command to run is \n             ./decomposePar -i <inputFile> -o <outputFile> -t <# threads> -p <# partitions to create> -s <peelSide> \n\n");
        printf("To peel U vertex set (LHS in edge list), use \"-s 0\", otherwise use \"-s 1\"\n");
        return 0;
    }
    if ((!ipExists) || ((peelSide != 0) && (peelSide != 1)))
    {
        printf("ERROR: correct command is \n             ./decomposeSeq -i <inputFile> -o <outputFile> -t <# threads> -p <# partitions to create> -s <peelSide> \n\n");
        printf("To peel U vertex set (LHS in edge list), use \"-s 0\", otherwise use \"-s 1\"\n");
        return -1;
    }
    printf("reading graph file\n");
    G.read_graph(graphFile, peelSide);
    //G.read_graph_bin(graphFile, peelSide);
    printf("graph file read\n");
    printf("# edges = %u, U  = %u, V = %u\n", G.numE, G.numU, G.numV); 
//    G.dump_graph();

    double start = omp_get_wtime();

    printf("start computation with %d threads\n", omp_get_max_threads());

    printf("sorting on degree\n");
    std::vector<intV> labelsToVertex;
    G.sort_deg(labelsToVertex);
    std::vector<intV> vertexToLabels;
    invertMap(labelsToVertex, vertexToLabels);
    //print_list_horizontal(vertexToLabels);

    double sortDone = omp_get_wtime();

    printf("reordering the graph\n");
    G.reorder_in_place(vertexToLabels); 
//    G.print_graph();
    double reOrderDone = omp_get_wtime();

    printf("counting butterflies\n");
    std::vector<intB> butterflyCnt (G.numT);
    std::vector<std::vector<intV>> wedgeCnt (NUM_THREADS, std::vector<intV> (G.numT, 0));
    count_per_vertex(G, butterflyCnt, wedgeCnt);
    double countDone = omp_get_wtime();
    intB totCnt = parallel_reduce<intB, intB>(butterflyCnt);
    printf("total butterflies = %lld\n", totCnt/4);
    return 0;
    std::vector<intB> tipVal;
    tipVal.swap(butterflyCnt);
    
    double stop = omp_get_wtime();
    printf("TIME: sort = %lf, reordering = %lf, counting = %lf, rearranging = %lf, total = %lf\n", (sortDone-start)*1000, (reOrderDone-sortDone)*1000, (countDone-reOrderDone)*1000, (stop-countDone)*1000, (stop-start)*1000);


    std::vector<std::pair<intB, intB>> partTipVals; //range of support values for each partition
    std::vector<std::vector<intV>> partVertices; //list of vertices in each partition
    std::vector<intB> partPeelWork;
    numParts = create_balanced_partitions(G, tipVal, 0, wedgeCnt, numParts, partTipVals, partVertices, partPeelWork); 
    ////write coarse decomposition details in a file
    //print_partitioning_details(cdFile, G, tipVal, numParts, partTipVals, partVertices, partPeelWork);
    double partsDone = omp_get_wtime();
    printf("TIME to coarse decompose = %lf\n", (partsDone-stop)*1000);


    
    printf("beginning fine-grained decomposition\n");
    double fineBegin = omp_get_wtime();
    process_partitions(G, partVertices, partTipVals, partPeelWork, tipVal, wedgeCnt); 
    double fineEnd = omp_get_wtime();
    printf("TIME to fine decompose = %lf\n", (fineEnd-fineBegin)*1000);

    if (opExists==true)
    {
        reorderArr(tipVal, labelsToVertex);
        intB maxTipVal = 0;
        FILE* fp = fopen(opCntFile.c_str(), "w");
        if (peelSide == 0)
            printf("printing exact tip value for U (left hand side vertices)\n");
        else
            printf("printing exact tip value for V (right hand side vertices)\n");
            
        for (intV i=0; i<G.numU; i++)
        {
            maxTipVal = std::max(maxTipVal, tipVal[i]);
            fprintf(fp, "%u, %lld \n", i, tipVal[i]);
        }
        printf("max tipval = %lld\n", maxTipVal);
        fclose(fp);
    }
    
    

    return 0;
}
