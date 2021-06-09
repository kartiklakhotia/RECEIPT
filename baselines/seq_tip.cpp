#define DEBUG 
#undef DEBUG
#include "count.h"
#include "kheap.h"

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
        printf("command to run is \n             ./decomposeSeq -i <inputFile> -o <outputFile> -s <peelSide> \n\n");
        printf("To peel U vertex set (LHS in edge list), use \"-s 0\", otherwise use \"-s 1\"\n");
        return 0;
    }
    if ((!ipExists) || ((peelSide != 0) && (peelSide != 1)))
    {
        printf("ERROR: correct command is \n             ./decomposeSeq -i <inputFile> -o <outputFile> -s <peelSide> \n\n");
        printf("To peel U vertex set (LHS in edge list), use \"-s 0\", otherwise use \"-s 1\"\n");
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

    printf("counting butterflies\n");
    std::vector<intB> butterflyCnt (G.numT);
    std::vector<std::vector<intV>> wedgeCnt (NUM_THREADS, std::vector<intV> (G.numT, 0));
    count_per_vertex(G, butterflyCnt, wedgeCnt);
    double countDone = omp_get_wtime();
    intB totCnt = parallel_reduce<intB, intB>(butterflyCnt);
    printf("total butterflies = %lld\n", totCnt/4);
    std::vector<intB> tipVal;
    tipVal.swap(butterflyCnt);
    
    double stop = omp_get_wtime();
    printf("TIME: sort = %lf, reordering = %lf, counting = %lf, rearranging = %lf, total = %lf\n", (sortDone-start)*1000, (reOrderDone-sortDone)*1000, (countDone-reOrderDone)*1000, (stop-countDone)*1000, (stop-start)*1000);


    printf("beginning decomposition\n");
    //init priority queue
    KHeap<intV, intB> queue(G.numU); 
    std::vector<intV> vtxToIdx (G.numT, G.numT);
    for (intV i=0; i<G.numU; i++)
    {
        queue.update(i, tipVal[G.uLabels[i]]); 
        vtxToIdx[G.uLabels[i]] = i;
    }
    //reset wedge counts
    std::vector<intV> &numW = wedgeCnt[0];
    for (intV i=0; i<numW.size(); i++)
        numW[i]=0;
    //start peeling
    intV finished = 0; 
    std::vector<intV> hop2Neighs;
    intB maxT = 0;
    bool suppUp = false;
    while(!queue.empty())
    {
        std::pair<intV, intB> kv = queue.top();
        intV v = G.uLabels[kv.first];
        intB k = kv.second;
        queue.pop();
        if (k==0) continue;
       
        intV deg;
        std::vector<intV> &neighList = G.get_neigh(v, deg);
        for (intV i=0; i<deg; i++)
        {
            intV neigh = neighList[i];
            intV neighDeg;
            std::vector<intV> &neighOfNeighList = G.get_neigh(neigh, neighDeg);
            for (intV j=0; j<neighDeg; j++)
            {
                intV neighOfNeigh = neighOfNeighList[j];
                if ((tipVal[neighOfNeigh] <= k)) continue;
                if (numW[neighOfNeigh]==0) hop2Neighs.push_back(neighOfNeigh);
                numW[neighOfNeigh]++; 
            }
        }        
        for (auto x : hop2Neighs)
        {
            if (numW[x] >= 2)
            {
                suppUp = true;
                intB butterflies = choose2<intB, intV>(numW[x]);
                tipVal[x] = std::max(k, tipVal[x]-butterflies);
                assert(vtxToIdx[x] < G.numT);
                assert(tipVal[x] >= k);
                queue.update(vtxToIdx[x], tipVal[x]); 
            }
            numW[x] = 0;
        } 
        hop2Neighs.clear();
        maxT = k;
    }
    if (suppUp) printf("updated supports\n");
    printf("maximum tip value = %lld\n", maxT);
    double decEnd = omp_get_wtime();
    printf("TIME to decompose = %lf\n", (decEnd-stop)*1000);

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
