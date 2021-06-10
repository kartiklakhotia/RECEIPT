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
        }
        if (strcmp(argv[i], "-h") == 0)
        {
            helpReq = true;
            break;
        }
    }
    if (helpReq)
    {
        printf("command to run is \n             ./decomposeSeqWing -i <inputFile> -o <outputFile> \n\n");
        return 0;
    }
    if ((!ipExists))
    {
        printf("ERROR: correct command is \n             ./decomposeSeqWing -i <inputFile> -o <outputFile> \n\n");
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
    printf("TIME: sort = %lf, reordering = %lf, counting = %lf, rearranging = %lf, total = %lf\n", (sortDone-start)*1000, (reOrderDone-sortDone)*1000, (countDone-labelingDone)*1000, (stop-countDone)*1000, (stop-start)*1000);


    printf("beginning decomposition\n");
    //edge ID to vertices (u,v)
    std::vector<std::pair<intV, intV>> eIdToV;
    edgeId_to_vertices(G, eIdToV);

    //init priority queue
    KHeap<intE, intE> queue(G.numE); 
    for (intV i=0; i<G.numE; i++)
        queue.update(i, tipVal[i]); 

    //reset wedge counts
    std::vector<intV> &numW = wedgeCnt[0];
    for (intV i=0; i<numW.size(); i++)
        numW[i]=0;

    //2-hop neighborhood size. Used to select which vertex to start from
    std::vector<intE> hop2Size (G.numT, 0);
    for (intV i=0; i<G.numT; i++)
        for (intV j=0; j<G.deg[i]; j++)
            hop2Size[i] += G.deg[G.adj[i][j]];

    //has the edge been processed
    std::vector<bool> isProcessed (G.numE, 0);

    //is the vertex in the neighborhood of the edge being peeled
    std::vector<intE> marked (G.numT, G.numE+1);

    //start peeling
    intV finished = 0; 
    std::vector<intV> hop2Neighs (std::max(G.numU, G.numV));
    intV hop2NeighsPtr = 0;
    
    intE maxT = 0;
    int numEdgesPeeled = 0;
    while(!queue.empty())
    {
        numEdgesPeeled++;
        assert(numEdgesPeeled <= G.numE);
        //std::cout << numEdgesPeeled << "\r";
        std::pair<intE, intE> kv = queue.top();

        intE e = kv.first;
        intE k = kv.second;
        queue.pop();
        isProcessed[e] = 1;

        if (k==0) continue;

        //u -> vertex whose 2-hop neighborhood will be explored
        intV u = eIdToV[e].first;
        intV v = eIdToV[e].second;
        if (hop2Size[u] > hop2Size[v]) std::swap(u,v);

        //Mark neighbors of v
        for (intV i=0; i<G.deg[v]; i++)
        {
            intV neigh = G.adj[v][i];
            intE e3 = G.eId[v][i];
            if ((neigh==v) || (isProcessed[e3]==1)) continue;
            marked[neigh] = e3;
        }
       
        //Explore 2-hop neighborhood and count #2-hop paths to 2-hop neighbors
        for (intV i=0; i<G.deg[u]; i++)
        {
            intV neigh = G.adj[u][i];
            intE e1 = G.eId[u][i];
            intE e1Updates = 0;
            if ((neigh==v) || (isProcessed[e1]==1)) continue;
            intV neighDeg;
            for (intV j=0; j<G.deg[neigh]; j++)
            {
                intV neighOfNeigh = G.adj[neigh][j];
                intV e2 = G.eId[neigh][j];
                if ((isProcessed[e2]==1) || (marked[neighOfNeigh] > G.numE)) continue;
                if (numW[neighOfNeigh]==0) hop2Neighs[hop2NeighsPtr++] = neighOfNeigh;
                numW[neighOfNeigh]++; 
                tipVal[e2] = std::max(k, tipVal[e2]-1);
                queue.update(e2, tipVal[e2]); 
                e1Updates++;
            }
            tipVal[e1] = std::max(k, tipVal[e1]-e1Updates);
            queue.update(e1, tipVal[e1]);
        }    

        for (intV i=0; i<hop2NeighsPtr; i++)
        {
            intE e3 = marked[hop2Neighs[i]];
            assert(e3 < G.numE);
            tipVal[e3] = std::max(k, tipVal[e3]-numW[hop2Neighs[i]]);
            queue.update(e3, tipVal[e3]);
            numW[hop2Neighs[i]] = 0;
        }

        //Reset 2-hop neighborhood
        hop2NeighsPtr = 0;

        //Reset marked neighbors of v
        for (intV i=0; i<G.deg[v]; i++)
            marked[G.adj[v][i]] = G.numE+1;
        
        maxT = k;
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
