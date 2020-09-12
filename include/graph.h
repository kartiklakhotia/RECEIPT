#ifndef MY_GRAPH_H
#define MY_GRAPH_H

#include "utils.h"

typedef unsigned int intV;
typedef unsigned int intE;
typedef long long int intB;

#define CL_SIZE 1

class Graph
{

private:

    std::vector<uint8_t> deleted; //is the vertex deleted in previous rounds
    void remove_duplicates();

public:

    intV numU;
    intV numV;
    intV numT;
    intE numE;
    std::vector<std::vector<intV>> adj;
    std::vector<std::vector<intE>> eId;
    std::vector<intV> deg;
    std::vector<intV> uLabels;
    std::vector<intV> vLabels;

    Graph(): numV(0), numU(0), numT(0), numE(0) {}
    Graph(intV V, intV U, intE E): numV(V), numU(U), numT(U+V), numE(E)
    {
        adj.resize(numT);
        uLabels.resize(numU);
        vLabels.resize(numV);
        deg.resize(numT, 0);
        deleted.resize(numT, 0);
    }

    //get neighbors of a vertex
    inline std::vector<intV>& get_neigh(intV v, intV &degV);

    //sort adjacency lists on increasing order of index or
    //decreasing order of user-specified priority 
    void sort_adj();
    void sort_adj(std::vector<intV> &priority);

    //sort vertices in the degree order
    void sort_deg(std::vector<intV> &opList);

    //reorder the graph and produce a new graph object
    void reorder(std::vector<intV> &newLabels, Graph &outG);
    //reorder the same graph object
    void reorder_in_place(std::vector<intV> &newLabels);
    //create a copy (excludes deleted vertice/edges)
    void copy(Graph &outG);

    //read graph from a file
    void read_graph(std::string &filename, int peelSide);
    void read_graph_bin(std::string &filename, int peelSide);

    //check if vertex is deleted
    inline bool is_deleted(intV v);

    //delete/restore vertices
    inline void delete_vertex(intV v);
    inline void restore_vertex(intV v);

    //delete edges
    inline void delete_edges();
    inline void restore_edges();

    void get_labels(std::vector<intV> &labels, int side);

    //FOR DEBUGGING//
    void dump_graph();
    void print_graph();

    ~Graph() {
        #pragma omp parallel for 
        for (intV i=0; i<numT; i++)
            free_vec(adj[i]);
        free_vec(adj);
        free_vec(deg);
        free_vec(uLabels);
        free_vec(vLabels);
        free_vec(eId);
        free_vec(deleted);
    }
};

void Graph::sort_adj()
{
    #pragma omp parallel for schedule(dynamic, NUM_THREADS) 
    for (intV i=0; i<numT; i++)
        std::stable_sort(adj[i].begin(), adj[i].end());
}

void Graph::sort_adj(std::vector<intV> &priority)
{
    #pragma omp parallel for schedule(dynamic, NUM_THREADS) 
    for (intV i=0; i<numT; i++)
        serial_sort_kv(adj[i], priority);
}

//sorted labels is the list of vertices in a degree sorted order
void Graph::sort_deg(std::vector<intV> &sortedLabels)
{
    sortedLabels.resize(numT);
    #pragma omp parallel for 
    for (intV i=0; i<numT; i++)
        sortedLabels[i] = i;
    parallel_sort_kv<intV, intV>(sortedLabels, deg);
}

//newLabels is the map from existing to new labels
//eg. newLabels = [1, 2, 0] means vertex 0 is now 1,
//vertex 1 is now 2 and vertex 2 is now 0.
void Graph::reorder(std::vector<intV> &newLabels, Graph &outG)
{
    outG.numU = numU;
    outG.numV = numV;
    outG.numT = numT;
    outG.numE = numE;
    
    outG.adj.resize(numT);
    outG.uLabels.resize(numU);
    outG.vLabels.resize(numV);
    outG.deg.resize(numT, 0);
    outG.deleted.resize(numT, false);
    #pragma omp parallel for 
    for (intV i=0; i<numU; i++)
        outG.uLabels[i] = newLabels[uLabels[i]];
    #pragma omp parallel for 
    for (intV i=0; i<numV; i++)
        outG.vLabels[i] = newLabels[vLabels[i]];
    parallel_sort_indices(outG.uLabels, std::less<intV>());
    parallel_sort_indices(outG.vLabels, std::less<intV>());
    #pragma omp parallel for schedule(dynamic, NUM_THREADS) 
    for (intV i=0; i<numT; i++)
    {
        intV numNeigh = adj[i].size();
        outG.adj[newLabels[i]].resize(numNeigh);        
        for (intV j=0; j<numNeigh; j++)
            outG.adj[newLabels[i]][j] = newLabels[adj[i][j]]; 
        outG.deg[newLabels[i]] = deg[i];
        outG.deleted[newLabels[i]] = deleted[i];
    }    
    outG.sort_adj();
}

void Graph::copy(Graph &outG)
{
    outG.numU = numU;
    outG.numV = numV;
    outG.numT = numT;
    outG.numE = numE;
    
    outG.adj.resize(numT);
    outG.deg.resize(numT);
    outG.deleted.resize(numT);

    parallel_vec_copy(outG.uLabels, uLabels);
    parallel_vec_copy(outG.vLabels, vLabels);
    parallel_vec_copy(outG.deleted, deleted);

    #pragma omp parallel for schedule(dynamic) 
    for (intV i=0; i<numT; i++)
    {
        intV newDeg = 0;
        if (deleted[i]==0)
        {
            intV numNeigh = adj[i].size();
            for (intV j=0; j<numNeigh; j++)
            {
                intV neigh = adj[i][j];
                if (deleted[neigh]) continue;
                newDeg++;
            }
            outG.adj[i].resize(newDeg);        
            newDeg = 0;
            for (intV j=0; j<numNeigh; j++)
            {
                intV neigh = adj[i][j];
                if (deleted[neigh]) continue;
                outG.adj[i][newDeg] = neigh; 
                newDeg++;
            }
        }
        outG.deleted[i] = deleted[i];
        outG.deg[i] = newDeg;
    }    

    outG.numE = parallel_reduce<intE, intV>(outG.deg)/2;
}



//newLabels is the map from existing to new labels
//eg. newLabels = [1, 2, 0] means vertex 0 is now 1,
//vertex 1 is now 2 and vertex 2 is now 0.
void Graph::reorder_in_place(std::vector<intV> &newLabels)
{
    std::vector<std::vector<intV>> adjNew (numT);
    std::vector<intV> uNew (numU);
    std::vector<intV> vNew (numV);
    std::vector<intV> degNew (numT);
    std::vector<uint8_t> delNew (numT);
    #pragma omp parallel for 
    for (intV i=0; i<numU; i++)
        uNew[i] = newLabels[uLabels[i]];
    #pragma omp parallel for 
    for (intV i=0; i<numV; i++)
        vNew[i] = newLabels[vLabels[i]];
    parallel_sort_indices(uNew, std::less<intV>());
    parallel_sort_indices(vNew, std::less<intV>());
    #pragma omp parallel for schedule(dynamic, NUM_THREADS) 
    for (intV i=0; i<numT; i++)
    {
        intV numNeigh = adj[i].size();
//        adjNew[newLabels[i]].resize(numNeigh);        
        adjNew[newLabels[i]].swap(adj[i]);
        for (intV j=0; j<numNeigh; j++)
            adjNew[newLabels[i]][j] = newLabels[adjNew[newLabels[i]][j]];
//            adjNew[newLabels[i]][j] = newLabels[adj[i][j]]; 
        degNew[newLabels[i]] = deg[i];
        delNew[newLabels[i]] = deleted[i];
    }    
    uNew.swap(uLabels);
    vNew.swap(vLabels);
    adjNew.swap(adj); 
    degNew.swap(deg);
    delNew.swap(deleted);
    sort_adj();
}

void Graph::read_graph_bin(std::string &filename, int peelSide)
{
    FILE* fp = fopen(filename.c_str(), "rb");
    if (fp==NULL)
    {
        fputs("file error\n", stderr);
        exit(EXIT_FAILURE);
    }
    fread(&numU, sizeof(intV), 1, fp);
    fread(&numV, sizeof(intV), 1, fp);
    if (peelSide==1)
        std::swap(numU, numV);
    numT = numU + numV;

    intE numEdgesRead = 0;
    fread(&numEdgesRead, sizeof(intE), 1, fp);
    std::vector<intV> uList (numEdgesRead);
    std::vector<intV> vList (numEdgesRead);
    fread(&uList[0], sizeof(intV), numEdgesRead, fp);
    fread(&vList[0], sizeof(intV), numEdgesRead, fp);
    if (peelSide==1)
        uList.swap(vList); 
    
    uLabels.resize(numU);
    for (intV i=0; i<numU; i++)
        uLabels[i] = i;

    //in Koblenz format, both U and V indices start from 1
    //to distinguish, add numU to V indices
    vLabels.resize(numV);
    for (intV i=0; i<numV; i++)
        vLabels[i] = i+numU;
        
    adj.resize(numT);
    //#pragma omp parallel for 
    for (intE i=0; i<numEdgesRead; i++)
    {
        intV u = uList[i]; intV v = vList[i] + numU;
        adj[u].push_back(v);
        adj[v].push_back(u);
    } 
    //printf("adjacency list created\n");

    deleted.resize(numT, false);
    remove_duplicates();
    fclose(fp);
}

void Graph::read_graph(std::string &filename, int peelSide)
{
    FILE* fp = fopen(filename.c_str(), "r");
    if (fp==NULL)
    {
        fputs("file error\n", stderr);
        exit(EXIT_FAILURE);
    }

    //ignore sentences starting with "%"
    fpos_t position;
    char buf[256];
    fgetpos(fp, &position);
    fgets(buf, sizeof(buf), fp);
    while((buf[0]=='%') && !feof(fp))
    {
        fgetpos(fp, &position);
        fgets(buf, sizeof(buf), fp);
    }
    if (feof(fp))
        return;
    fsetpos(fp, &position);
    //printf("comments finished\n");
    

    std::vector<std::pair<intV, intV>> edges;
    intV u, v;
    while(!feof(fp))
    {
        if (fscanf(fp, "%d", &u) <= 0)
            break;
        if (fscanf(fp, "%d", &v) <= 0)
            break;
        if (peelSide==1)
            std::swap(u, v);
        numU = std::max(u+1, numU);
        numV = std::max(v+1, numV); 
        edges.push_back(std::make_pair(u, v));
    }
    fclose(fp);
    //printf("%d edges read\n", edges.size());
    numT = numU + numV;
    
    uLabels.resize(numU);
    for (intV i=0; i<numU; i++)
        uLabels[i] = i;

    //in Koblenz format, both U and V indices start from 1
    //to distinguish, add numU to V indices
    vLabels.resize(numV);
    for (intV i=0; i<numV; i++)
        vLabels[i] = i+numU;
        
    adj.resize(numT);
    intE numEdgesRead = edges.size();
    for (intE i=0; i<numEdgesRead; i++)
    {
        u = edges[i].first; v = edges[i].second+numU;
        adj[u].push_back(v);
        adj[v].push_back(u);
    } 
    //printf("adjacency list created\n");

    deleted.resize(numT, false);
    remove_duplicates();
}


inline std::vector<intV>& Graph::get_neigh(intV v, intV &degV)
{
    degV = deg[v];
    return adj[v];
}


//remove duplicate edges (also computes final degree)
void Graph::remove_duplicates()
{
    deg.resize(numT);
    sort_adj();
    #pragma omp parallel for schedule(dynamic) 
    for (intV i=0; i<numT; i++)
    {
        intV numNeigh = 0;
        for (intV j=0; j<adj[i].size(); j++)
        {
            if (j==0)
                numNeigh++;
            else if (adj[i][j] != adj[i][j-1])
                numNeigh++;
        }
        //set degree
        deg[i] = numNeigh;
        if(numNeigh!=adj[i].size())
        {
            std::vector<intV> vec(numNeigh);
            numNeigh = 0;
            for (intV j=0; j<adj[i].size(); j++)
            {
                if (j==0)
                    vec[numNeigh++] = adj[i][j];
                else if (adj[i][j] != adj[i][j-1])
                    vec[numNeigh++] = adj[i][j];
            }
            vec.swap(adj[i]);
        }
        //assert(deg[i]==adj[i].size());
    }

    intE tempE = 0;
    #pragma omp parallel for reduction(+:tempE) 
    for (intV i=0; i<numT; i++)
        tempE += deg[i];
    numE = tempE/2;
}

inline bool Graph::is_deleted(intV v)
{
    return ((deleted[v]==1) ? true : false);
}

inline void Graph::delete_vertex(intV v)
{
    deleted[v] = 1;
}

inline void Graph::restore_vertex(intV v)
{
    deleted[v] = 0;
}

//delete edges incident on deleted vertices
//assuming only vertices on 'U' side are deleted
inline void Graph::delete_edges()
{
    #pragma omp parallel for
    for (intV i=0; i<numU; i++)
    {
        intV vId = uLabels[i];
        if (is_deleted(vId))
            deg[vId] = 0;
    }
    #pragma omp parallel for schedule(static, 1)
    for (intV i=0; i<numV; i++)
    {
        intV vId = vLabels[i];
        intV start = 0;
        intV end = deg[vId];
        while(start != end)
        {
            if (is_deleted(adj[vId][start]))
            {
                end--;
                std::swap(adj[vId][start], adj[vId][end]);
            }
            else
                start++;
        }
        deg[vId] = end;
        std::sort(adj[vId].begin(), adj[vId].begin() + deg[vId]); 
    }
}

//restore all edges
//assuming only vertices on 'U' side were deleted
inline void Graph::restore_edges()
{
    #pragma omp parallel for
    for (intV i=0; i<numT; i++)
        deg[i] = adj[i].size();    

    //#pragma omp parallel for
    //for (intV i=0; i<numU; i++)
    //{
    //    intV vId = uLabels[i];
    //    deg[vId] = adj[vId].size();
    //}
    //#pragma omp parallel for schedule(static, 1)
    //for (intV i=0; i<numV; i++)
    //{
    //    intV vId = vLabels[i];
    //    if(deg[vId] != adj[vId].size())
    //    {
    //        deg[vId] = adj[vId].size();
    //        //sorting not required if we use "delete_edges()" 
    //        //before every partition <TODO>
    //        std::sort(adj[vId].begin(), adj[vId].begin() + deg[vId]); 
    //    }
    //}
}

void Graph::dump_graph()
{
    FILE* fp = fopen("dump.txt", "w"); 
    for (intV i=0; i<numU; i++)
    {
        for (intV j=0; j<deg[i]; j++)
            fprintf(fp,"%d %d\n", i, adj[i][j]-numU);
    }
    fclose(fp);
}

void Graph::print_graph()
{
    for (intV i=0; i<numU; i++)
    {
        for (intV j=0; j<deg[i]; j++)
            printf("%d %d\n", i, adj[i][j]);
    }
}

void Graph::get_labels(std::vector<intV> &labels, int side)
{
    if (side==0)
        parallel_vec_copy(labels, uLabels); 
    else
        parallel_vec_copy(labels, vLabels);
}

#endif
