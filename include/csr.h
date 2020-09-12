/*******************************************************************************
    DATA structures to store projection graph
    Functions to create projection graph
    Use CSR format to avoid individual memory allocations for each adj. list
    Functions in this file are SEQUENTIAL as each partition is processed by 
    a single thread
*******************************************************************************/
#ifndef MY_CSR
#define MY_CSR

#include "graph.h"

/*********************************************************
Basic CSR structure.
A thread will instantiate it once and use
it for all the partitions that it processes.
*********************************************************/
typedef struct myCSR
{
    intV numU;
    intV numV;
    intV numT;
    intE numE;
    std::vector<intE> VI;
    std::vector<intV> deg; 
    std::vector<intV> EI;
    std::vector<intB> EW;
} myCSR;

/******************************************************
create CSR from undeleted vertices
assuming only U vertices are deleted
WILL NOT MODIFY THE GRAPH OBJECT
Arguments:
1. G -> graph object
2. partId -> partitionId
3. partVertices -> list of vertices in the partition 
4. vtxToPart -> map from vertex to partition IDs
5. oG -> output CSR
******************************************************/
void create_part_csr(Graph &G, int partId, std::vector<intV> &partVertices, std::vector<int> &vtxToPart, myCSR &oG)
{

    oG.numU = G.numU;
    oG.numV = G.numV;
    oG.numT = G.numT;

    //create offset array
    if (oG.VI.size() != G.numT+1)
        oG.VI.resize(G.numT+1);

    //reset the degrees
    if (oG.deg.size() != G.numT)
        oG.deg.resize(G.numT, 0);
    else
    {
        for (intV i=0; i<G.numT; i++)
            oG.deg[i] = 0;
    }

    //compute number of edges
    intE numEdges = 0;
    for (auto x : partVertices)
        numEdges += G.deg[x];
    numEdges = 2*numEdges; //undirected edges
    oG.numE = numEdges;

    //allocate space for edges
    if (oG.EI.size() != numEdges)
    {
        oG.EI.clear();
        oG.EI.resize(numEdges);
    }

    //compute degrees and prefix sum for offsets and insert edges
    //for 'U' vertices
    //compute degrees
    for(intV i=0; i<G.numU; i++)
    {
        intV v = G.uLabels[i];
        if (vtxToPart[v]!=partId) //vertices not in this partition
            continue;
        else
        {
            std::vector<intV> &neighList = G.get_neigh(v, oG.deg[v]);
            for (intV j=0; j<oG.deg[v]; j++)
                oG.deg[neighList[j]]++; //increase degree of the neighbor
        }
    }

    //compute offsets
    oG.VI[0] = 0;
    for (intV i=0; i<G.numT; i++)
    {
        oG.VI[i+1] = oG.deg[i] + oG.VI[i];
        oG.deg[i] = 0;
    } 

    //Insert edges (undirected)
    //edges will be sorted by default
    for (intV i=0; i<G.numU; i++)
    {
        intV u = G.uLabels[i];
        if (vtxToPart[u]!=partId) continue;
        std::vector<intV> &neighList = G.get_neigh(u, oG.deg[u]);
        for (intV j=0; j<oG.deg[u]; j++)
        {
            intV v = neighList[j];
            oG.EI[oG.VI[u] + j] = v; 
            oG.EI[oG.VI[v] + oG.deg[v]] = u;
            oG.deg[v]++;
        } 
    }
}


void delete_edges_csr(Graph &G, myCSR &oG)
{
    for (intV i=0; i<G.numU; i++)
    {
        intV u = G.uLabels[i];
        if(G.is_deleted(u)) oG.deg[u] = 0;
    }
    for (intV i=0; i<G.numV; i++)
    {
        intV v = G.vLabels[i];
        intV start = oG.VI[v];
        intV end =  oG.VI[v+1];
        while(start != end)
        {
            if (G.is_deleted(oG.EI[start]))
            {
                end--;
                std::swap(oG.EI[start], oG.EI[end]);
            }
            else
                start++;
        }
        oG.deg[v] = end - oG.VI[v];
        std::sort(oG.EI.begin() + oG.VI[v], oG.EI.begin() + oG.VI[v] + oG.deg[v]);
    }
}

void restore_edges_csr(myCSR &oG)
{
    for (intV i=0; i<oG.numT; i++)
    {
        oG.deg[i] = oG.VI[i+1] - oG.VI[i];
        std::sort(oG.EI.begin() + oG.VI[i], oG.EI.begin() + oG.VI[i] + oG.deg[i]);
    }
}

#endif
