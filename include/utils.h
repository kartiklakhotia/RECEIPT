#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <assert.h>
#include <utility>
#include <omp.h>
#include <string>
#include <cstring>
#include <ctime>
#include <map>
#include <atomic>
#include <boost/sort/sort.hpp>

unsigned int NUM_THREADS = 1;
size_t SERIAL_TO_PAR_THRESH = NUM_THREADS*100;

template <typename T1, typename T2>
class valCompareGreater
{
    const std::vector<T2> &value_vector;

    public:
    valCompareGreater(const std::vector<T2> &val_vec):
        value_vector(val_vec) {}

    bool operator() (const T1 &i1, const T1 &i2) const
    {
        return value_vector[i1] > value_vector[i2];
    }
};

template <typename T1, typename T2>
class valCompareLesser
{
    const std::vector<T2> &value_vector;

    public:
    valCompareLesser(const std::vector<T2> &val_vec):
        value_vector(val_vec) {}

    bool operator() (const T1 &i1, const T1 &i2) const
    {
        return value_vector[i1] < value_vector[i2];
    }
};

template <typename T>
void free_vec (std::vector<T> &vec)
{
    std::vector<T> dummy;
    vec.swap(dummy);
}

template <typename Tout, typename Tin>
Tout parallel_reduce (std::vector<Tin> &in)
{
    Tout retVal = 0;
    size_t vecSize = in.size();
    #pragma omp parallel for reduction(+:retVal) 
    for (size_t i=0; i<vecSize; i++)
        retVal += in[i];
    return retVal;
}


//exclusive sum
template <typename T1, typename T2>
void serial_prefix_sum (std::vector<T1> &output, std::vector<T2> &input)
{
    if (output.size() != (input.size() + 1))
        output.resize(input.size() + 1);
    size_t start = 0;
    size_t end = input.size();
    
    output[0] = 0;
    for (size_t i=start; i<end; i++)
        output[i+1] = output[i] + input[i];
}

//exclusive sum
template <typename T1, typename T2>
void serial_prefix_sum_inclusive (std::vector<T1> &output, std::vector<T2> &input)
{
    if (output.size() != (input.size()))
        output.resize(input.size());
    size_t start = 0;
    size_t end = input.size();
    
    output[0] = input[0];
    for (size_t i=start+1; i<end; i++)
        output[i] = output[i-1] + input[i];
}

//exclusive sum
template <typename T1, typename T2>
void parallel_prefix_sum (std::vector<T1> &output, std::vector<T2> &input)
{
    if (input.size() < 50*NUM_THREADS)
    {
        serial_prefix_sum(output, input);
        return;
    }
    std::vector<T1> scan(input.size()+1);
    std::vector<T1> indAcc(NUM_THREADS+1, 0);
    size_t BS = (input.size()-1)/NUM_THREADS + 1;
    scan[0] = 0;
    #pragma omp parallel 
    {
        size_t tid = omp_get_thread_num();
        size_t start = tid*BS+1;
        size_t end = (tid+1)*BS+1;
        end = (end > input.size()+1) ? input.size()+1 : end;
        
        if (start <= end)
            scan[start] = input[start-1];
        for (size_t i=start+1; i<end; i++)
            scan[i] = scan[i-1] + input[i-1];
        indAcc[tid+1] = scan[end-1];
        #pragma omp barrier
        #pragma omp single
        {
            for (size_t i=1; i<NUM_THREADS+1; i++)
                indAcc[i] += indAcc[i-1];
        }
        for (size_t i=start; i<end; i++)
            scan[i] += indAcc[tid];
    }     
    output.swap(scan);
}

//inclusive sum
template <typename T1, typename T2>
void parallel_prefix_sum_inclusive(std::vector<T1> &output, std::vector<T2> &input)
{
    if (input.size() < 50*NUM_THREADS)
    {
        serial_prefix_sum_inclusive(output, input);
        return;
    }
    
    std::vector<T1> scan(input.size());
    std::vector<T1> indAcc(NUM_THREADS+1, 0); 
    size_t BS = (input.size()-1)/NUM_THREADS + 1;
    #pragma omp parallel 
    {
        size_t tid = omp_get_thread_num();
        size_t start = tid*BS;
        size_t end = std::min((tid+1)*BS, (size_t)input.size());
        if (start <= end)
            scan[start] = input[start];
        for (size_t i=start+1; i<end; i++)
            scan[i] = scan[i-1] + input[i];
        indAcc[tid+1] = scan[end-1];
        #pragma omp barrier
        #pragma omp single
        {
            for (size_t i=1; i<NUM_THREADS+1; i++)
                indAcc[i] += indAcc[i-1];
        }
        for (size_t i=start; i<end; i++)
            scan[i] += indAcc[tid];
    }
    output.swap(scan);
}

template <typename T>
T parallel_find_max (std::vector<T> &vec)
{
    T maxVal = vec[0];
    size_t vec_size = vec.size();
    #pragma omp parallel for reduction (max:maxVal) 
    for (size_t i=0; i<vec_size; i++)
        maxVal = std::max(vec[i], maxVal);
    return maxVal; 
}

template <typename T>
T serial_find_max (std::vector<T> &vec)
{
    T maxVal = vec[0];
    size_t vec_size = vec.size();
    for (size_t i=0; i<vec_size; i++)
        maxVal = std::max(vec[i], maxVal);
    return maxVal; 
}

template <typename T1, typename T2>
void serial_sort_kv (std::vector<T1> &idxVec, std::vector<T2> &valueVec)
{
    std::stable_sort(idxVec.begin(), idxVec.end(), valCompareGreater<T1,T2>(valueVec));  
}

//sort keys based on the values. Note that this does not change the value array
//The value array is indexed by the keys
template <typename T1, typename T2>
void parallel_sort_kv (std::vector<T1> &idxVec, std::vector<T2> &valueVec)
{
    boost::sort::parallel_stable_sort(idxVec.begin(), idxVec.end(), valCompareGreater<T1,T2>(valueVec), NUM_THREADS);  
}

//sort keys based on the values. Note that this does not change the value array
//The value array is indexed by the keys
template <typename T1, typename T2>
void parallel_sort_kv_increasing (std::vector<T1> &idxVec, std::vector<T2> &valueVec)
{
    boost::sort::parallel_stable_sort(idxVec.begin(), idxVec.end(), valCompareLesser<T1,T2>(valueVec), NUM_THREADS);  
}

template <typename T1, class compare>
void parallel_sort_indices (std::vector<T1> &idxVec, compare comp)
{
    boost::sort::parallel_stable_sort(idxVec.begin(), idxVec.end(), comp, NUM_THREADS);  
}

template <typename T>
void invertMap (std::vector<T> &ipMap, std::vector<T> &opMap)
{
    opMap.resize(ipMap.size());
    #pragma omp parallel for 
    for (size_t i=0; i<ipMap.size(); i++)
        opMap[ipMap[i]] = i;
}

template <typename T1, typename T2>
void reorderArr (std::vector<T1> &ipArr, std::vector<T2> &newOrder)
{
    std::vector<T1> opArr (ipArr.size());
    #pragma omp parallel for 
    for (size_t i=0; i<ipArr.size(); i++)
        opArr[newOrder[i]] = ipArr[i];
    opArr.swap(ipArr);
}

template <typename T1, typename T2>
inline T1 choose2(T2 n)
{
    if (n < 2) return 0;
    T1 retVal = (T1)n;
    retVal = retVal*(retVal-1);
    return retVal/2;
}

template <typename T>
void print_list_horizontal(std::vector<T> &list)
{
    std::cout<<"printing list : ";
    for (auto x:list)
        std::cout << x << ", ";
    std::cout<<std::endl;
}

template <typename T>
void print_list_vertical(std::vector<T> &list)
{
    std::cout<<"printing list : ";
    for (auto x:list)
        std::cout << x << std::endl;
}


//nET - data type for number of elements, for eg. intV for vertices
template <typename T, typename nET>
nET sequential_compact(std::vector<T> &in, std::vector<uint8_t> &keep, std::vector<T> &out)
{
    nET numKeep = 0;
    for (nET i=0; i<in.size(); i++)
        numKeep += keep[i];
    if (out.size() < numKeep)
    {
        out.clear();
        out.resize(numKeep);
    }
    numKeep = 0;
    for (nET i=0; i<in.size(); i++)
    {
        if (keep[i])
            out[numKeep++] = in[i];
    }
    return numKeep;
} 



//nET - data type for number of elements, for eg. intV for vertices
template <typename T, typename nET>
void parallel_compact(std::vector<T> &in, std::vector<uint8_t> &keep, std::vector<T> &out)
{
    if (in.size() < NUM_THREADS*16)
    {
        nET numKeep = sequential_compact<T, nET>(in, keep, out);
        return;
    }
    std::vector<nET> activePerThread(NUM_THREADS, 0);
    std::vector<nET> offset;
    nET partSize = (in.size()-1)/NUM_THREADS + 1;
    #pragma omp parallel 
    {
        size_t tid = omp_get_thread_num();
        nET start = partSize*tid;
        nET end = std::min(start+partSize, (nET)in.size());
        nET numActive = 0;
        for (nET i=start; i<end; i++)
            numActive += (keep[i] > 0);
        activePerThread[tid] = numActive;
        #pragma omp barrier
        #pragma omp single
        {
            serial_prefix_sum(offset, activePerThread);
            out.resize(offset[NUM_THREADS]);
        }
        #pragma omp barrier
        numActive = 0;
        for (nET i=start; i<end; i++)
        {
            if (keep[i])
            {
                out[offset[tid]+numActive] = in[i];
                numActive++;
            }
        }
    }
} 

//nET - data type for number of elements, for eg. intV for vertices
template <typename T, typename nET>
void parallel_compact_in_place(std::vector<T> &in, std::vector<uint8_t> &keep)
{
    std::vector<T> out;
    parallel_compact<T, nET>(in, keep, out);
    in.swap(out); 
} 



//overloaded function
//construct two separate vectors for elements with keep[i]==true and keep[i]==false
//nET - data type for number of elements, for eg. intV for vertices
template <typename T, typename nET>
void parallel_compact(std::vector<T> &in, std::vector<uint8_t> &keep, std::vector<T> &outTrue, std::vector<T> &outFalse)
{
    std::vector<nET> activePerThread(NUM_THREADS, 0);
    std::vector<nET> inActivePerThread(NUM_THREADS, 0);
    std::vector<nET> offsetTrue;
    std::vector<nET> offsetFalse;
    nET partSize = (in.size()-1)/NUM_THREADS + 1;
    #pragma omp parallel 
    {
        size_t tid = omp_get_thread_num();
        nET start = partSize*tid;
        nET end = std::min(start+partSize, (nET)in.size());
        nET range = (end>start) ? end-start : 0;
        nET numActive = 0;
        for (nET i=start; i<end; i++)
            numActive += (keep[i] > 0);
        activePerThread[tid] = numActive;
        inActivePerThread[tid] = range - numActive;
        #pragma omp barrier
        #pragma omp single
        {
            serial_prefix_sum(offsetTrue, activePerThread);
            serial_prefix_sum(offsetFalse, inActivePerThread);
            outTrue.resize(offsetTrue[NUM_THREADS]);
            outFalse.resize(offsetFalse[NUM_THREADS]);
        }
        #pragma omp barrier
        numActive = 0;
        nET numInActive = 0;
        for (nET i=start; i<end; i++)
        {
            if (keep[i])
            {
                outTrue[offsetTrue[tid]+numActive] = in[i];
                numActive++;
            }
            else
            {
                outFalse[offsetFalse[tid]+numInActive] = in[i];
                numInActive++;
            }
        }
    }
} 

//overloaded function
//compact input vector and return a vector of deleted elements
//nET - data type for number of elements, for eg. intV for vertices
template <typename T, typename nET>
void parallel_compact_in_place(std::vector<T> &in, std::vector<uint8_t> &keep, std::vector<T> &delElems)
{
    std::vector<T> out;
    parallel_compact<T, nET>(in, keep, out, delElems);
    in.swap(out); 
} 



//nET - data type for number of elements, for eg. intV for vertices
//use elements of "in" as indices/keys into "keep" vector
template <typename T, typename nET>
void serial_compact_kv(std::vector<T> &in, std::vector<uint8_t> &keep, std::vector<T> &out)
{
    int numActive = 0;
    for (nET i=0; i<in.size(); i++)
        numActive += (keep[in[i]] > 0);
    out.resize(numActive);
    numActive = 0;
    for (nET i=0; i<in.size(); i++)
    {
        if (keep[in[i]])
           out[numActive++] = in[i];
    }  
} 



//nET - data type for number of elements, for eg. intV for vertices
//use elements of "in" as indices/keys into "keep" vector
template <typename T, typename nET>
void parallel_compact_kv(std::vector<T> &in, std::vector<uint8_t> &keep, std::vector<T> &out)
{
    std::vector<nET> activePerThread(NUM_THREADS, 0);
    std::vector<nET> offset;
    nET partSize = (in.size()-1)/NUM_THREADS + 1;
    #pragma omp parallel 
    {
        size_t tid = omp_get_thread_num();
        nET start = partSize*tid;
        nET end = std::min(start+partSize, (nET)in.size());
        nET numActive = 0;
        for (nET i=start; i<end; i++)
            numActive += (keep[in[i]] > 0);
        activePerThread[tid] = numActive;
        #pragma omp barrier
        #pragma omp single
        {
            serial_prefix_sum(offset, activePerThread);
            out.resize(offset[NUM_THREADS]);
        }
        #pragma omp barrier
        numActive = 0;
        for (nET i=start; i<end; i++)
        {
            if (keep[in[i]])
            {
                out[offset[tid]+numActive] = in[i];
                numActive++;
            }
        }
    }
} 

//copy entire vector
template <typename T>
void parallel_vec_copy(std::vector<T> &op, std::vector<T> &in)
{
    op.resize(in.size());
    size_t numElems = in.size();
    #pragma omp parallel for 
    for (size_t i=0; i<numElems; i++)
        op[i] = in[i];
}

//copy given elements from input vector to output vector
template <typename T, typename eT>
void parallel_vec_elems_copy(std::vector<T> &op, std::vector<T> &in, std::vector<eT> &elems)
{
    eT numElems = elems.size(); 
    #pragma omp parallel for 
    for (eT i=0; i<numElems; i++)
        op[elems[i]] = in[elems[i]];
}
