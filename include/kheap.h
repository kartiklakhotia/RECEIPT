#include <vector>
#include <algorithm>
#include <limits>


// k-way heap implementation
template<typename V, typename K, unsigned k = 4> class KHeap {
    size_t size;                    // current number of elements in the heap
    size_t invalid;                 // invalid position
    std::vector<V> heap;            // array to store the heap
    std::vector<size_t> heap_pos;   // heap_pos[v] is index of v in the heap
    std::vector<K> key;             // key[v] is a vertex key

    // Swap elements with positions i and j
    void swap(size_t i, size_t j) {
        std::swap(heap_pos[heap[i]], heap_pos[heap[j]]);
        std::swap(heap[i], heap[j]);
    }
    // Get position of heap[i]'s smallest child
    size_t kid(size_t i) const {
        size_t c0 = i*k+1, m = 0;
        for (size_t j = 1; j < k && c0 + j < size; ++j)
            if (key[heap[c0 + j]] < key[heap[c0 + m]]) m = j;
        return c0 + m;
    }
    // Move heap[i] up or down
    void fixup(size_t i) {
        size_t c;
        while(i*k + 1 < size && key[heap[i]] > key[heap[(c=kid(i))]]) { swap(i, c); i = c; }
        while(i > 0 && key[heap[i]] < key[heap[(i-1)/k]]) { swap(i, (i-1)/k); i = (i-1)/k; }
    }

public:
    KHeap(size_t n) : size(0), invalid(std::numeric_limits<size_t>::max()), heap(n), heap_pos(n, invalid), key(n) {}

    bool empty() const { return size == 0; }        // Is heap empty?
    std::pair<V, K> top() const { return std::make_pair(heap[0], key[heap[0]]); }               // Get minimum elenment (heap should be non-empty)
    V pop() { std::pair<V, K> kvp = top(); V v = kvp.first; extract(v); return v; }  // Get minimum element and extract it from the queue

    // Update v's key to new_key
    void update(V v, K kk) {
        if (heap_pos[v] == invalid) { heap_pos[v] = size++; heap[heap_pos[v]] = v; }
        key[v] = kk;
        fixup(heap_pos[v]);
    }

    // Extract v from the queue (if v is present)
    void extract(V v) {
        if (heap_pos[v] == invalid) return;
        if (heap_pos[v] < --size) {
            size_t pos = heap_pos[v];
            swap(pos, size);
            fixup(pos);
        }
        heap_pos[v] = invalid;
    }

    // Clear heap
    void clear() {
        for (size_t i = 0; i < size; ++i) heap_pos[heap[i]] = invalid;
        size = 0;
    }
};

