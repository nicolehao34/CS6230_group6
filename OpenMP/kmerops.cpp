#include "kmerops.hpp"
#include "dnaseq.hpp"
#include "timer.hpp"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <omp.h>
#include <vector>

// ForeachKmer is a function that takes a DnaBuffer object and a KmerHandler object.
// It iterates over the reads in the DnaBuffer object, extracts kmers from the reads, 
// and calls the KmerHandler object for each kmer.
template <typename KmerHandler>
void ForeachKmer(const DnaBuffer& myreads, KmerHandler& handler)
{
    for (size_t i = 0; i < myreads.size(); ++i)
    {
        if (myreads[i].size() < KMER_SIZE)
            continue;

        std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);

        for (auto meritr = repmers.begin(); meritr != repmers.end(); ++meritr)
        {
            handler(*meritr);
        }
    }
}

// This function takes a kmer and the number of tasks and returns the owner of the kmer.
// The owner is calculated by hashing the kmer and then dividing the hash by the number of tasks.
// It is not used in the serial version of count_kmer,
// However it may come in handy when parallelizing the code.
int GetKmerOwner(const TKmer& kmer, int ntasks) {
    uint64_t myhash = kmer.GetHash();
    double range = static_cast<double>(myhash) * static_cast<double>(ntasks);
    size_t owner = range / std::numeric_limits<uint64_t>::max();
    assert(owner >= 0 && owner < static_cast<size_t>(ntasks));
    return static_cast<int>(owner);
}

// -----------------------------------------------------------------------------------------------------------------------

// Helper function: parallel merge sort
namespace {
    static const size_t threshold = 1000;
    void parallel_merge_sort(KmerSeedBucket::iterator begin, KmerSeedBucket::iterator end)
    {
        size_t len = end - begin;
        if (len < threshold)
        {
            std::sort(begin, end);
        }
        else
        {
            auto mid = begin + len/2;
            #pragma omp task shared(begin, mid)
            {
                parallel_merge_sort(begin, mid);
            }
            #pragma omp task shared(mid, end)
            {
                parallel_merge_sort(mid, end);
            }
            #pragma omp taskwait

            std::vector<KmerSeedStruct> temp(begin, end);
            std::merge(temp.begin(), temp.begin() + (len/2),
                       temp.begin() + (len/2), temp.end(),
                       begin);
        }
    }
}

std::unique_ptr<KmerList> count_kmer(const DnaBuffer& myreads)
{
    Timer timer;
    int num_threads = omp_get_max_threads();

    // Assigning buckets to threads
    std::vector< std::vector<KmerSeedBucket> > local_buckets(num_threads, std::vector<KmerSeedBucket>(num_threads));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& myBuckets = local_buckets[tid];

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < myreads.size(); ++i)
        {
            if (myreads[i].size() < KMER_SIZE)
                continue;

            std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
            for (const auto& kmer : repmers)
            {
                int owner = GetKmerOwner(kmer, num_threads);
                myBuckets[owner].push_back(KmerSeedStruct(kmer));
            }
        }
    }

    // Pre-allocate memory for reserve
    size_t total_seeds = 0;
    for (int owner = 0; owner < num_threads; ++owner)
    {
        for (int tid = 0; tid < num_threads; ++tid)
        {
            total_seeds += local_buckets[tid][owner].size();
        }
    }

    // Merge buckets into global bucket
    KmerSeedBucket* kmerseeds = new KmerSeedBucket;
    kmerseeds->reserve(total_seeds);
    for (int owner = 0; owner < num_threads; ++owner)
    {
        for (int tid = 0; tid < num_threads; ++tid)
        {
            auto& bucket = local_buckets[tid][owner];
            kmerseeds->insert(kmerseeds->end(), bucket.begin(), bucket.end());
        }
    }

    // Parallel merge sort
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            parallel_merge_sort(kmerseeds->begin(), kmerseeds->end());
        }
    }

    // Count kmers
    KmerList* kmerlist = new KmerList;
    if (!kmerseeds->empty())
    {
        TKmer last_mer = (*kmerseeds)[0].kmer;
        uint64_t cur_kmer_cnt = 1;

        for (size_t idx = 1; idx < kmerseeds->size(); idx++) 
        {
            TKmer cur_mer = (*kmerseeds)[idx].kmer;
            if (cur_mer == last_mer)
            {
                cur_kmer_cnt++;
            }
            else
            {
                kmerlist->push_back(KmerListEntry(last_mer, static_cast<int>(cur_kmer_cnt)));
                cur_kmer_cnt = 1;
                last_mer = cur_mer;
            }
        }
        
        // Last kmer
        kmerlist->push_back(KmerListEntry(last_mer, static_cast<int>(cur_kmer_cnt)));
    }

    delete kmerseeds;
    return std::unique_ptr<KmerList>(kmerlist);
}