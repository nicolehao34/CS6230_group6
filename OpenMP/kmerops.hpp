#ifndef KMEROPS_HPP
#define KMEROPS_HPP

#include "kmer.hpp"
#include "timer.hpp"
#include "dnaseq.hpp"
#include "dnabuffer.hpp"
#include <unordered_map>
#include <omp.h>  // Include OpenMP

// KmerList is a list of tuples, where the first element is a kmer and the second element is its frequency.
// Please DO NOT change the definition of KmerList and KmerListEntry.
typedef std::tuple<TKmer, int> KmerListEntry;
typedef std::vector<KmerListEntry> KmerList;

// KmerSeedStruct is a simple wrapper struct that contains a kmer.
struct KmerSeedStruct{
    TKmer kmer; 

    KmerSeedStruct(TKmer kmer) : kmer(kmer) {};
    KmerSeedStruct(const KmerSeedStruct& o) : kmer(o.kmer) {};
    KmerSeedStruct(KmerSeedStruct&& o) : kmer(std::move(o.kmer)) {};
    KmerSeedStruct() {};

    int GetByte(int &i) const { return kmer.getByte(i); }
    bool operator<(const KmerSeedStruct& o) const { return kmer < o.kmer; }
    bool operator==(const KmerSeedStruct& o) const { return kmer == o.kmer; }
    bool operator!=(const KmerSeedStruct& o) const { return kmer != o.kmer; }
    KmerSeedStruct& operator=(const KmerSeedStruct& o) {
        kmer = o.kmer;
        return *this;
    }
};
typedef std::vector<KmerSeedStruct> KmerSeedBucket;

// KmerSeedHash is a hash function for KmerSeedStruct.
struct KmerSeedHash {
    size_t operator()(const KmerSeedStruct& kmerseed) const {
        return kmerseed.kmer.GetHash();
    }
};

// KmerParserHandler is a function that takes a kmer and stores it in a vector.
// When implementing a parallel version of count_kmer, you may want to 
// create your own k-mer parser handler.
struct KmerParserHandler {
    std::vector<KmerSeedStruct>& kmerseeds;

    KmerParserHandler(std::vector<KmerSeedStruct>& kmerseeds) : kmerseeds(kmerseeds) {}

    void operator()(const TKmer& kmer) {
        kmerseeds.emplace_back(kmer);
    }
};

// KmerHashmapHandler is a function that takes a kmer and stores it in a hashmap.
struct KmerHashmapHandler {
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash>* kmermap;

    KmerHashmapHandler(std::unordered_map<KmerSeedStruct, int, KmerSeedHash>* kmermap) : kmermap(kmermap) {}

    void operator()(const TKmer& kmer) {
        KmerSeedStruct kmerseed(kmer);
        auto it = kmermap->find(kmerseed);
        if (it == kmermap->end()) {
            (*kmermap)[kmerseed] = 1;
        } else {
            it->second++;
        }
    }

};

int GetKmerOwner(const TKmer& kmer, int ntasks);

std::unique_ptr<KmerList> count_kmer(const DnaBuffer& myreads);

std::unique_ptr<KmerList> count_kmer_hashmap(const DnaBuffer& myreads);

#endif // KMEROPS_HPP