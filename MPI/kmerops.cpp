#include "kmerops.hpp"
#include "dnaseq.hpp"
#include "timer.hpp"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <mpi.h>


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

// This function takes a DnaBuffer object, counts the kmers in it, and returns a KmerList object.
// The current implementation is sequential and only uses one thread.
// Your task is to parallelize this function using OpenMP.
std::unique_ptr<KmerList> count_kmer_sort(const DnaBuffer& myreads)
{
    Timer timer;

    // Step1: Parse the kmers from the reads
    KmerSeedBucket* kmerseeds = new KmerSeedBucket;
    KmerParserHandler handler(*kmerseeds);
    ForeachKmer(myreads, handler);

    // Step2: Sort the kmers
    std::sort(kmerseeds->begin(), kmerseeds->end());

    // Step3: Count the kmers
    uint64_t valid_kmer = 0;
    KmerList* kmerlist = new KmerList();
    
    TKmer last_mer = (*kmerseeds)[0].kmer;
    uint64_t cur_kmer_cnt = 1;

    for(size_t idx = 1; idx < (*kmerseeds).size(); idx++) 
    {
        TKmer cur_mer = (*kmerseeds)[idx].kmer;
        if (cur_mer == last_mer) {
            cur_kmer_cnt++;
        } else {
            // the next kmer has different value from the current one
            kmerlist->push_back(KmerListEntry());
            KmerListEntry& entry    = kmerlist->back();
            TKmer& kmer             = std::get<0>(entry);
            int& count              = std::get<1>(entry);

            count = cur_kmer_cnt;
            kmer = last_mer;
            valid_kmer++;

            cur_kmer_cnt = 1;
            last_mer = cur_mer;
        }
    }

    // deal with the last kmer
    kmerlist->push_back(KmerListEntry());
    KmerListEntry& entry         = kmerlist->back();
    TKmer& kmer             = std::get<0>(entry);
    int& count              = std::get<1>(entry);

    count = cur_kmer_cnt;
    kmer = last_mer;
    valid_kmer++;

    // Step4: Clean up
    delete kmerseeds;
    
    return std::unique_ptr<KmerList>(kmerlist);
}

// Another implementation using unordered_map
std::unique_ptr<KmerList> count_kmer(const DnaBuffer& myreads)
{
    Timer timer;

    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> kmermap;
    KmerHashmapHandler handler(&kmermap);
    ForeachKmer(myreads, handler);

    KmerList* kmerlist = new KmerList();
    for (auto& entry : kmermap)
    {
        auto kmer = std::get<0>(entry);
        auto count = std::get<1>(entry);
        kmerlist->push_back(KmerListEntry(kmer.kmer, count));
    }

    return std::unique_ptr<KmerList>(kmerlist);
}

#include "kmerops.hpp"
#include "dnaseq.hpp"
#include "timer.hpp"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <mpi.h>

std::unique_ptr<KmerList> count_kmer_mpi(const DnaBuffer& myreads) {
    Timer timer;

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmer_map;
    KmerHashmapHandler handler(&local_kmer_map);
    ForeachKmer(myreads, handler);

    // DEBUG print local k-mer count before distribution
    printf("Rank %d: Local k-mer count before exchange: %lu\n", rank, local_kmer_map.size());

    // Distribute k-mers among processes based on GetKmerOwner()
    std::vector<std::vector<KmerListEntry>> send_buffers(num_procs);
    for (const auto& entry : local_kmer_map) {
        int owner = GetKmerOwner(entry.first.kmer, num_procs);
        if (entry.second <= 0) {
            printf("Warning: Rank %d found local k-mer %llu with count = %d before sending!\n",
                   rank, entry.first.kmer.GetHash(), entry.second);
        }
        send_buffers[owner].emplace_back(entry.first.kmer, entry.second);
    }

    // exchange sizes with MPI_Alltoall
    std::vector<int> send_sizes(num_procs, 0), recv_sizes(num_procs);
    for (int i = 0; i < num_procs; ++i) {
        send_sizes[i] = send_buffers[i].size();
    }

    MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // preparocess data for MPI_Alltoallv
    std::vector<int> send_displs(num_procs, 0), recv_displs(num_procs, 0);
    int total_recv = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
    
    std::vector<KmerListEntry> send_data, recv_data(total_recv);
    
    for (int i = 0; i < num_procs; ++i) {
        send_displs[i] = (i == 0) ? 0 : send_displs[i - 1] + send_sizes[i - 1];
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recv_sizes[i - 1];
        send_data.insert(send_data.end(), send_buffers[i].begin(), send_buffers[i].end());
    }

    // DEBUG print sample send data before communication
    if (!send_data.empty()) {
        printf("Rank %d: Sample k-mer before MPI_Alltoallv: (Kmer = %llu, Count = %d)\n",
               rank, std::get<0>(send_data[0]).GetHash(), std::get<1>(send_data[0]));
    }

    // call MPI_Alltoallv
    MPI_Alltoallv(send_data.data(), send_sizes.data(), send_displs.data(), MPI_BYTE,
                  recv_data.data(), recv_sizes.data(), recv_displs.data(), MPI_BYTE, MPI_COMM_WORLD);

    // DEBUG Print number of received k-mers
    printf("Rank %d: Received k-mers count: %lu\n", rank, recv_data.size());

    // DEBUG print sample received data
    if (!recv_data.empty()) {
        printf("Rank %d: Sample k-mer received: (Kmer = %llu, Count = %d)\n",
               rank, std::get<0>(recv_data[0]).GetHash(), std::get<1>(recv_data[0]));
    }

    // Merge received k-mers 
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> final_kmer_map;

    // DEBUG Print received k-mers before merging
    for (size_t i = 0; i < recv_data.size(); i++) {
        printf("Rank %d: Raw received data %lu -> Kmer = %llu, Count = %d\n",
            rank, i, std::get<0>(recv_data[i]).GetHash(), std::get<1>(recv_data[i]));
    }

    for (const auto& entry : recv_data) {
        TKmer kmer = std::get<0>(entry);
        int count = std::get<1>(entry);

        if (count <= 0) {
            printf("Warning: Rank %d received k-mer %llu with invalid count = %d!\n",
                rank, kmer.GetHash(), count);
        }

        final_kmer_map[kmer] += count;  // Merge k-mer counts
    }

    // DEBUG Check for invalid final k-mer counts
    for (const auto& entry : final_kmer_map) {
        if (entry.second <= 0) {
            printf("Warning: Rank %d encountered k-mer %llu with count = %d after merging!\n",
                rank, entry.first.kmer.GetHash(), entry.second);
        }
    }

    // Convert to list
    std::vector<KmerListEntry> final_kmers;
    for (const auto& entry : final_kmer_map) {
        if (entry.second > 0) {
            final_kmers.emplace_back(entry.first.kmer, entry.second);
        } else {
            printf("Warning: Rank %d filtering out k-mer %llu with invalid count = %d before final output!\n",
                   rank, entry.first.kmer.GetHash(), entry.second);
        }
    }

    printf("Rank %d: Final K-mer count: %lu\n", rank, final_kmers.size());

    return std::make_unique<KmerList>(final_kmers.begin(), final_kmers.end());
}
