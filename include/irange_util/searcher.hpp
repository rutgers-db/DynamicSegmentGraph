#pragma once 

#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include "memory.hpp"

namespace searcher
{
    // Bitset structure to manage flags for a set of elements, optimized for memory alignment.
    template <typename Block = uint64_t>
    struct Bitset
    {
    private:
        constexpr static int block_size = sizeof(Block) * 8; // Number of bits per block.
        int nbytes; // Total bytes needed to store all bits.
        Block *data; // Pointer to the memory storing the bitset.

    public:
        // Constructor: Allocates and initializes memory for the bitset.
        explicit Bitset(int n)
            : nbytes((n + block_size - 1) / block_size * sizeof(Block)), // Calculate bytes needed.
              data(static_cast<uint64_t*>(memory::align_mm<64>(nbytes))) // Allocate aligned memory.
        {
            std::memset(data, 0, nbytes); // Initialize all bits to 0.
        }

        // Destructor: Frees allocated memory.
        ~Bitset() { free(data); }

        // Sets the bit at position `i`.
        void set(int i)
        {
            data[i / block_size] |= (Block(1) << (i & (block_size - 1))); // Set specific bit within the block.
        }

        // Retrieves the value of the bit at position `i`.
        bool get(int i)
        {
            return (data[i / block_size] >> (i & (block_size - 1))) & 1; // Shift and mask to get the bit.
        }

        // Gets the memory address of the block containing the bit at index `i`.
        void *block_address(int i) { return data + i / block_size; }
    };
}
