#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fastcsv/reader.hpp"

namespace fastcsv {

struct PackedStringView {
    const std::uint64_t* offsets = nullptr;  // rows+1 offsets
    const char* bytes = nullptr;
};

// Source column. Numeric data is read with an element stride, so a column can
// be a slice of a row-major matrix (stride == number of columns) and the
// matrix is written straight from its own memory without transposing.
struct WriteColumnView {
    Type type = Type::Int64;
    const void* numeric = nullptr;
    std::size_t stride = 1;
    PackedStringView strings;
};

struct WriteOptions {
    char delimiter = ',';
    bool header = true;
    // Chunk metadata for the sidecar: a chunk closes at a block boundary once
    // it holds target_chunk_bytes, or exactly every chunk_rows rows when set.
    std::size_t chunk_rows = 0;
    std::size_t target_chunk_bytes = 4 * 1024 * 1024;
    std::size_t threads = 1;        // 0 = hardware concurrency
    std::size_t block_rows = 0;     // rows formatted per work item; 0 = auto
    int float_precision = -1;       // -1 = shortest round-trip representation
};

struct WriteResult {
    std::size_t rows = 0;
    std::uint64_t bytes_written = 0;
    std::size_t workers = 0;
    std::vector<std::uint64_t> chunk_offsets;      // starts of chunks 1..N-1
    std::vector<std::uint64_t> chunk_row_counts;   // N entries
    std::vector<std::uint64_t> chunk_byte_counts;  // N entries
    // Per chunk, bytes for string columns in schema string-column order.
    std::vector<std::vector<std::uint64_t>> chunk_string_bytes;
};

WriteResult write_canonical_csv(const std::string& path,
                                const std::vector<SchemaColumn>& schema,
                                const std::vector<WriteColumnView>& columns,
                                std::size_t rows,
                                const WriteOptions& options);

}  // namespace fastcsv
