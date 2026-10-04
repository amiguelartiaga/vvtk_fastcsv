#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fastcsv {

// Skip columns are scanned past and receive no target buffer.
enum class Type { Int64, Int32, Float64, Float32, String, Skip };

bool is_numeric(Type t) noexcept;
std::size_t item_size(Type t) noexcept;
Type parse_type(std::string_view s);
std::string_view type_name(Type t) noexcept;

struct SchemaColumn {
    std::string name;
    Type type;
};

// Final-storage view. The caller owns every pointer and guarantees that it is
// large enough for the metadata supplied to read_strict_csv_into().
//
// Numeric columns point at `rows` elements spaced `stride` elements apart, so a
// column may be a slice of a row-major matrix (stride == number of columns).
// String columns receive rows+1 uint64 offsets plus exactly the sum of the
// per-chunk string byte counts. The parser writes directly into these buffers;
// there is no intermediate copy.
struct ReadColumnView {
    Type type = Type::Int64;
    void* numeric = nullptr;          // int64_t*, int32_t*, double* or float*
    std::size_t stride = 1;           // element stride for numeric columns
    std::uint64_t* string_offsets = nullptr;
    char* string_bytes = nullptr;
};

// Result of the inspection pass. Chunk vectors describe a parallel plan that
// read_strict_csv_into() accepts directly: `chunk_offsets` are interior row
// boundaries (file offsets), `chunk_row_counts` has one entry per chunk and
// `chunk_string_bytes[chunk][string_ordinal]` gives packed byte counts.
struct InspectResult {
    std::size_t rows = 0;
    std::uint64_t input_bytes = 0;
    std::uint64_t data_offset = 0;    // first byte after the header
    std::vector<std::uint64_t> chunk_offsets;
    std::vector<std::uint64_t> chunk_row_counts;
    std::vector<std::vector<std::uint64_t>> chunk_string_bytes;
    std::vector<std::uint64_t> string_bytes;  // totals, string columns only
};

struct ReadStats {
    std::size_t rows = 0;
    std::size_t chunks = 0;
    std::size_t workers = 0;
    std::uint64_t input_bytes = 0;
};

struct ReadOptions {
    bool has_header = true;
    char delimiter = ',';
    std::size_t threads = 1;              // 0 = hardware concurrency
    bool sequential_io_hint = true;
    bool empty_float_is_nan = true;       // "" in a float column parses as NaN
};

// One preparatory pass for strict files that do not have an acceleration
// sidecar. It splits the data into byte-balanced chunks aligned to row
// boundaries and, in parallel, counts rows and exact UTF-8 bytes per string
// column. Canonical files written by FastCSV carry this plan in their sidecar.
InspectResult inspect_strict_csv(const std::string& path,
                                 const std::vector<SchemaColumn>& schema,
                                 const ReadOptions& options,
                                 std::size_t target_chunk_bytes = 4 * 1024 * 1024);

// Parse directly into caller-owned final buffers. Safe parallel parsing requires
// chunk offsets and row counts. For packed strings, per-chunk string byte counts
// make every worker's destination byte slice known in advance.
ReadStats read_strict_csv_into(const std::string& path,
                               const std::vector<SchemaColumn>& schema,
                               const std::vector<ReadColumnView>& targets,
                               const ReadOptions& options,
                               std::size_t expected_rows,
                               const std::vector<std::uint64_t>& chunk_offsets = {},
                               const std::vector<std::uint64_t>& chunk_row_counts = {},
                               const std::vector<std::vector<std::uint64_t>>& chunk_string_bytes = {});

// Homogeneous numeric matrix fast path. Output is row-major [rows, columns]
// and is written directly into the caller's allocation (NumPy or torch).
ReadStats read_strict_numeric_matrix_into(const std::string& path,
                                          Type type,
                                          std::size_t columns,
                                          void* output,
                                          const ReadOptions& options,
                                          std::size_t expected_rows,
                                          const std::vector<std::uint64_t>& chunk_offsets = {},
                                          const std::vector<std::uint64_t>& chunk_row_counts = {},
                                          const std::vector<std::string>& names = {});

// Reads the raw header line (without the trailing newline) or "" if none.
std::string read_first_line(const std::string& path, std::size_t max_bytes = 1 << 20);

bool simd_scanner_compiled() noexcept;
std::string simd_scanner_runtime();
bool fast_float_compiled() noexcept;

}  // namespace fastcsv
