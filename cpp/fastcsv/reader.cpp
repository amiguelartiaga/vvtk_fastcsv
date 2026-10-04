#include "fastcsv/reader.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && !defined(FASTCSV_DISABLE_SIMD) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define FASTCSV_X86_SIMD 1
#endif

#ifdef FASTCSV_HAS_FAST_FLOAT
#include <fast_float/fast_float.h>
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fastcsv {

bool is_numeric(Type t) noexcept { return t != Type::String && t != Type::Skip; }

std::size_t item_size(Type t) noexcept {
    switch (t) {
        case Type::Int64: return 8;
        case Type::Int32: return 4;
        case Type::Float64: return 8;
        case Type::Float32: return 4;
        case Type::String: return 0;
        case Type::Skip: return 0;
    }
    return 0;
}

Type parse_type(std::string_view s) {
    if (s == "int64") return Type::Int64;
    if (s == "int32") return Type::Int32;
    if (s == "float64") return Type::Float64;
    if (s == "float32") return Type::Float32;
    if (s == "string") return Type::String;
    if (s == "skip") return Type::Skip;
    throw std::invalid_argument("unsupported dtype: " + std::string(s));
}

std::string_view type_name(Type t) noexcept {
    switch (t) {
        case Type::Int64: return "int64";
        case Type::Int32: return "int32";
        case Type::Float64: return "float64";
        case Type::Float32: return "float32";
        case Type::String: return "string";
        case Type::Skip: return "skip";
    }
    return "?";
}

namespace {

using FindByteFn = const char* (*)(const char*, const char*, char);

enum class SimdMode { Scalar, SSE2, AVX2 };

// ---------------------------------------------------------------------------
// Memory-mapped input
// ---------------------------------------------------------------------------

struct MappedFile {
    const char* data = nullptr;
    std::size_t size = 0;
#ifndef _WIN32
    int fd = -1;
    bool mapped = false;
#else
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = nullptr;
    bool mapped = false;
#endif

    explicit MappedFile(const std::string& path, bool sequential_hint) {
        static const char empty = '\0';
#ifndef _WIN32
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("cannot open file: " + path);
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd); fd = -1;
            throw std::runtime_error("cannot stat file: " + path);
        }
        size = static_cast<std::size_t>(st.st_size);
        if (size == 0) {
            data = &empty;
            return;
        }
#if defined(POSIX_FADV_SEQUENTIAL)
        if (sequential_hint) (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
        void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            ::close(fd); fd = -1;
            throw std::runtime_error("mmap failed: " + path);
        }
        data = static_cast<const char*>(p);
        mapped = true;
#if defined(MADV_SEQUENTIAL)
        if (sequential_hint) (void)::madvise(p, size, MADV_SEQUENTIAL);
#endif
#else
        const DWORD flags = FILE_ATTRIBUTE_NORMAL | (sequential_hint ? FILE_FLAG_SEQUENTIAL_SCAN : 0);
        file_handle = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                    OPEN_EXISTING, flags, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE)
            throw std::runtime_error("cannot open file: " + path);
        LARGE_INTEGER bytes{};
        if (!::GetFileSizeEx(file_handle, &bytes) || bytes.QuadPart < 0) {
            ::CloseHandle(file_handle); file_handle = INVALID_HANDLE_VALUE;
            throw std::runtime_error("cannot stat file: " + path);
        }
        if (static_cast<unsigned long long>(bytes.QuadPart) >
            static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            ::CloseHandle(file_handle); file_handle = INVALID_HANDLE_VALUE;
            throw std::runtime_error("file is too large for this process address space");
        }
        size = static_cast<std::size_t>(bytes.QuadPart);
        if (size == 0) {
            data = &empty;
            return;
        }
        mapping_handle = ::CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_handle) {
            ::CloseHandle(file_handle); file_handle = INVALID_HANDLE_VALUE;
            throw std::runtime_error("CreateFileMapping failed: " + path);
        }
        void* p = ::MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
        if (!p) {
            ::CloseHandle(mapping_handle); mapping_handle = nullptr;
            ::CloseHandle(file_handle); file_handle = INVALID_HANDLE_VALUE;
            throw std::runtime_error("MapViewOfFile failed: " + path);
        }
        data = static_cast<const char*>(p);
        mapped = true;
#endif
    }

    ~MappedFile() {
#ifndef _WIN32
        if (mapped) ::munmap(const_cast<char*>(data), size);
        if (fd >= 0) ::close(fd);
#else
        if (mapped) ::UnmapViewOfFile(data);
        if (mapping_handle) ::CloseHandle(mapping_handle);
        if (file_handle != INVALID_HANDLE_VALUE) ::CloseHandle(file_handle);
#endif
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const char* begin() const noexcept { return data; }
    const char* end() const noexcept { return data + size; }
};

// ---------------------------------------------------------------------------
// Byte scanning: runtime-dispatched AVX2 / SSE2 / memchr
// ---------------------------------------------------------------------------

inline const char* find_byte_scalar(const char* p, const char* end, char needle) {
    if (p >= end) return end;
    const void* hit = std::memchr(p, static_cast<unsigned char>(needle), static_cast<std::size_t>(end - p));
    return hit ? static_cast<const char*>(hit) : end;
}

#if defined(FASTCSV_X86_SIMD)
__attribute__((target("sse2")))
const char* find_byte_sse2(const char* p, const char* end, char needle) {
    const __m128i target = _mm_set1_epi8(needle);
    while (end - p >= 16) {
        const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(block, target)));
        if (mask) return p + std::countr_zero(mask);
        p += 16;
    }
    return find_byte_scalar(p, end, needle);
}

__attribute__((target("avx2")))
const char* find_byte_avx2(const char* p, const char* end, char needle) {
    const __m256i target = _mm256_set1_epi8(needle);
    while (end - p >= 32) {
        const __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const unsigned mask = static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(block, target)));
        if (mask) return p + std::countr_zero(mask);
        p += 32;
    }
    return find_byte_sse2(p, end, needle);
}

__attribute__((target("sse2")))
std::size_t count_newlines_sse2(const char* begin, const char* end) {
    const __m128i target = _mm_set1_epi8('\n');
    const char* p = begin;
    std::size_t n = 0;
    while (end - p >= 16) {
        const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(block, target)));
        n += static_cast<std::size_t>(std::popcount(mask));
        p += 16;
    }
    while (p < end) n += (*p++ == '\n');
    return n;
}

__attribute__((target("avx2")))
std::size_t count_newlines_avx2(const char* begin, const char* end) {
    const __m256i target = _mm256_set1_epi8('\n');
    const char* p = begin;
    std::size_t n = 0;
    while (end - p >= 32) {
        const __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const unsigned mask = static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(block, target)));
        n += static_cast<std::size_t>(std::popcount(mask));
        p += 32;
    }
    while (p < end) n += (*p++ == '\n');
    return n;
}
#endif

SimdMode detect_simd_mode() noexcept {
#if defined(FASTCSV_X86_SIMD)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) return SimdMode::AVX2;
    if (__builtin_cpu_supports("sse2")) return SimdMode::SSE2;
#endif
    return SimdMode::Scalar;
}

SimdMode runtime_simd_mode() noexcept {
    static const SimdMode mode = detect_simd_mode();
    return mode;
}

FindByteFn resolved_find_byte() noexcept {
    static const FindByteFn fn = []() -> FindByteFn {
#if defined(FASTCSV_X86_SIMD)
        switch (runtime_simd_mode()) {
            case SimdMode::AVX2: return &find_byte_avx2;
            case SimdMode::SSE2: return &find_byte_sse2;
            default: break;
        }
#endif
        return &find_byte_scalar;
    }();
    return fn;
}

std::size_t count_newlines(const char* begin, const char* end) {
    if (begin >= end) return 0;
#if defined(FASTCSV_X86_SIMD)
    switch (runtime_simd_mode()) {
        case SimdMode::AVX2: return count_newlines_avx2(begin, end);
        case SimdMode::SSE2: return count_newlines_sse2(begin, end);
        default: break;
    }
#endif
    std::size_t n = 0;
    const char* p = begin;
    while (p < end) {
        const char* nl = find_byte_scalar(p, end, '\n');
        if (nl == end) break;
        ++n;
        p = nl + 1;
    }
    return n;
}

// Rows in [begin, end): newline count, plus one for an unterminated final row.
std::size_t count_rows(const char* begin, const char* end) {
    if (begin >= end) return 0;
    std::size_t n = count_newlines(begin, end);
    if (end[-1] != '\n') ++n;
    return n;
}

const char* data_begin_for(const MappedFile& file, bool has_header, FindByteFn finder) {
    const char* begin = file.begin();
    const char* end = file.end();
    if (!has_header || begin == end) return begin;
    const char* nl = finder(begin, end, '\n');
    return nl == end ? end : nl + 1;
}

// ---------------------------------------------------------------------------
// Schema helpers
// ---------------------------------------------------------------------------

std::vector<std::size_t> string_ordinals(const std::vector<SchemaColumn>& schema) {
    std::vector<std::size_t> out(schema.size(), static_cast<std::size_t>(-1));
    std::size_t ord = 0;
    for (std::size_t c = 0; c < schema.size(); ++c)
        if (schema[c].type == Type::String) out[c] = ord++;
    return out;
}

std::size_t string_column_count(const std::vector<SchemaColumn>& schema) {
    return static_cast<std::size_t>(std::count_if(schema.begin(), schema.end(),
        [](const SchemaColumn& c) { return c.type == Type::String; }));
}

std::size_t resolve_threads(std::size_t requested) {
    if (requested == 0) requested = std::thread::hardware_concurrency();
    return std::max<std::size_t>(1, requested);
}

// ---------------------------------------------------------------------------
// Field parsing
// ---------------------------------------------------------------------------

// Parses one number starting at p. Returns the first byte after the number or
// nullptr if no valid number starts there. Parsing stops naturally at the
// delimiter, so numeric fields are scanned exactly once.
template <typename T>
inline const char* parse_number_at(const char* p, const char* end, T& out) noexcept {
#ifdef FASTCSV_HAS_FAST_FLOAT
    if constexpr (std::is_floating_point_v<T>) {
        const auto r = fast_float::from_chars(p, end, out);
        return r.ec == std::errc{} ? r.ptr : nullptr;
    } else
#endif
    {
        const auto r = std::from_chars(p, end, out);
        return r.ec == std::errc{} ? r.ptr : nullptr;
    }
}

// Validates the byte that ends a field. Returns the start of the next field,
// `end` when the chunk is exhausted, or nullptr when the terminator is wrong.
// A "\r\n" row ending is accepted so files produced on Windows still parse.
inline const char* consume_terminator(const char* q, const char* end, char delimiter, bool last) noexcept {
    if (q == end) return last ? end : nullptr;
    const char ch = *q;
    if (!last) return ch == delimiter ? q + 1 : nullptr;
    if (ch == '\n') return q + 1;
    if (ch == '\r') {
        if (q + 1 == end) return end;
        if (q[1] == '\n') return q + 2;
    }
    return nullptr;
}

inline bool field_is_empty(const char* p, const char* end, char delimiter, bool last) noexcept {
    if (p == end) return true;
    const char ch = *p;
    return last ? (ch == '\n' || ch == '\r') : (ch == delimiter);
}

[[noreturn]] void throw_field_error(const char* what, std::size_t row, const SchemaColumn& column) {
    throw std::runtime_error(std::string(what) + " at data row " + std::to_string(row) +
                             ", column '" + column.name + "' (" + std::string(type_name(column.type)) + ")");
}

template <typename T>
inline void store_strided(void* base, std::size_t row, std::size_t stride, T value) noexcept {
    static_cast<T*>(base)[row * stride] = value;
}

template <typename T>
inline const char* parse_numeric_field(const char* p, const char* end, char delimiter, bool last,
                                       bool empty_is_nan, const ReadColumnView& target,
                                       std::size_t row, const SchemaColumn& column) {
    T value;
    const char* q;
    if constexpr (std::is_floating_point_v<T>) {
        if (empty_is_nan && field_is_empty(p, end, delimiter, last)) {
            value = std::numeric_limits<T>::quiet_NaN();
            q = p;
        } else {
            q = parse_number_at<T>(p, end, value);
            if (!q) throw_field_error("invalid numeric field", row, column);
        }
    } else {
        q = parse_number_at<T>(p, end, value);
        if (!q) throw_field_error("invalid integer field", row, column);
    }
    store_strided<T>(target.numeric, row, target.stride, value);
    const char* next = consume_terminator(q, end, delimiter, last);
    if (!next) throw_field_error(last ? "unexpected bytes after last column" : "unexpected field terminator",
                                 row, column);
    return next;
}

struct ChunkPlan {
    std::vector<std::uint64_t> offsets;   // includes data start and EOF
    std::vector<std::size_t> rows;
    std::vector<std::size_t> row_starts;
};

ChunkPlan make_chunk_plan(const MappedFile& file,
                          const char* data_begin,
                          std::size_t expected_rows,
                          const std::vector<std::uint64_t>& chunk_offsets,
                          const std::vector<std::uint64_t>& chunk_row_counts) {
    ChunkPlan plan;
    const std::uint64_t data_off = static_cast<std::uint64_t>(data_begin - file.data);
    plan.offsets.reserve(chunk_offsets.size() + 2);
    plan.offsets.push_back(data_off);
    std::uint64_t previous = data_off;
    for (const auto off : chunk_offsets) {
        if (off <= previous || off >= file.size)
            throw std::runtime_error("chunk offsets must be strictly increasing row boundaries inside the file");
        if (file.data[off - 1] != '\n')
            throw std::runtime_error("chunk offset " + std::to_string(off) +
                                     " is not a row boundary; the sidecar does not describe this file");
        plan.offsets.push_back(off);
        previous = off;
    }
    plan.offsets.push_back(static_cast<std::uint64_t>(file.size));

    const std::size_t nchunks = plan.offsets.size() - 1;
    plan.rows.resize(nchunks, 0);
    if (!chunk_row_counts.empty()) {
        if (chunk_row_counts.size() != nchunks)
            throw std::runtime_error("chunk_row_counts size does not match chunk offsets");
        for (std::size_t i = 0; i < nchunks; ++i)
            plan.rows[i] = static_cast<std::size_t>(chunk_row_counts[i]);
    } else if (nchunks == 1) {
        plan.rows[0] = expected_rows;
    } else {
        throw std::runtime_error("parallel chunk offsets require chunk_row_counts");
    }

    plan.row_starts.resize(nchunks + 1, 0);
    for (std::size_t i = 0; i < nchunks; ++i)
        plan.row_starts[i + 1] = plan.row_starts[i] + plan.rows[i];
    if (plan.row_starts.back() != expected_rows)
        throw std::runtime_error("row count/chunk metadata do not match CSV");
    return plan;
}

void validate_targets(const std::vector<SchemaColumn>& schema,
                      const std::vector<ReadColumnView>& targets) {
    if (schema.empty()) throw std::invalid_argument("schema must not be empty");
    if (schema.size() != targets.size()) throw std::invalid_argument("schema/target mismatch");
    for (std::size_t c = 0; c < schema.size(); ++c) {
        if (schema[c].type != targets[c].type) throw std::invalid_argument("target dtype mismatch");
        if (schema[c].type == Type::Skip) continue;
        if (schema[c].type == Type::String) {
            if (!targets[c].string_offsets) throw std::invalid_argument("missing string offsets target");
        } else {
            if (!targets[c].numeric) throw std::invalid_argument("missing numeric target buffer");
            if (targets[c].stride == 0) throw std::invalid_argument("numeric target stride must be >= 1");
        }
    }
}

// Parses rows [row_begin, row_end) of one chunk straight into the final buffers.
void parse_chunk_into(const char* begin,
                      const char* end,
                      const std::vector<SchemaColumn>& schema,
                      const std::vector<ReadColumnView>& targets,
                      const std::vector<std::size_t>& str_ord,
                      const ReadOptions& options,
                      FindByteFn finder,
                      std::size_t row_begin,
                      std::size_t row_end,
                      const std::vector<std::uint64_t>& string_byte_bases) {
    const char delimiter = options.delimiter;
    const bool empty_nan = options.empty_float_is_nan;
    const std::size_t ncols = schema.size();
    const char* p = begin;
    std::vector<std::uint64_t> cursors = string_byte_bases;

    for (std::size_t row = row_begin; row < row_end; ++row) {
        if (p == end) throw std::runtime_error("CSV has fewer rows than expected (stopped at data row " +
                                               std::to_string(row) + ")");
        for (std::size_t c = 0; c < ncols; ++c) {
            const bool last = (c + 1 == ncols);
            const ReadColumnView& target = targets[c];
            switch (schema[c].type) {
                case Type::Int64:
                    p = parse_numeric_field<std::int64_t>(p, end, delimiter, last, empty_nan, target, row, schema[c]);
                    break;
                case Type::Int32:
                    p = parse_numeric_field<std::int32_t>(p, end, delimiter, last, empty_nan, target, row, schema[c]);
                    break;
                case Type::Float64:
                    p = parse_numeric_field<double>(p, end, delimiter, last, empty_nan, target, row, schema[c]);
                    break;
                case Type::Float32:
                    p = parse_numeric_field<float>(p, end, delimiter, last, empty_nan, target, row, schema[c]);
                    break;
                case Type::String: {
                    const char* q = finder(p, end, last ? '\n' : delimiter);
                    if (q == end && !last) throw_field_error("truncated row or column count mismatch", row, schema[c]);
                    const char* field_end = q;
                    if (last && field_end > p && field_end[-1] == '\r') --field_end;
                    const std::size_t ord = str_ord[c];
                    const std::size_t n = static_cast<std::size_t>(field_end - p);
                    const std::uint64_t dst = cursors[ord];
                    target.string_offsets[row] = dst;
                    if (n) std::memcpy(target.string_bytes + dst, p, n);
                    cursors[ord] += static_cast<std::uint64_t>(n);
                    p = q == end ? end : q + 1;
                    break;
                }
                case Type::Skip: {
                    const char* q = finder(p, end, last ? '\n' : delimiter);
                    if (q == end && !last) throw_field_error("truncated row or column count mismatch", row, schema[c]);
                    p = q == end ? end : q + 1;
                    break;
                }
            }
        }
    }
    if (p != end)
        throw std::runtime_error("CSV has more rows than expected after data row " + std::to_string(row_end));
}

// Runs fn(i) for i in [0, n) on up to `threads` workers; rethrows the first error.
template <typename Fn>
std::size_t parallel_for(std::size_t n, std::size_t threads, Fn&& fn) {
    if (n == 0) return 0;
    const std::size_t workers = std::min<std::size_t>(resolve_threads(threads), n);
    if (workers == 1) {
        for (std::size_t i = 0; i < n; ++i) fn(i);
        return 1;
    }
    std::atomic<std::size_t> next{0};
    std::atomic<bool> cancel{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) {
        pool.emplace_back([&]() {
            while (!cancel.load(std::memory_order_relaxed)) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) break;
                try {
                    fn(i);
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!error) error = std::current_exception();
                    }
                    cancel.store(true, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
    return workers;
}

// Splits [begin, end) into byte-balanced parts whose boundaries sit right
// after a newline. Returns part boundaries including begin and end.
std::vector<const char*> split_at_rows(const char* begin, const char* end, std::size_t nparts, FindByteFn finder) {
    std::vector<const char*> bounds;
    bounds.push_back(begin);
    const std::size_t size = static_cast<std::size_t>(end - begin);
    for (std::size_t i = 1; i < nparts; ++i) {
        const char* tentative = begin + (size / nparts) * i;
        if (tentative <= bounds.back()) continue;
        const char* nl = finder(tentative, end, '\n');
        if (nl == end) break;
        const char* boundary = nl + 1;
        if (boundary >= end) break;
        if (boundary > bounds.back()) bounds.push_back(boundary);
    }
    bounds.push_back(end);
    return bounds;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool simd_scanner_compiled() noexcept {
#if defined(FASTCSV_X86_SIMD)
    return true;
#else
    return false;
#endif
}

std::string simd_scanner_runtime() {
    switch (runtime_simd_mode()) {
        case SimdMode::AVX2: return "avx2";
        case SimdMode::SSE2: return "sse2";
        default: return "scalar";
    }
}

bool fast_float_compiled() noexcept {
#ifdef FASTCSV_HAS_FAST_FLOAT
    return true;
#else
    return false;
#endif
}

std::string read_first_line(const std::string& path, std::size_t max_bytes) {
    MappedFile file(path, false);
    const char* begin = file.begin();
    const char* end = file.end();
    if (static_cast<std::size_t>(end - begin) > max_bytes) end = begin + max_bytes;
    const char* nl = resolved_find_byte()(begin, end, '\n');
    if (nl > begin && nl[-1] == '\r') --nl;
    return std::string(begin, nl);
}

InspectResult inspect_strict_csv(const std::string& path,
                                 const std::vector<SchemaColumn>& schema,
                                 const ReadOptions& options,
                                 std::size_t target_chunk_bytes) {
    if (schema.empty()) throw std::invalid_argument("schema must not be empty");
    MappedFile file(path, options.sequential_io_hint);
    const FindByteFn finder = resolved_find_byte();
    const char* begin = data_begin_for(file, options.has_header, finder);
    const char* end = file.end();
    const std::size_t nstr = string_column_count(schema);

    InspectResult result;
    result.input_bytes = static_cast<std::uint64_t>(file.size);
    result.data_offset = static_cast<std::uint64_t>(begin - file.data);
    result.string_bytes.assign(nstr, 0);
    if (begin == end) return result;

    // Choose a part count that gives every worker something to do while
    // keeping chunks near the requested byte target.
    const std::size_t size = static_cast<std::size_t>(end - begin);
    const std::size_t workers = resolve_threads(options.threads);
    std::size_t nparts = target_chunk_bytes ? (size + target_chunk_bytes - 1) / target_chunk_bytes : 1;
    nparts = std::max(nparts, workers * 2);
    constexpr std::size_t min_part_bytes = 64 * 1024;
    nparts = std::min(nparts, std::max<std::size_t>(1, size / min_part_bytes));
    const std::vector<const char*> bounds = split_at_rows(begin, end, nparts, finder);
    const std::size_t nchunks = bounds.size() - 1;

    std::vector<std::uint64_t> rows(nchunks, 0);
    std::vector<std::vector<std::uint64_t>> str_bytes(nchunks, std::vector<std::uint64_t>(nstr, 0));
    const auto str_ord = string_ordinals(schema);
    const char delimiter = options.delimiter;

    parallel_for(nchunks, options.threads, [&](std::size_t i) {
        const char* p = bounds[i];
        const char* e = bounds[i + 1];
        if (nstr == 0) {
            rows[i] = count_rows(p, e);
            return;
        }
        std::uint64_t n = 0;
        auto& bytes = str_bytes[i];
        while (p < e) {
            for (std::size_t c = 0; c < schema.size(); ++c) {
                const bool last = c + 1 == schema.size();
                const char* q = finder(p, e, last ? '\n' : delimiter);
                if (q == e && !last)
                    throw std::runtime_error("truncated row or column count mismatch during inspection at data row " +
                                             std::to_string(n));
                if (schema[c].type == Type::String) {
                    const char* fe = q;
                    if (last && fe > p && fe[-1] == '\r') --fe;
                    bytes[str_ord[c]] += static_cast<std::uint64_t>(fe - p);
                }
                p = q == e ? e : q + 1;
            }
            ++n;
        }
        rows[i] = n;
    });

    result.chunk_row_counts = rows;
    result.chunk_string_bytes = str_bytes;
    for (std::size_t i = 1; i < nchunks; ++i)
        result.chunk_offsets.push_back(static_cast<std::uint64_t>(bounds[i] - file.data));
    for (std::size_t i = 0; i < nchunks; ++i) {
        result.rows += static_cast<std::size_t>(rows[i]);
        for (std::size_t s = 0; s < nstr; ++s) result.string_bytes[s] += str_bytes[i][s];
    }
    return result;
}

ReadStats read_strict_csv_into(const std::string& path,
                               const std::vector<SchemaColumn>& schema,
                               const std::vector<ReadColumnView>& targets,
                               const ReadOptions& options,
                               std::size_t expected_rows,
                               const std::vector<std::uint64_t>& chunk_offsets,
                               const std::vector<std::uint64_t>& chunk_row_counts,
                               const std::vector<std::vector<std::uint64_t>>& chunk_string_bytes) {
    validate_targets(schema, targets);
    MappedFile file(path, options.sequential_io_hint);
    const FindByteFn finder = resolved_find_byte();
    const char* data_begin = data_begin_for(file, options.has_header, finder);
    const auto str_ord = string_ordinals(schema);
    const std::size_t nstr = string_column_count(schema);

    if (expected_rows == 0) {
        if (data_begin != file.end())
            throw std::invalid_argument("expected_rows must be known before direct-buffer parsing");
        for (std::size_t c = 0; c < schema.size(); ++c)
            if (schema[c].type == Type::String) targets[c].string_offsets[0] = 0;
        return {0, 0, 0, static_cast<std::uint64_t>(file.size)};
    }

    const ChunkPlan plan = make_chunk_plan(file, data_begin, expected_rows, chunk_offsets, chunk_row_counts);
    const std::size_t nchunks = plan.rows.size();

    std::vector<std::vector<std::uint64_t>> string_bases(nchunks, std::vector<std::uint64_t>(nstr, 0));
    if (nstr > 0) {
        if (chunk_string_bytes.size() != nchunks)
            throw std::runtime_error("packed strings require per-chunk string byte counts");
        std::vector<std::uint64_t> totals(nstr, 0);
        for (std::size_t i = 0; i < nchunks; ++i) {
            if (chunk_string_bytes[i].size() != nstr)
                throw std::runtime_error("chunk_string_bytes column count mismatch");
            string_bases[i] = totals;
            for (std::size_t s = 0; s < nstr; ++s) totals[s] += chunk_string_bytes[i][s];
        }
        for (std::size_t c = 0; c < schema.size(); ++c) {
            if (schema[c].type != Type::String) continue;
            const std::size_t s = str_ord[c];
            targets[c].string_offsets[expected_rows] = totals[s];
            if (totals[s] && !targets[c].string_bytes)
                throw std::invalid_argument("missing string byte target");
        }
    }

    const std::vector<std::uint64_t> no_strings;
    const std::size_t workers = parallel_for(nchunks, options.threads, [&](std::size_t i) {
        parse_chunk_into(file.data + plan.offsets[i], file.data + plan.offsets[i + 1],
                         schema, targets, str_ord, options, finder,
                         plan.row_starts[i], plan.row_starts[i + 1],
                         nstr ? string_bases[i] : no_strings);
    });
    return {expected_rows, nchunks, workers, static_cast<std::uint64_t>(file.size)};
}

ReadStats read_strict_numeric_matrix_into(const std::string& path,
                                          Type type,
                                          std::size_t columns,
                                          void* output,
                                          const ReadOptions& options,
                                          std::size_t expected_rows,
                                          const std::vector<std::uint64_t>& chunk_offsets,
                                          const std::vector<std::uint64_t>& chunk_row_counts,
                                          const std::vector<std::string>& names) {
    if (!output && expected_rows) throw std::invalid_argument("missing matrix output buffer");
    if (columns == 0) throw std::invalid_argument("matrix must have at least one column");
    if (!is_numeric(type)) throw std::invalid_argument("numeric matrix needs a numeric dtype");
    if (expected_rows == 0) {
        MappedFile file(path, options.sequential_io_hint);
        if (data_begin_for(file, options.has_header, resolved_find_byte()) != file.end())
            throw std::invalid_argument("expected_rows must be known before direct matrix parsing");
        return {0, 0, 0, static_cast<std::uint64_t>(file.size)};
    }

    // A row-major matrix is just `columns` strided views over one allocation.
    std::vector<SchemaColumn> schema(columns);
    std::vector<ReadColumnView> targets(columns);
    char* base = static_cast<char*>(output);
    for (std::size_t c = 0; c < columns; ++c) {
        schema[c] = {c < names.size() ? names[c] : "col" + std::to_string(c), type};
        targets[c].type = type;
        targets[c].numeric = base + c * item_size(type);
        targets[c].stride = columns;
    }
    return read_strict_csv_into(path, schema, targets, options, expected_rows, chunk_offsets, chunk_row_counts, {});
}

}  // namespace fastcsv
