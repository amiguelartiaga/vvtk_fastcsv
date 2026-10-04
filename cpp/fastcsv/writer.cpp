#include "fastcsv/writer.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fastcsv {
namespace {

class RawFile {
public:
    explicit RawFile(const std::string& path) {
#ifndef _WIN32
        fd_ = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
        if (fd_ < 0) throw std::runtime_error("cannot open file for writing: " + path);
#if defined(POSIX_FADV_SEQUENTIAL)
        (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
#else
        handle_ = ::CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("cannot open file for writing: " + path);
#endif
    }

    ~RawFile() {
#ifndef _WIN32
        if (fd_ >= 0) ::close(fd_);
#else
        if (handle_ != INVALID_HANDLE_VALUE) ::CloseHandle(handle_);
#endif
    }

    RawFile(const RawFile&) = delete;
    RawFile& operator=(const RawFile&) = delete;

    void write_all(const char* data, std::size_t size) {
#ifndef _WIN32
        while (size) {
            const auto n = ::write(fd_, data, size);
            if (n <= 0) throw std::runtime_error("write failed");
            data += static_cast<std::size_t>(n);
            size -= static_cast<std::size_t>(n);
        }
#else
        while (size) {
            const DWORD request = static_cast<DWORD>(std::min<std::size_t>(size, 0x7ffff000u));
            DWORD written = 0;
            if (!::WriteFile(handle_, data, request, &written, nullptr) || written == 0)
                throw std::runtime_error("write failed");
            data += written;
            size -= written;
        }
#endif
    }

private:
#ifndef _WIN32
    int fd_ = -1;
#else
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#endif
};

// Growable byte buffer that lets formatters write in place.
struct Block {
    std::vector<char> data;
    std::size_t used = 0;
    std::size_t rows = 0;
    std::vector<std::uint64_t> string_bytes;

    void clear(std::size_t nstr) {
        used = 0;
        rows = 0;
        string_bytes.assign(nstr, 0);
    }

    inline char* reserve(std::size_t n) {
        if (used + n > data.size()) data.resize(std::max(data.size() * 2, used + n + 4096));
        return data.data() + used;
    }

    inline void put(char c) {
        *reserve(1) = c;
        ++used;
    }

    inline void append(const char* s, std::size_t n) {
        if (n) std::memcpy(reserve(n), s, n);
        used += n;
    }

    template <typename T>
    inline void append_integer(T value) {
        char* out = reserve(32);
        const auto r = std::to_chars(out, out + 32, value);
        if (r.ec != std::errc{}) throw std::runtime_error("integer formatting failed");
        used += static_cast<std::size_t>(r.ptr - out);
    }

    template <typename T>
    inline void append_float(T value, int precision) {
        char* out = reserve(400);
        std::to_chars_result r = precision < 0
            ? std::to_chars(out, out + 400, value)
            : std::to_chars(out, out + 400, value, std::chars_format::general, precision);
        if (r.ec != std::errc{}) {
            const int n = std::snprintf(out, 400, std::is_same_v<T, float> ? "%.9g" : "%.17g",
                                        static_cast<double>(value));
            if (n <= 0) throw std::runtime_error("float formatting failed");
            r.ptr = out + n;
        }
        used += static_cast<std::size_t>(r.ptr - out);
    }
};

template <typename T>
inline T load_strided(const void* base, std::size_t row, std::size_t stride) noexcept {
    return static_cast<const T*>(base)[row * stride];
}

void format_rows(Block& block,
                 std::size_t row_begin,
                 std::size_t row_end,
                 const std::vector<SchemaColumn>& schema,
                 const std::vector<WriteColumnView>& columns,
                 const std::vector<std::size_t>& str_ord,
                 const WriteOptions& options) {
    const char delimiter = options.delimiter;
    const int precision = options.float_precision;
    const std::size_t ncols = schema.size();
    // Rough pre-size to avoid many small regrowths.
    block.reserve((row_end - row_begin) * (ncols * 12 + 1));
    for (std::size_t row = row_begin; row < row_end; ++row) {
        for (std::size_t c = 0; c < ncols; ++c) {
            if (c) block.put(delimiter);
            const WriteColumnView& view = columns[c];
            switch (schema[c].type) {
                case Type::Int64:
                    block.append_integer(load_strided<std::int64_t>(view.numeric, row, view.stride));
                    break;
                case Type::Int32:
                    block.append_integer(load_strided<std::int32_t>(view.numeric, row, view.stride));
                    break;
                case Type::Float64:
                    block.append_float(load_strided<double>(view.numeric, row, view.stride), precision);
                    break;
                case Type::Float32:
                    block.append_float(load_strided<float>(view.numeric, row, view.stride), precision);
                    break;
                case Type::String: {
                    const std::uint64_t b = view.strings.offsets[row];
                    const std::uint64_t e = view.strings.offsets[row + 1];
                    if (e < b) throw std::invalid_argument("invalid packed string offsets");
                    const std::size_t n = static_cast<std::size_t>(e - b);
                    block.append(view.strings.bytes + b, n);
                    block.string_bytes[str_ord[c]] += static_cast<std::uint64_t>(n);
                    break;
                }
                case Type::Skip: break;
            }
        }
        block.put('\n');
    }
    block.rows = row_end - row_begin;
}

// Ordered producer/consumer pipeline: `workers` threads format blocks into a
// ring of reusable buffers while the calling thread writes finished blocks in
// order. Formatting of later blocks overlaps the file writes.
template <typename Format, typename Emit>
void ordered_pipeline(std::size_t nblocks, std::size_t workers, std::vector<Block>& slots,
                      Format&& format, Emit&& emit) {
    if (nblocks == 0) return;
    const std::size_t nslots = slots.size();
    if (workers <= 1 || nblocks == 1) {
        for (std::size_t i = 0; i < nblocks; ++i) {
            format(i, slots[0]);
            emit(i, slots[0]);
        }
        return;
    }

    std::mutex mu;
    std::condition_variable cv;
    std::vector<char> done(nblocks, 0);
    std::size_t written = 0;
    bool failed = false;
    std::exception_ptr error;
    std::atomic<std::size_t> next{0};

    auto fail = [&](std::exception_ptr e) {
        std::lock_guard<std::mutex> lock(mu);
        if (!error) error = std::move(e);
        failed = true;
        cv.notify_all();
    };

    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) {
        pool.emplace_back([&]() {
            for (;;) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= nblocks) break;
                {
                    // Slot i % nslots is free once block i - nslots has been written.
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&] { return failed || i < written + nslots; });
                    if (failed) break;
                }
                try {
                    format(i, slots[i % nslots]);
                } catch (...) {
                    fail(std::current_exception());
                    break;
                }
                {
                    std::lock_guard<std::mutex> lock(mu);
                    done[i] = 1;
                }
                cv.notify_all();
            }
        });
    }

    try {
        for (std::size_t i = 0; i < nblocks; ++i) {
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return failed || done[i]; });
                if (failed) break;
            }
            emit(i, slots[i % nslots]);
            {
                std::lock_guard<std::mutex> lock(mu);
                ++written;
            }
            cv.notify_all();
        }
    } catch (...) {
        fail(std::current_exception());
    }
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace

WriteResult write_canonical_csv(const std::string& path,
                                const std::vector<SchemaColumn>& schema,
                                const std::vector<WriteColumnView>& columns,
                                std::size_t rows,
                                const WriteOptions& options) {
    if (schema.empty() || schema.size() != columns.size())
        throw std::invalid_argument("schema/column mismatch");
    for (std::size_t c = 0; c < schema.size(); ++c) {
        if (schema[c].type != columns[c].type) throw std::invalid_argument("column dtype mismatch");
        if (schema[c].type == Type::Skip) throw std::invalid_argument("skip columns cannot be written");
        if (schema[c].type == Type::String) {
            if (rows && !columns[c].strings.offsets) throw std::invalid_argument("missing packed string offsets");
        } else {
            if (rows && !columns[c].numeric) throw std::invalid_argument("missing numeric column data");
            if (columns[c].stride == 0) throw std::invalid_argument("numeric column stride must be >= 1");
        }
    }

    const std::size_t nstr = static_cast<std::size_t>(std::count_if(
        schema.begin(), schema.end(), [](const auto& x) { return x.type == Type::String; }));
    std::vector<std::size_t> str_ord(schema.size(), static_cast<std::size_t>(-1));
    for (std::size_t c = 0, s = 0; c < schema.size(); ++c)
        if (schema[c].type == Type::String) str_ord[c] = s++;

    std::size_t workers = options.threads == 0 ? std::thread::hardware_concurrency() : options.threads;
    workers = std::max<std::size_t>(1, workers);

    std::size_t block_rows = options.block_rows;
    if (block_rows == 0) {
        if (options.chunk_rows > 0) {
            block_rows = options.chunk_rows;  // chunks must close exactly on row multiples
        } else {
            // Small blocks start the ordered write stream early; the stream is the
            // bottleneck on fast storage, not formatting.
            block_rows = std::clamp<std::size_t>(rows / (workers * 8), 1024, 8192);
        }
    }
    const std::size_t nblocks = rows ? (rows + block_rows - 1) / block_rows : 0;
    workers = std::min(workers, std::max<std::size_t>(1, nblocks));

    RawFile out(path);
    WriteResult result;
    result.rows = rows;
    result.workers = workers;
    std::uint64_t position = 0;

    if (options.header) {
        Block header;
        header.clear(0);
        for (std::size_t c = 0; c < schema.size(); ++c) {
            if (c) header.put(options.delimiter);
            header.append(schema[c].name.data(), schema[c].name.size());
        }
        header.put('\n');
        out.write_all(header.data.data(), header.used);
        position += header.used;
    }

    // Chunk bookkeeping for the sidecar, advanced as blocks are written in order.
    std::vector<std::uint64_t> current_string_bytes(nstr, 0);
    std::uint64_t current_chunk_rows = 0;
    std::uint64_t chunk_start = position;
    auto close_chunk = [&]() {
        result.chunk_row_counts.push_back(current_chunk_rows);
        result.chunk_byte_counts.push_back(position - chunk_start);
        result.chunk_string_bytes.push_back(current_string_bytes);
        current_chunk_rows = 0;
        std::fill(current_string_bytes.begin(), current_string_bytes.end(), 0);
        chunk_start = position;
    };

    std::vector<Block> slots(std::min<std::size_t>(std::max<std::size_t>(1, nblocks), workers > 1 ? workers * 2 : 1));
    ordered_pipeline(
        nblocks, workers, slots,
        [&](std::size_t i, Block& block) {
            block.clear(nstr);
            const std::size_t rb = i * block_rows;
            const std::size_t re = std::min(rows, rb + block_rows);
            format_rows(block, rb, re, schema, columns, str_ord, options);
        },
        [&](std::size_t i, Block& block) {
            out.write_all(block.data.data(), block.used);
            position += block.used;
            current_chunk_rows += block.rows;
            for (std::size_t s = 0; s < nstr; ++s) current_string_bytes[s] += block.string_bytes[s];
            if (i + 1 == nblocks) return;
            const bool hit_rows = options.chunk_rows > 0 && current_chunk_rows >= options.chunk_rows;
            const bool hit_bytes = options.target_chunk_bytes > 0 &&
                (position - chunk_start) >= static_cast<std::uint64_t>(options.target_chunk_bytes);
            if (hit_rows || hit_bytes) {
                close_chunk();
                result.chunk_offsets.push_back(position);
            }
        });

    if (rows > 0) close_chunk();
    result.bytes_written = position;
    return result;
}

}  // namespace fastcsv
