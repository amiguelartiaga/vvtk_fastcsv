// Standalone C++ smoke test: builds without pybind11 or Python.
#include "fastcsv/reader.hpp"
#include "fastcsv/writer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::cerr << __FILE__ << ":" << __LINE__ << ": check failed: " #cond "\n"; \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

int main() {
    using namespace fastcsv;
    const std::string path = ".fastcsv_core_smoke.csv";
    const std::size_t rows = 10003;
    std::vector<std::int64_t> id(rows);
    std::vector<double> x(rows);
    std::vector<std::uint64_t> so(rows + 1);
    std::vector<char> sb;
    so[0] = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        id[i] = static_cast<std::int64_t>(i * 7) - 5000;
        x[i] = static_cast<double>(i) / 17.0;
        const std::string s = "v" + std::to_string(i % 997);
        sb.insert(sb.end(), s.begin(), s.end());
        so[i + 1] = sb.size();
    }

    std::vector<SchemaColumn> schema = {{"id", Type::Int64}, {"x", Type::Float64}, {"s", Type::String}};
    std::vector<WriteColumnView> w(3);
    w[0].type = Type::Int64; w[0].numeric = id.data();
    w[1].type = Type::Float64; w[1].numeric = x.data();
    w[2].type = Type::String; w[2].strings = {so.data(), sb.data()};
    WriteOptions wo;
    wo.target_chunk_bytes = 8 * 1024;
    wo.threads = 4;
    wo.block_rows = 500;
    const auto wr = write_canonical_csv(path, schema, w, rows, wo);
    CHECK(wr.rows == rows);
    CHECK(wr.chunk_row_counts.size() > 2);
    CHECK(wr.chunk_byte_counts.size() == wr.chunk_row_counts.size());
    CHECK(wr.chunk_string_bytes.size() == wr.chunk_row_counts.size());

    ReadOptions ro;
    ro.threads = 4;

    // Sidecar-driven parallel read into final buffers.
    std::vector<std::int64_t> id2(rows);
    std::vector<double> x2(rows);
    std::vector<std::uint64_t> so2(rows + 1);
    std::vector<char> sb2(sb.size());
    std::vector<ReadColumnView> r(3);
    r[0].type = Type::Int64; r[0].numeric = id2.data();
    r[1].type = Type::Float64; r[1].numeric = x2.data();
    r[2].type = Type::String; r[2].string_offsets = so2.data(); r[2].string_bytes = sb2.data();
    const auto rr = read_strict_csv_into(path, schema, r, ro, rows,
                                         wr.chunk_offsets, wr.chunk_row_counts, wr.chunk_string_bytes);
    CHECK(rr.rows == rows);
    CHECK(rr.workers >= 1);
    for (std::size_t i = 0; i < rows; ++i) {
        CHECK(id2[i] == id[i]);
        CHECK(x2[i] == x[i]);
        CHECK(so2[i] == so[i]);
    }
    CHECK(so2[rows] == so[rows]);
    CHECK(sb2 == sb);

    // Sidecar-free: parallel inspection must produce an equivalent plan.
    const auto insp = inspect_strict_csv(path, schema, ro, 8 * 1024);
    CHECK(insp.rows == rows);
    CHECK(insp.string_bytes.size() == 1 && insp.string_bytes[0] == sb.size());
    CHECK(insp.chunk_row_counts.size() > 1);
    std::vector<std::int64_t> id3(rows);
    std::vector<double> x3(rows);
    std::vector<std::uint64_t> so3(rows + 1);
    std::vector<char> sb3(sb.size());
    r[0].numeric = id3.data(); r[1].numeric = x3.data();
    r[2].string_offsets = so3.data(); r[2].string_bytes = sb3.data();
    const auto rr2 = read_strict_csv_into(path, schema, r, ro, insp.rows,
                                          insp.chunk_offsets, insp.chunk_row_counts, insp.chunk_string_bytes);
    CHECK(rr2.chunks == insp.chunk_row_counts.size());
    CHECK(id3 == id && x3 == x && so3 == so && sb3 == sb);

    // Direct homogeneous matrix path, written from a row-major matrix.
    const std::string mpath = ".fastcsv_matrix_smoke.csv";
    std::vector<SchemaColumn> ms = {{"a", Type::Float32}, {"b", Type::Float32}};
    std::vector<float> matrix_in(rows * 2);
    for (std::size_t i = 0; i < rows; ++i) { matrix_in[i * 2] = i * 0.5f; matrix_in[i * 2 + 1] = i * -0.25f; }
    std::vector<WriteColumnView> mw(2);
    mw[0].type = Type::Float32; mw[0].numeric = matrix_in.data(); mw[0].stride = 2;
    mw[1].type = Type::Float32; mw[1].numeric = matrix_in.data() + 1; mw[1].stride = 2;
    WriteOptions mwo;
    mwo.target_chunk_bytes = 8 * 1024;
    const auto mwr = write_canonical_csv(mpath, ms, mw, rows, mwo);
    std::vector<float> matrix(rows * 2);
    const auto mr = read_strict_numeric_matrix_into(mpath, Type::Float32, 2, matrix.data(), ro, rows,
                                                    mwr.chunk_offsets, mwr.chunk_row_counts);
    CHECK(mr.rows == rows);
    CHECK(matrix == matrix_in);

    // Foreign file: CRLF endings, empty float cells become NaN, no sidecar.
    const std::string fpath = ".fastcsv_foreign_smoke.csv";
    {
        std::ofstream f(fpath, std::ios::binary);
        f << "a,b\r\n1,2.5\r\n2,\r\n3,-1e3";
    }
    std::vector<SchemaColumn> fs = {{"a", Type::Int32}, {"b", Type::Float64}};
    ReadOptions fro;
    const auto finsp = inspect_strict_csv(fpath, fs, fro, 0);
    CHECK(finsp.rows == 3);
    std::vector<std::int32_t> fa(3);
    std::vector<double> fb(3);
    std::vector<ReadColumnView> fr(2);
    fr[0].type = Type::Int32; fr[0].numeric = fa.data();
    fr[1].type = Type::Float64; fr[1].numeric = fb.data();
    read_strict_csv_into(fpath, fs, fr, fro, 3, finsp.chunk_offsets, finsp.chunk_row_counts, {});
    CHECK(fa[0] == 1 && fa[1] == 2 && fa[2] == 3);
    CHECK(fb[0] == 2.5 && std::isnan(fb[1]) && fb[2] == -1000.0);
    CHECK(read_first_line(fpath) == "a,b");

    // Malformed input must be rejected, not silently accepted.
    {
        std::ofstream f(fpath, std::ios::binary);
        f << "a,b\n1,2.5\n2,x\n";
    }
    bool threw = false;
    try {
        read_strict_csv_into(fpath, fs, fr, fro, 2, {}, {}, {});
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("row 1") != std::string::npos;
    }
    CHECK(threw);

    std::cout << "ok rows=" << rows
              << " write_chunks=" << wr.chunk_row_counts.size()
              << " inspect_chunks=" << insp.chunk_row_counts.size()
              << " simd=" << simd_scanner_runtime()
              << " fast_float=" << (fast_float_compiled() ? "yes" : "no")
              << " bytes=" << wr.bytes_written << "\n";
    std::remove(path.c_str());
    std::remove(mpath.c_str());
    std::remove(fpath.c_str());
    return 0;
}
