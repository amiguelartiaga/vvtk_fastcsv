// Python bindings. The parser never allocates result memory itself: Python
// hands in writable buffers (NumPy arrays, torch tensors viewed as NumPy
// arrays, or anything else exposing the buffer protocol) and C++ parses
// straight into them. The GIL is released while the C++ core runs.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "fastcsv/reader.hpp"
#include "fastcsv/writer.hpp"

namespace py = pybind11;

namespace {

std::vector<fastcsv::SchemaColumn> convert_schema(const py::list& schema) {
    std::vector<fastcsv::SchemaColumn> result;
    result.reserve(schema.size());
    for (const auto item : schema) {
        const auto tup = py::cast<py::tuple>(item);
        result.push_back({py::cast<std::string>(tup[0]), fastcsv::parse_type(py::cast<std::string>(tup[1]))});
    }
    if (result.empty()) throw std::invalid_argument("schema must not be empty");
    return result;
}

char one_byte_delimiter(const std::string& delimiter) {
    if (delimiter.size() != 1) throw std::invalid_argument("delimiter must be one byte");
    return delimiter[0];
}

// Buffer-protocol format strings vary across platforms ("l" vs "q" for
// int64, "<d" vs "d"), so match on the base code plus item size.
std::string base_format(const std::string& format) {
    std::size_t i = 0;
    while (i < format.size() && (format[i] == '@' || format[i] == '=' || format[i] == '<' ||
                                 format[i] == '>' || format[i] == '!'))
        ++i;
    return format.substr(i);
}

bool format_matches(const py::buffer_info& info, fastcsv::Type type) {
    const std::string f = base_format(info.format);
    const auto size = static_cast<std::size_t>(info.itemsize);
    switch (type) {
        case fastcsv::Type::Int64: return size == 8 && (f == "q" || f == "l" || f == "n");
        case fastcsv::Type::Int32: return size == 4 && (f == "i" || f == "l");
        case fastcsv::Type::Float64: return size == 8 && f == "d";
        case fastcsv::Type::Float32: return size == 4 && f == "f";
        case fastcsv::Type::String: return false;
        case fastcsv::Type::Skip: return false;
    }
    return false;
}

fastcsv::Type type_from_buffer(const py::buffer_info& info) {
    for (auto t : {fastcsv::Type::Int64, fastcsv::Type::Int32, fastcsv::Type::Float64, fastcsv::Type::Float32})
        if (format_matches(info, t)) return t;
    throw std::invalid_argument("unsupported buffer dtype (format '" + info.format +
                                "'); expected int64, int32, float64 or float32");
}

bool is_uint64_buffer(const py::buffer_info& info) {
    const std::string f = base_format(info.format);
    return info.itemsize == 8 && (f == "Q" || f == "L" || f == "N");
}

bool is_byte_buffer(const py::buffer_info& info) {
    const std::string f = base_format(info.format);
    return info.itemsize == 1 && (f == "B" || f == "b" || f == "c");
}

struct NumericView {
    void* ptr = nullptr;
    std::size_t stride = 1;
    std::size_t length = 0;
};

// Validates a 1-D numeric buffer of the given type. Any positive element
// stride is accepted, so a column slice of a row-major matrix works directly.
NumericView numeric_view(const py::buffer_info& info, fastcsv::Type type, const std::string& what) {
    if (info.ndim != 1) throw std::invalid_argument(what + ": numeric buffer must be 1-D");
    if (!format_matches(info, type))
        throw std::invalid_argument(what + ": buffer dtype (format '" + info.format + "') does not match schema dtype " +
                                    std::string(fastcsv::type_name(type)));
    const auto item = static_cast<py::ssize_t>(info.itemsize);
    const py::ssize_t stride_bytes = info.strides[0];
    if (stride_bytes <= 0 || stride_bytes % item != 0)
        throw std::invalid_argument(what + ": buffer must have a positive element-aligned stride");
    NumericView view;
    view.ptr = info.ptr;
    view.stride = static_cast<std::size_t>(stride_bytes / item);
    view.length = static_cast<std::size_t>(info.shape[0]);
    return view;
}

struct PackedStringViews {
    std::uint64_t* offsets = nullptr;
    char* bytes = nullptr;
    std::size_t offsets_length = 0;
    std::size_t bytes_length = 0;
};

PackedStringViews packed_views(std::vector<py::buffer_info>& keep, const py::tuple& pair, bool writable,
                               const std::string& what) {
    if (pair.size() != 2) throw std::invalid_argument(what + ": packed strings need (offsets, data)");
    PackedStringViews out;
    keep.emplace_back(py::cast<py::buffer>(pair[0]).request(writable));
    const py::buffer_info& offsets = keep.back();
    if (offsets.ndim != 1 || !is_uint64_buffer(offsets) || offsets.strides[0] != 8)
        throw std::invalid_argument(what + ": packed offsets must be a contiguous 1-D uint64 buffer");
    out.offsets = static_cast<std::uint64_t*>(offsets.ptr);
    out.offsets_length = static_cast<std::size_t>(offsets.shape[0]);
    if (!pair[1].is_none()) {
        keep.emplace_back(py::cast<py::buffer>(pair[1]).request(writable));
        const py::buffer_info& data = keep.back();
        if (data.ndim != 1 || !is_byte_buffer(data) || data.strides[0] != 1)
            throw std::invalid_argument(what + ": packed data must be a contiguous 1-D uint8 buffer");
        out.bytes = static_cast<char*>(data.ptr);
        out.bytes_length = static_cast<std::size_t>(data.shape[0]);
    }
    return out;
}

py::dict stats_dict(const fastcsv::ReadStats& s) {
    py::dict out;
    out["rows"] = s.rows;
    out["chunks"] = s.chunks;
    out["workers"] = s.workers;
    out["input_bytes"] = s.input_bytes;
    return out;
}

fastcsv::ReadOptions read_options(bool has_header, const std::string& delimiter, std::size_t threads,
                                  bool sequential_io_hint, bool empty_float_is_nan) {
    fastcsv::ReadOptions o;
    o.has_header = has_header;
    o.delimiter = one_byte_delimiter(delimiter);
    o.threads = threads;
    o.sequential_io_hint = sequential_io_hint;
    o.empty_float_is_nan = empty_float_is_nan;
    return o;
}

// Owns packed UTF-8 built from a Python sequence of str for the writer.
struct PackedOwner {
    std::vector<std::uint64_t> offsets;
    std::vector<char> bytes;
};

std::unique_ptr<PackedOwner> pack_python_strings(const py::handle& obj, std::size_t rows, char delimiter,
                                                 const std::string& what) {
    const auto seq = py::reinterpret_borrow<py::sequence>(obj);
    if (static_cast<std::size_t>(py::len(seq)) != rows)
        throw std::invalid_argument(what + ": string column length must match row count");
    auto storage = std::make_unique<PackedOwner>();
    storage->offsets.resize(rows + 1);
    storage->offsets[0] = 0;

    auto utf8 = [&](py::ssize_t i, py::object& holder) -> std::pair<const char*, std::size_t> {
        py::object item = seq[i];
        // Missing values (None / NaN from pandas) become empty strings.
        if (item.is_none() || (PyFloat_Check(item.ptr()) && std::isnan(PyFloat_AsDouble(item.ptr()))))
            return {"", 0};
        holder = PyUnicode_Check(item.ptr()) ? std::move(item) : py::str(item);
        Py_ssize_t n = 0;
        const char* s = PyUnicode_AsUTF8AndSize(holder.ptr(), &n);
        if (!s) throw py::error_already_set();
        return {s, static_cast<std::size_t>(n)};
    };

    // Pass 1: exact sizing + strict canonical validation.
    for (std::size_t i = 0; i < rows; ++i) {
        py::object holder;
        const auto [s, n] = utf8(static_cast<py::ssize_t>(i), holder);
        if (std::memchr(s, delimiter, n) || std::memchr(s, '\n', n) || std::memchr(s, '\r', n))
            throw std::invalid_argument(what + ": value at row " + std::to_string(i) +
                                        " contains the delimiter or a newline; quote-free canonical CSV cannot represent it");
        storage->offsets[i + 1] = storage->offsets[i] + static_cast<std::uint64_t>(n);
    }
    storage->bytes.resize(static_cast<std::size_t>(storage->offsets[rows]));
    // Pass 2: one exact copy into the packed byte arena.
    for (std::size_t i = 0; i < rows; ++i) {
        py::object holder;
        const auto [s, n] = utf8(static_cast<py::ssize_t>(i), holder);
        if (n) std::memcpy(storage->bytes.data() + storage->offsets[i], s, n);
    }
    return storage;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "vvtk_fastcsv C++ core: strict CSV parsing straight into caller-owned buffers";
    m.attr("has_fast_float") = fastcsv::fast_float_compiled();
    m.attr("has_simd_scanner") = fastcsv::simd_scanner_compiled();
    m.attr("simd_runtime") = fastcsv::simd_scanner_runtime();
    m.attr("format_version") = 7;
    m.attr("supported_dtypes") = std::vector<std::string>{"int64", "int32", "float64", "float32", "string", "skip"};

    m.def("first_line", [](const std::string& path, std::size_t max_bytes) {
        return py::bytes(fastcsv::read_first_line(path, max_bytes));
    }, py::arg("path"), py::arg("max_bytes") = 1 << 20,
       "Raw first line of the file without its line terminator.");

    m.def("inspect", [](const std::string& path, const py::list& schema, bool has_header,
                        const std::string& delimiter, std::size_t threads, std::size_t target_chunk_bytes,
                        bool sequential_io_hint) {
        const auto cpp_schema = convert_schema(schema);
        const auto options = read_options(has_header, delimiter, threads, sequential_io_hint, true);
        fastcsv::InspectResult inspected;
        {
            py::gil_scoped_release release;
            inspected = fastcsv::inspect_strict_csv(path, cpp_schema, options, target_chunk_bytes);
        }
        py::dict out;
        out["rows"] = inspected.rows;
        out["input_bytes"] = inspected.input_bytes;
        out["data_offset"] = inspected.data_offset;
        out["chunk_offsets"] = inspected.chunk_offsets;
        out["chunk_row_counts"] = inspected.chunk_row_counts;
        out["chunk_string_bytes"] = inspected.chunk_string_bytes;
        out["string_bytes"] = inspected.string_bytes;
        return out;
    }, py::arg("path"), py::arg("schema"), py::arg("has_header") = true, py::arg("delimiter") = ",",
       py::arg("threads") = 0, py::arg("target_chunk_bytes") = 4 * 1024 * 1024,
       py::arg("sequential_io_hint") = true,
       "Count rows and packed string bytes per chunk, in parallel, without materializing values.");

    m.def("read_into", [](const std::string& path, const py::list& schema, const py::list& targets,
                          bool has_header, const std::string& delimiter, std::size_t expected_rows,
                          const std::vector<std::uint64_t>& chunk_offsets,
                          const std::vector<std::uint64_t>& chunk_row_counts,
                          const std::vector<std::vector<std::uint64_t>>& chunk_string_bytes,
                          std::size_t threads, bool sequential_io_hint, bool empty_float_is_nan) {
        const auto cpp_schema = convert_schema(schema);
        const auto options = read_options(has_header, delimiter, threads, sequential_io_hint, empty_float_is_nan);
        if (targets.size() != cpp_schema.size()) throw std::invalid_argument("one target per schema column is required");

        std::vector<py::buffer_info> keep;
        keep.reserve(targets.size() * 2);
        std::vector<fastcsv::ReadColumnView> views(cpp_schema.size());
        std::size_t string_ord = 0;
        for (std::size_t c = 0; c < cpp_schema.size(); ++c) {
            const auto& spec = cpp_schema[c];
            const std::string what = "column '" + spec.name + "'";
            views[c].type = spec.type;
            if (spec.type == fastcsv::Type::Skip) continue;
            if (spec.type == fastcsv::Type::String) {
                std::uint64_t total = 0;
                for (const auto& chunk : chunk_string_bytes) {
                    if (string_ord >= chunk.size()) throw std::invalid_argument("chunk_string_bytes column count mismatch");
                    total += chunk[string_ord];
                }
                const auto packed = packed_views(keep, py::cast<py::tuple>(targets[c]), true, what);
                if (packed.offsets_length < expected_rows + 1)
                    throw std::invalid_argument(what + ": offsets buffer needs rows+1 entries");
                if (packed.bytes_length < total)
                    throw std::invalid_argument(what + ": data buffer is smaller than the packed string bytes");
                views[c].string_offsets = packed.offsets;
                views[c].string_bytes = packed.bytes;
                ++string_ord;
            } else {
                keep.emplace_back(py::cast<py::buffer>(targets[c]).request(true));
                const NumericView view = numeric_view(keep.back(), spec.type, what);
                if (view.length < expected_rows) throw std::invalid_argument(what + ": target buffer has fewer than rows elements");
                views[c].numeric = view.ptr;
                views[c].stride = view.stride;
            }
        }

        fastcsv::ReadStats stats;
        {
            py::gil_scoped_release release;
            stats = fastcsv::read_strict_csv_into(path, cpp_schema, views, options, expected_rows,
                                                  chunk_offsets, chunk_row_counts, chunk_string_bytes);
        }
        return stats_dict(stats);
    }, py::arg("path"), py::arg("schema"), py::arg("targets"), py::arg("has_header") = true,
       py::arg("delimiter") = ",", py::arg("expected_rows") = 0,
       py::arg("chunk_offsets") = std::vector<std::uint64_t>{},
       py::arg("chunk_row_counts") = std::vector<std::uint64_t>{},
       py::arg("chunk_string_bytes") = std::vector<std::vector<std::uint64_t>>{},
       py::arg("threads") = 0, py::arg("sequential_io_hint") = true, py::arg("empty_float_is_nan") = true,
       "Parse columns straight into caller-owned buffers (numeric 1-D views or (offsets, data) tuples).");

    m.def("read_matrix_into", [](const std::string& path, py::buffer matrix, bool has_header,
                                 const std::string& delimiter, std::size_t expected_rows,
                                 const std::vector<std::uint64_t>& chunk_offsets,
                                 const std::vector<std::uint64_t>& chunk_row_counts,
                                 std::size_t threads, bool sequential_io_hint, bool empty_float_is_nan,
                                 const std::vector<std::string>& names) {
        const auto options = read_options(has_header, delimiter, threads, sequential_io_hint, empty_float_is_nan);
        py::buffer_info info = matrix.request(true);
        if (info.ndim != 2) throw std::invalid_argument("matrix target must be 2-D");
        const auto type = type_from_buffer(info);
        const std::size_t columns = static_cast<std::size_t>(info.shape[1]);
        if (columns == 0) throw std::invalid_argument("matrix target must have at least one column");
        if (static_cast<std::size_t>(info.shape[0]) < expected_rows)
            throw std::invalid_argument("matrix target has fewer rows than the CSV");
        if (info.strides[1] != info.itemsize || info.strides[0] != info.itemsize * info.shape[1])
            throw std::invalid_argument("matrix target must be C-contiguous (row-major)");

        fastcsv::ReadStats stats;
        {
            py::gil_scoped_release release;
            stats = fastcsv::read_strict_numeric_matrix_into(path, type, columns, info.ptr, options, expected_rows,
                                                             chunk_offsets, chunk_row_counts, names);
        }
        return stats_dict(stats);
    }, py::arg("path"), py::arg("matrix"), py::arg("has_header") = true, py::arg("delimiter") = ",",
       py::arg("expected_rows") = 0,
       py::arg("chunk_offsets") = std::vector<std::uint64_t>{},
       py::arg("chunk_row_counts") = std::vector<std::uint64_t>{},
       py::arg("threads") = 0, py::arg("sequential_io_hint") = true, py::arg("empty_float_is_nan") = true,
       py::arg("names") = std::vector<std::string>{},
       "Parse a homogeneous numeric CSV straight into a row-major 2-D buffer.");

    m.def("write", [](const std::string& path, const py::list& schema, const py::list& columns, std::size_t rows,
                      const std::string& delimiter, bool header, std::size_t chunk_rows,
                      std::size_t target_chunk_bytes, std::size_t threads, std::size_t block_rows,
                      int float_precision) {
        const auto cpp_schema = convert_schema(schema);
        if (columns.size() != cpp_schema.size()) throw std::invalid_argument("one column per schema entry is required");
        fastcsv::WriteOptions options;
        options.delimiter = one_byte_delimiter(delimiter);
        options.header = header;
        options.chunk_rows = chunk_rows;
        options.target_chunk_bytes = target_chunk_bytes;
        options.threads = threads;
        options.block_rows = block_rows;
        options.float_precision = float_precision;

        std::vector<py::buffer_info> keep;
        keep.reserve(columns.size() * 2);
        std::vector<std::unique_ptr<PackedOwner>> owners;
        std::vector<fastcsv::WriteColumnView> views(cpp_schema.size());
        for (std::size_t c = 0; c < cpp_schema.size(); ++c) {
            const auto& spec = cpp_schema[c];
            const std::string what = "column '" + spec.name + "'";
            views[c].type = spec.type;
            if (spec.type == fastcsv::Type::Skip) throw std::invalid_argument(what + ": skip columns cannot be written");
            if (spec.type == fastcsv::Type::String) {
                if (py::isinstance<py::tuple>(columns[c])) {
                    const auto packed = packed_views(keep, py::cast<py::tuple>(columns[c]), false, what);
                    if (packed.offsets_length < rows + 1) throw std::invalid_argument(what + ": offsets buffer needs rows+1 entries");
                    if (rows && packed.offsets[rows] > packed.bytes_length)
                        throw std::invalid_argument(what + ": packed offsets exceed the data buffer");
                    views[c].strings = {packed.offsets, packed.bytes};
                } else {
                    owners.push_back(pack_python_strings(columns[c], rows, options.delimiter, what));
                    views[c].strings = {owners.back()->offsets.data(), owners.back()->bytes.data()};
                }
            } else {
                keep.emplace_back(py::cast<py::buffer>(columns[c]).request(false));
                const NumericView view = numeric_view(keep.back(), spec.type, what);
                if (view.length < rows) throw std::invalid_argument(what + ": buffer has fewer than rows elements");
                views[c].numeric = view.ptr;
                views[c].stride = view.stride;
            }
        }

        fastcsv::WriteResult result;
        {
            py::gil_scoped_release release;
            result = fastcsv::write_canonical_csv(path, cpp_schema, views, rows, options);
        }
        py::dict out;
        out["rows"] = result.rows;
        out["bytes_written"] = result.bytes_written;
        out["workers"] = result.workers;
        out["chunk_offsets"] = result.chunk_offsets;
        out["chunk_row_counts"] = result.chunk_row_counts;
        out["chunk_byte_counts"] = result.chunk_byte_counts;
        out["chunk_string_bytes"] = result.chunk_string_bytes;
        return out;
    }, py::arg("path"), py::arg("schema"), py::arg("columns"), py::arg("rows"), py::arg("delimiter") = ",",
       py::arg("header") = true, py::arg("chunk_rows") = 0, py::arg("target_chunk_bytes") = 4 * 1024 * 1024,
       py::arg("threads") = 0, py::arg("block_rows") = 0, py::arg("float_precision") = -1,
       "Write canonical CSV from numeric buffers (strided views allowed) and packed or Python strings.");

    m.def("fill_object_array", [](py::buffer offsets_buf, py::object data_obj, py::array out) {
        py::buffer_info offsets = offsets_buf.request(false);
        if (offsets.ndim != 1 || !is_uint64_buffer(offsets) || offsets.strides[0] != 8)
            throw std::invalid_argument("offsets must be a contiguous 1-D uint64 buffer");
        const auto* off = static_cast<const std::uint64_t*>(offsets.ptr);
        const std::size_t n = offsets.shape[0] ? static_cast<std::size_t>(offsets.shape[0]) - 1 : 0;
        const char* bytes = "";
        std::size_t nbytes = 0;
        py::buffer_info data;
        if (!data_obj.is_none()) {
            data = py::cast<py::buffer>(data_obj).request(false);
            if (data.ndim != 1 || !is_byte_buffer(data) || data.strides[0] != 1)
                throw std::invalid_argument("data must be a contiguous 1-D uint8 buffer");
            bytes = static_cast<const char*>(data.ptr);
            nbytes = static_cast<std::size_t>(data.shape[0]);
        }
        if (out.ndim() != 1 || static_cast<std::size_t>(out.shape(0)) != n || !out.writeable() ||
            out.dtype().kind() != 'O')
            throw std::invalid_argument("out must be a writable 1-D object array with one slot per string");
        auto** slots = static_cast<PyObject**>(out.mutable_data());
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t b = off[i], e = off[i + 1];
            if (e < b || e > nbytes) throw std::invalid_argument("packed offsets exceed the data buffer");
            PyObject* s = PyUnicode_DecodeUTF8(bytes + b, static_cast<Py_ssize_t>(e - b), nullptr);
            if (!s) throw py::error_already_set();
            PyObject* old = slots[i];
            slots[i] = s;
            Py_XDECREF(old);
        }
        return out;
    }, py::arg("offsets"), py::arg("data"), py::arg("out"),
       "Decode packed UTF-8 strings into a preallocated NumPy object array.");
}
