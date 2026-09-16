#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <pybind11/chrono.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::object c10d = py::module_::import("torch._C._distributed_c10d");
    py::object backend = c10d.attr("Backend");
    py::object backendOptions = c10d.attr("Backend").attr("Options");

    py::class_<::c10d::ProcessGroupNCCL, c10::intrusive_ptr<::c10d::ProcessGroupNCCL>> pg(
        m, "ProcessGroupNCCL", backend);

    pg.def(py::init<
        const c10::intrusive_ptr<::c10d::Store>&,
        int,
        int,
        c10::intrusive_ptr<::c10d::ProcessGroupNCCL::Options>>(),
        py::call_guard<py::gil_scoped_release>(),
        py::arg("store"),
        py::arg("rank"),
        py::arg("size"),
        py::arg("options"))
      .def(py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                       int rank,
                       int size,
                       const std::chrono::milliseconds& timeout) {
          auto options = ::c10d::ProcessGroupNCCL::Options::create();
          options->is_high_priority_stream = false;
          options->timeout = timeout;
          return c10::make_intrusive<::c10d::ProcessGroupNCCL>(store, rank, size, options);
      }),
      py::arg("store"),
      py::arg("rank"),
      py::arg("size"),
      py::arg("timeout") = ::c10d::kProcessGroupNCCLDefaultTimeout,
      py::call_guard<py::gil_scoped_release>())
      .def("_group_start", &::c10d::ProcessGroupNCCL::groupStart)
      .def("_group_end", &::c10d::ProcessGroupNCCL::groupEnd)
      .def_property_readonly("options", &::c10d::ProcessGroupNCCL::getOptions)
      .def_property_readonly("uid", &::c10d::ProcessGroupNCCL::getUid)
      .def("abort", &::c10d::ProcessGroupNCCL::abort, py::call_guard<py::gil_scoped_release>())
      .def("_shutdown", [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self) {
          return self->shutdown();
      }, py::call_guard<py::gil_scoped_release>())
      .def("_is_initialized", &::c10d::ProcessGroupNCCL::isInitialized, py::call_guard<py::gil_scoped_release>());

    py::class_<::c10d::ProcessGroupNCCL::Options, c10::intrusive_ptr<::c10d::ProcessGroupNCCL::Options>> opts(
        pg, "Options", backendOptions);
    opts.def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
        .def_readwrite("is_high_priority_stream", &::c10d::ProcessGroupNCCL::Options::is_high_priority_stream)
        .def_readwrite("split_from", &::c10d::ProcessGroupNCCL::Options::split_from)
        .def_readwrite("split_color", &::c10d::ProcessGroupNCCL::Options::split_color);
}
