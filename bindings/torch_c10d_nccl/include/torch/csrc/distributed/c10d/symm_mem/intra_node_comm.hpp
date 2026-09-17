#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d::intra_node_comm {

enum class Topology : uint8_t {
  UNKNOWN = 0,
  FULLY_CONNECTED = 1,
};

enum class AllReduceAlgo : uint8_t {
  NONE = 0,
  ONE_SHOT = 1,
  TWO_SHOT = 2,
};

class IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      c10::intrusive_ptr<c10d::Store> store,
      size_t rank,
      size_t worldSize,
      std::optional<size_t> bufferSize = std::nullopt,
      std::string groupName = "") {
    (void)store;
    (void)rank;
    (void)worldSize;
    (void)bufferSize;
    (void)groupName;
  }
  virtual ~IntraNodeComm() override = default;

  static bool isEnabled() {
    return false;
  }

  bool rendezvous() {
    return false;
  }

  AllReduceAlgo selectAllReduceAlgo(const at::Tensor& input) {
    (void)input;
    return AllReduceAlgo::NONE;
  }

  at::Tensor allReduce(const at::Tensor& input, AllReduceAlgo algo) {
    (void)algo;
    return input;
  }
};

class IntraNodeCommWork : public c10d::Work {
 public:
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    (void)timeout;
    return true;
  }
};

inline int64_t getIntraNodeCommUsageCounter() {
  return 0;
}

inline bool isIntraNodeCommSupported() {
  return false;
}

} // namespace c10d::intra_node_comm
