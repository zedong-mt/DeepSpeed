#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>
#include <torch/extension.h>
#include <chrono>
#include <pybind11/embed.h>
namespace py = pybind11;

#include <c10/util/irange.h>

#include <iostream>
#include <string>

#include <comm.h>

// TODO: remove
#include <stdio.h>

#define MPICHECK(cmd)                                                        \
    do {                                                                     \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS) {                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDACHECK(cmd)                                                                            \
    do {                                                                                          \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess) {                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define NCCLCHECK(cmd)                                                                           \
    do {                                                                                         \
        ncclResult_t ret = cmd;                                                                  \
        if (ret != ncclSuccess) {                                                                \
            printf(                                                                              \
                "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(ret)); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

#define CUDA_STREAM_SYNCHRONIZE(_nccl_stream)                                            \
    do {                                                                                 \
        cudaError_t err = cudaErrorNotReady;                                             \
        int flag;                                                                        \
        while (err == cudaErrorNotReady) {                                               \
            err = cudaStreamQuery(_nccl_stream);                                         \
            MPICHECK(MPI_Iprobe(                                                         \
                MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE)); \
        }                                                                                \
        CUDACHECK(err);                                                                  \
    } while (0)

namespace nccl {

int counter = 0;
cudaStream_t s;
ncclComm_t ncclcomm;

//py::module_ dist = py::module_::import("deepspeed.comm");

std::vector<MPI_Comm> global_mpi_comms;
std::vector<ncclComm_t> global_nccl_comms;
std::vector<cudaStream_t> global_streams;

int get_rank(int group = 0)
{
    int world_rank;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    return world_rank;
}

int get_world_size(int group = 0)
{
    int world_size;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    return world_size;
}

// Given a ncclUniqueId, convert it to a string representation that can be put
// in the store.
std::string buildNcclUniqueIdStr(const ncclUniqueId& ncclID)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
    std::ostringstream oss;
    for (const auto i : c10::irange(NCCL_UNIQUE_ID_BYTES)) {
        oss << std::hex << static_cast<int>(bytes[i]);
    }
    return oss.str();
}

std::string getNcclId()
{
    ncclUniqueId ncclID;
    NCCLCHECK(ncclGetUniqueId(&ncclID));
    return buildNcclUniqueIdStr(ncclID);

    // std::string id = "hello";
    // for (int i=0; i<128; i++)
    //     std::cout << "ncclID =" << ncclID[i];
    // std::cout<< std::endl;
    // return id;
}

void barrier() { MPICHECK(MPI_Barrier(MPI_COMM_WORLD)); }

void create_comms(int number = 1)
{
    ncclUniqueId ncclID;
    int world_rank = get_rank(0);
    int world_size = get_world_size(0);
    int ngpus;

    CUDACHECK(cudaGetDeviceCount(&ngpus));

    CUDACHECK(cudaSetDevice(world_rank % ngpus));
    CUDACHECK(cudaStreamCreate(&s));
    if (world_rank == 0) { ncclGetUniqueId(&ncclID); }
    MPICHECK(MPI_Bcast(&ncclID, sizeof(ncclID), MPI_BYTE, 0, MPI_COMM_WORLD));

    NCCLCHECK(ncclCommInitRank(&ncclcomm, world_size, ncclID, world_rank));
}

void print_comm_number() { std::cout << "Number of Comms:" << global_mpi_comms.size() << "\n"; }

void increase_counter() { counter++; }

void decrease_counter() { counter--; }

void print_counter() { std::cout << "Counter is:" << counter << "\n"; }

void initialize_nccl(int rank, int size)
{
    //initialize_mpi();
    create_comms();
}

void finalize_nccl()
{
    NCCLCHECK(ncclCommDestroy(ncclcomm));
    //finalize_mpi();
}


ncclDataType_t get_nccl_datatype(c10::ScalarType type)
{
    ncclDataType_t nccl_type;
    switch (type) {
        case c10::ScalarType::Int: nccl_type = ncclInt; break;
        case c10::ScalarType::Float: nccl_type = ncclFloat; break;
        case c10::ScalarType::Double: nccl_type = ncclDouble; break;
        default: nccl_type = ncclChar;
    }
    return nccl_type;
}


ncclRedOp_t get_nccl_reduce_op(py::object op, at::Tensor& input)
{
    py::object ReduceOp = py::module_::import("deepspeed.comm").attr("ReduceOp");
    if (!py::isinstance(op, ReduceOp)) {
        throw std::runtime_error ("Error: Op must be of type ReduceOp");
    }

    int op_val = py::int_(op.attr("value"));
    ncclRedOp_t nccl_op;

    if (input.scalar_type() == at::kBool) {
        if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
            // For bool tensors, map sum to max, which both represent a bitwise or.
            // This is to prevent overflow issues with sum, since we use uint8 to
            // represent a bool (see ncclDataType mapping).
            nccl_op = ncclMax;
        } else if (op_val == (int)py::int_(ReduceOp.attr("AVG").attr("value"))) {
            throw std::runtime_error ("Error: For bool tensors, op must be of type ReduceOp");
        }
    }

    if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
        nccl_op = ncclSum;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MIN").attr("value"))) {
        nccl_op = ncclMin;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MAX").attr("value"))) {
        nccl_op = ncclMax;
    } else if (op_val == (int)py::int_(ReduceOp.attr("PRODUCT").attr("value"))) {
        nccl_op = ncclProd;
    //} else if (op_val == (int)py::int_(ReduceOp.attr("AVERAGE").attr("value"))) {
    //    nccl_op = ncclAvg;
    } else {
        throw std::runtime_error ("Error: Unrecognized ReduceOp type");
    }
    return nccl_op;
}

void send(torch::Tensor data, int rank, int tag)
{
    NCCLCHECK(ncclSend(
        data.data_ptr(), data.numel(), get_nccl_datatype(data.scalar_type()), rank, ncclcomm, s));
    //CUDACHECK(cudaStreamSynchronize(s));
}

void recv(torch::Tensor data, int rank, int tag)
{
    NCCLCHECK(ncclRecv(
        data.data_ptr(), data.numel(), get_nccl_datatype(data.scalar_type()), rank, ncclcomm, s));
    //CUDACHECK(cudaStreamSynchronize(s));
}


//TODO: implement torch's async_op behavior, document it.
void allreduce(torch::Tensor& data, py::object op, bool async_op)
{

    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclAllReduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_nccl_datatype(data.scalar_type()),
                            get_nccl_reduce_op(op, data),
                            ncclcomm,
                            s));
    if (!async_op) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

//TODO: implement torch's async_op behavior, document it.
void allgather(torch::Tensor& output, torch::Tensor& input, bool async_op)
{
    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclAllGather(input.data_ptr(),
                            output.data_ptr(),
                            input.numel(),
                            get_nccl_datatype(input.scalar_type()),
                            ncclcomm,
                            s));
    if (!async_op) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

inline at::Tensor newLikeFlat(
    std::vector<std::vector<at::Tensor>>& tensors,
    size_t deviceIdx) {
  if (tensors.size() == 0 || tensors[0].size() == 0) {
    throw std::runtime_error ("Received an empty list");
  }
  if (deviceIdx >= tensors.size()) {
    throw std::runtime_error ("Invalid device index");
  }
  auto& t = tensors[deviceIdx][0];
  auto device = t.device();
  for (const auto i : c10::irange(1, tensors[deviceIdx].size())) {
    if (tensors[deviceIdx][i].device() != device) {
      throw std::runtime_error ("Expecting all tensors on the same device");
    }
  }
  at::DeviceGuard gpuGuard(device);
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors[deviceIdx].size())};
  std::vector<int64_t> strides{static_cast<int64_t>(t.numel())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  strides.insert(strides.end(), t.strides().begin(), t.strides().end());
  return at::empty_strided(
      sizes, strides, t.options().memory_format(c10::nullopt));
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
//std::vector<at::Tensor> flatten_for_scatter_gather(
//    std::vector<std::vector<at::Tensor>>& tensor_lists,
//    size_t world_size) {
//  const auto num_devices = tensor_lists.size();
//
//  std::vector<at::Tensor> flattened;
//  flattened.resize(num_devices);
//
//  for (const auto i : c10::irange(size_t{}, num_devices)) {
//    // Flatten the tensors (from all ranks) into a single big tensor.
//    flattened[i] = newLikeFlat(tensor_lists, i);
//  }
//  return flattened;
//}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error ("Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (const auto i : c10::irange(size_t{}, num_devices)) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error ("Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error ("Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error ("All tensor operands to scatter/gather must have the same number of elements");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}


//void coll_

//TODO: implement torch's async_op behavior, document it.
void allgather_list(std::vector<std::vector<torch::Tensor>>& outputTensors, std::vector<torch::Tensor>& inputTensors, bool async_op)
{
    //std::vector<at::Tensor> flattenOutputTensors;
    //flattenOutputTensors.resize(outputTensors.size());
//
    //for (size_t i = 0; i < outputTensors.size(); ++i) {
    //  // Flatten the output tensors (from all ranks) to a single big tensor
    //  flattenOutputTensors[i] = newLikeFlat(outputTensors);
    //}

      auto outputFlattened = flatten_for_scatter_gather(outputTensors, inputTensors, get_world_size(0));

    
    NCCLCHECK(ncclGroupStart());

    for (size_t i = 0; i < inputTensors.size(); ++i) {

        NCCLCHECK(ncclAllGather(
            inputTensors[i].data_ptr(),
            outputFlattened[i].data_ptr(),
            inputTensors[i].numel(),
            get_nccl_datatype(inputTensors[i].scalar_type()),
            ncclcomm,
            s));
        }

    NCCLCHECK(ncclGroupEnd());




    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    //NCCLCHECK(ncclAllGather(input.data_ptr(),
    //                        output.data_ptr(),
    //                        input.numel(),
    //                        get_nccl_datatype(data.scalar_type()),
    //                        ncclcomm,
    //                        s));
    if (!async_op) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}

    for (const auto i : c10::irange(outputTensors.size())) {
          //at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
          for (const auto j : c10::irange(outputTensors[0].size())) {
            outputTensors[i][j].copy_(outputFlattened[i][j], true);
          }
        }
}

//TODO: implement torch's async_op behavior, document it.
void reduce(torch::Tensor& data, int root, py::object op, bool async_op)
{

    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclReduce(data.data_ptr(),
                         data.data_ptr(),
                         data.numel(),
                         get_nccl_datatype(data.scalar_type()),
                         get_nccl_reduce_op(op, data),
                         root,
                         ncclcomm,
                         s));
    //if (!async_op) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

//TODO: implement torch's async_op behavior, document it.
void reduce_scatter(torch::Tensor& data, py::object op, bool async_op)
{
    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclReduceScatter(data.data_ptr(),
                                data.data_ptr(),
                                data.numel(),
                                get_nccl_datatype(data.scalar_type()),
                                get_nccl_reduce_op(op, data),
                                ncclcomm,
                                s));
    //if (!async_op) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

void bcast(torch::Tensor& data, int src)
{
    NCCLCHECK(ncclBroadcast(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_nccl_datatype(data.scalar_type()),
                            src,
                            ncclcomm,
                            s));
}

void alltoall(torch::Tensor outputTensor, torch::Tensor inputTensor, bool async_op)
{
    //std::chrono::steady_clock::time_point begin, end;
    const auto* sendbuff = reinterpret_cast<char*>(inputTensor.data_ptr());
    auto* recvbuff = reinterpret_cast<char*>(outputTensor.data_ptr());
    int nRanks;
    NCCLCHECK(ncclCommCount(ncclcomm, &nRanks));
    size_t rankdiff = inputTensor.nbytes() / nRanks;
    //if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclGroupStart());
    int count = inputTensor.numel() / nRanks;
    ncclDataType_t type = get_nccl_datatype(inputTensor.scalar_type());
    for (int r = 0; r < nRanks; r++) {
        if (count != 0) {
            NCCLCHECK(ncclSend(sendbuff + r * rankdiff, count, type, r, ncclcomm, s));
            NCCLCHECK(ncclRecv(recvbuff + r * rankdiff, count, type, r, ncclcomm, s));
        }
    }
    NCCLCHECK(ncclGroupEnd());
    //if (!async_op) { CUDACHECK(cudaStreamSynchronize(s)); }
    // CUDACHECK(cudaStreamSynchronize(s));
    //if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL alltoall time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
    //                  << " us, Size = " << inputTensor.numel() * inputTensor.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

void alltoall_list(std::vector<torch::Tensor>& inputTensors,
                        std::vector<torch::Tensor>& outputTensors)
{
    NCCLCHECK(ncclGroupStart());
    for (int t = 0; t < inputTensors.size(); t++) {
        torch::Tensor& input = inputTensors[t];
        torch::Tensor& output = outputTensors[t];
        if (input.numel() != 0) {
            NCCLCHECK(ncclSend(input.data_ptr(),
                               input.numel(),
                               get_nccl_datatype(input.scalar_type()),
                               t,
                               ncclcomm,
                               s));
        }
        if (output.numel() != 0) {
            NCCLCHECK(ncclRecv(output.data_ptr(),
                               output.numel(),
                               get_nccl_datatype(output.scalar_type()),
                               t,
                               ncclcomm,
                               s));
        }
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(s));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("send", &send, "nccl send");
    m.def("recv", &recv, "nccl recv");
    m.def("allreduce", &allreduce, "nccl allreduce");
    m.def("bcast", &bcast, "nccl broadcast");
    m.def("alltoall", &alltoall, "nccl alltoall");
    m.def("alltoall_list", &alltoall_list, "nccl alltoall list");
    m.def("allgather", &allgather, "nccl allgather");
    m.def("allgather_list", &allgather_list, "nccl allgather list");
    m.def("reduce", &reduce, "nccl reduce");
    m.def("reduce_scatter", &reduce_scatter, "nccl reduce scatter");
    m.def("initialize_nccl", &initialize_nccl, "nccl initialize");
    m.def("finalize_nccl", &finalize_nccl, "nccl finalize");
    m.def("getNcclId", &getNcclId, "Get Unique NCCL ID");
    m.def("get_rank", &get_rank, "get rank");
    m.def("barrier", &barrier, "barrier");
    m.def("get_world_size", &get_world_size, "get world size");
    m.def("increase_counter", &increase_counter, "mpi increase counter");
    m.def("decrease_counter", &decrease_counter, "mpi decrease counter");
    m.def("print_counter", &print_counter, "mpi print counter");
    // m.def("create_comms", &create_comms, "nccl create comms");
    m.def("print_comm_number", &print_comm_number, "mpi print comm number");
}

} // namespace nccl