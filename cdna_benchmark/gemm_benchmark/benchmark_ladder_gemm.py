import welder
from welder.graph import IRNode, OutputNode
from welder.policy import *
import os
from tvm.script import tir as T
from tvm import te
from welder.te_utils import connect_tensor_graph
import time
# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname
arch = "MI300"
arch = welder.arch.__getattribute__(arch)()

shapes = [
    [1, 1024, 4096],
    [1, 4096, 4096],
    [1, 14336, 4096],
    [1, 4096, 14336],
    [32, 1024, 4096],
    [32, 4096, 4096],
    [32, 14336, 4096],
    [32, 4096, 14336],
    [4096, 1024, 4096],
    [4096, 4096, 4096],
    [4096, 14336, 4096],
    [4096, 4096, 14336],
    [1, 1024, 8192],
    [1, 8192, 8192],
    [1, 28672, 8192],
    [1, 8192, 28672],
    [32, 1024, 8192],
    [32, 8192, 8192],
    [32, 28672, 8192],
    [32, 8192, 28672],
    [4096, 1024, 8192],
    [4096, 8192, 8192],
    [4096, 28672, 8192],
    [4096, 8192, 28672],
]

wmma_m = 16
wmma_n = 16
wmma_k = 16
# shapes = ft_shapes
# shapes = llm_shapes
perf_map = {}
cost_map = {}
# out_dtype = 'float32'
out_dtype = 'float32'

from tvm.contrib.popen_pool import PopenPoolExecutor
def select_best(output_nodes, compile_results):
    for cpresult in compile_results:
        print(cpresult.config)
        if cpresult.lib_name is None:
            cpresult.latency = 1e8
        else:
            profiler = PopenPoolExecutor(max_workers=1, timeout=None, initializer=welder.engine.profiler.init_server, initargs=[arch])
            future = profiler.submit(welder.engine.profiler.call_profile, cpresult.lib_name, cpresult.args, "cuda:0")
            try:
                cpresult.latency = future.result()
            except ChildProcessError as e:
                cpresult.latency = 1e8
            finally:
                cpresult.remove_lib()
        print(cpresult.latency)
    compile_results = list(filter(lambda x:x.latency<1e8, compile_results))
    compile_results = sorted(compile_results, key=lambda x:x.latency)
    if len(compile_results) == 0:
        return None

    print(f"Best Config: {compile_results[0].config}")
    print(f"result: {compile_results[0].latency}")
    return compile_results[0]

for M, N, K in shapes:

    def ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k):
        A = te.placeholder((M // wmma_m, K // wmma_k, wmma_m ,wmma_k), name='A', dtype='float16')
        B = te.placeholder((N // wmma_n, K // wmma_k, wmma_n, wmma_k), name='B', dtype='float16')
        
        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K // wmma_k), name='k')
        kk = te.reduce_axis((0, wmma_k), name='kk')
        C = te.compute(
            (M // wmma_m, N // wmma_n, wmma_m, wmma_n),
            lambda i, j, ii, jj: te.sum(A[i, k, ii, kk].astype(out_dtype) * B[j, k, jj, kk].astype(out_dtype), axis=[k, kk]),
            # lambda i, j, ii, jj: te.sum(A[i, k, ii, kk] * B[j, k, jj, kk], axis=[k, kk]),
            name='C'
        )
        return A, B, C
    
    def gemm(M, N, K):
        A = te.placeholder((M, K), name='A', dtype='float16')
        B = te.placeholder((N, K), name='B', dtype='float16')

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K), name='k')
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k] * B[j, k], axis=k),
            name='C'
        )

        return A, B, C

    if M == 1:
        arg1 = gemm(M, N, K)
        args = arg1
        input_args = args[:-1]
        output_args = [args[-1]]

        node = IRNode([None for _ in input_args], arg1, "ladder_matmul")
        output_nodes = [OutputNode(node)]
        policy = DefaultPolicy(output_nodes, arch)
        configs = policy.emit_config(20)
    else:
        arg1 = ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k)
        args = arg1

        input_args = args[:2]
        output_args = [args[-1]]
        node = IRNode([None for _ in input_args], args, "ladder_matmul")
        node.add_tag("tensorCoreConfig", [2, 3])
        node.add_tag("ladder_config", (True, True))
        output_nodes = [OutputNode(node)]
        compile_end = time.time()
        policy = LadderPolicy(output_nodes, arch)
        configs = policy.emit_config(20)

    compile_results = []
    cgen = welder.CodeGenerator()
    for config in configs:
        try:
            cpresult = cgen.compile(output_nodes, config, "hip", kernel_name="Fused")
            compile_results.append(cpresult)
        except Exception as e:
            print(e)
            continue
    welder.utils.compile_parallel(compile_results, arch, timeout=30)
    best = select_best(output_nodes, compile_results)
    best_latency = best.latency

    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))

    # example output
    # print(best.get_example_outputs())
    key = "{}_{}_{}".format(M, N, K)
    perf_map[key] = best_latency

print("perf_map: {}".format(perf_map))
print("M\tN\tK\tperf\tcost")

for M, N, K in shapes:
    key = "{}_{}_{}".format(M, N, K)
    print("{}\t{}\t".format(key, perf_map[key]))
