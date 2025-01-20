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
    # [16384, 16384, 16384] ,
    # [8192, 43008, 14336],
    # [8192, 14336, 14336],
    # [8192, 57344, 14336],
    [32, 2560, 4280],
    # [8192, 9216, 9216],
    # [8192, 36864, 9216],
    # [8192, 9216, 36864],
    # [8192, 22016, 8192],
    # [8192, 8192, 22016],
    # [8192, 8192, 8192], # 5.76 ms in peak 191 TFLOPs
    # [8192*16, 128, 8192], # 
    # [128, 8192*32, 8192], # 
    # [8192, 28672, 8192],
    # [8192, 8192, 28672],
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

    def ladder_gemm(M, N, K):
        A = te.placeholder((M, K), name='A', dtype='float16')
        B = te.placeholder((N, K), name='B', dtype='float16')
        
        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K), name='k')
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k].astype("float16") * B[j, k].astype("float16"), axis=[k]),
            # lambda i, j, ii, jj: te.sum(A[i, k, ii, kk] * B[j, k, jj, kk], axis=[k, kk]),
            name='C'
        )
        return A, B, C

    def cast(M, N, wmma_m, wmma_n):
        C = te.placeholder((M // wmma_m, N // wmma_n, wmma_m, wmma_n), name='C', dtype=out_dtype)
        C_reshape = te.compute(
            (M // wmma_m, N // wmma_n, wmma_m, wmma_n),
            lambda i, j, ii, jj: C[i, j, ii, jj].astype('float16'),
            name='C_reshape'
        )
        return C, C_reshape

    def reshape(M, N, wmma_m, wmma_n):
        C = te.placeholder((M // wmma_m, N // wmma_n, wmma_m, wmma_n), name='C', dtype='float16')
        C_reshape = te.compute(
            (M, N),
            lambda i, j: C[i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n],
            name='C_reshape'
        )
        return C, C_reshape

    arg1 = ladder_gemm(M, N, K)
    args = arg1
    # args = tuple(connect_tensor_graph(args, arg2, {arg2[0]:args[2]}))
    # args = tuple(connect_tensor_graph(args, arg3, {arg3[0]:args[2]}))

    input_args = args[:2]
    output_args = [args[-1]]
    node = IRNode([None for _ in input_args], args, "ladder_matmul")
    node.add_tag("tensorCoreConfig", [0, 1])
    node.add_tag("ladder_config", (False, False))
    output_nodes = [OutputNode(node)]
    compile_end = time.time()
    policy = LadderPolicy(output_nodes, arch)
    
    configs = policy.emit_config(20)
    print(f"configs: {len(configs)}")
    for config in configs:
        print(config)

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
    compile_end = time.time() - compile_end
    best_latency = best.latency

    # best_latency = 10000
    # best = None
    # values = []
    # for cpresult in compile_results:
    #     print(cpresult.config)
    #     # code = cpresult.code
    #     # code = cpresult.profiling_code
    #     if cpresult.lib is None:
    #         latency = 10000
    #     else:
    #         latency = cpresult.profile()
    #     values.append(latency)
    #     if latency < best_latency:
    #         best_latency = latency
    #         best = cpresult
    #     print(latency)
    
    code= best.profiling_code
    # print(best.profiling_code)
    with open("best_code.cu", "w+") as f:
        f.write(code)
    # print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    print("best block_size: {}".format(best.block_size))
    # example output
    # print(best.get_example_outputs())
    key = "{}_{}_{}".format(M, N, K)
    perf_map[key] = best_latency
    cost_map[key] = compile_end

print("perf_map: {}".format(perf_map))
print("cost_map: {}".format(cost_map))
print("M\tN\tK\tperf\tcost")

for M, N, K in shapes:
    key = "{}_{}_{}".format(M, N, K)
    print("{}\t{}\t{}".format(key, perf_map[key], cost_map[key]))