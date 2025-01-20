import welder
from tvm import te, tir
from welder.graph import IRNode, OutputNode
from welder.policy import *
import logging
welder.set_log_level(logging.DEBUG)
import argparse

arch = 'MI300'
arch = welder.arch.__getattribute__(arch)()
dtype="float16"
out_dtype = 'float32'

from tvm.contrib.popen_pool import PopenPoolExecutor
def select_best(output_nodes, compile_results):
    for cpresult in compile_results:
        print(cpresult) 
    for cpresult in compile_results:
        if cpresult is None:
            continue
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

def conv_nhwc_hwnc(n, f, h, w, c, kh, kw, s, d, p):
    
    A = te.placeholder((n, h, w, c), name='input', dtype='float16')
    B = te.placeholder((kh*kw*c, f), name='weight', dtype='float16')
    pad_shape = (n, h + 2 * p, w + 2 * p, c)
    pad_value = tir.const(0.0, A.dtype)
    pad = te.compute(
                    pad_shape,
                    lambda n, h, w, c: te.if_then_else(
                        tir.all(
                            h >= p,
                            w >= p,
                            h < pad_shape[1] - p,
                            w < pad_shape[2] - p,
                        ),
                        A[n, h - p, w - p, c],
                        pad_value,
                    ),
                    name="pad",
                )
    kernel_h, kernel_w = kh, kw
    stride_h, stride_w = s, s
    dilation_h, dilation_w = d, d
    k_size = kernel_h * kernel_w * c
    k_axis = te.reduce_axis((0, k_size), name="k")
    out_h = (
        h + p + p - 1 - (kernel_h - 1) * dilation_h
    ) // stride_h + 1
    out_w = (
        w + p + p - 1 - (kernel_w - 1) * dilation_w
    ) // stride_w + 1
    n_size = out_h * out_w * n
    # Describe the matrix multiplication in TE
    data = te.compute(
                [n_size, k_size],
                lambda n, k: pad[
                    n // (out_h * out_w),
                    (n % (out_h * out_w) // out_w) * stride_h
                    + (k // (kernel_w * c)) * dilation_h,
                    (n % out_w) * stride_w + ((k // c) % kernel_w) * dilation_w,
                    k % c,
                ],
                name="data",
            )
    C = te.compute(
            [n_size, f],
            lambda i, j: te.sum(data[i, k_axis].astype('float16') * B[k_axis, j].astype('float16'), axis=[k_axis]),
            "T_conv",
        )
    return A, B, C

def ladder_conv_nhwc_hwnc(n, f, h, w, c, kh, kw, s, d, p, warp_i = 16, warp_j = 16, warp_k = 16, dtype="float16"):
    
    A = te.placeholder((n // warp_i, h, w, c // warp_k, warp_i, warp_k), name='input', dtype='float16')
    B = te.placeholder((kh*kw*c // warp_k, f // warp_j, warp_k, warp_j), name='weight', dtype='float16')
    pad_shape = (n // warp_i, h + 2 * p, w + 2 * p, c // warp_k, warp_i, warp_k)
    pad_value = tir.const(0.0, A.dtype)
    pad = te.compute(
                    pad_shape,
                    lambda n, h, w, c, nn, cc: te.if_then_else(
                        tir.all(
                            h >= p,
                            w >= p,
                            h < pad_shape[1] - p,
                            w < pad_shape[2] - p,
                        ),
                        A[n, h - p, w - p, c, nn, cc],
                        pad_value,
                    ),
                    name="pad",
                )
    kernel_h, kernel_w = kh, kw
    stride_h, stride_w = s, s
    dilation_h, dilation_w = d, d
    k_size = kernel_h * kernel_w * c
    k_axis = te.reduce_axis((0, k_size // warp_k), name="k")
    wk_axis = te.reduce_axis((0, warp_k), name="wk")
    out_h = (
        h + p + p - 1 - (kernel_h - 1) * dilation_h
    ) // stride_h + 1
    out_w = (
        w + p + p - 1 - (kernel_w - 1) * dilation_w
    ) // stride_w + 1
    n_size = out_h * out_w * n
    # Describe the matrix multiplication in TE
    data = te.compute(
                [n_size // warp_i, k_size // warp_k, warp_i, warp_k],
                lambda n, k, nn, kk: pad[
                    n // (out_h * out_w),
                    (n % (out_h * out_w) // out_w) * stride_h
                    + (k // (kernel_w * (c // warp_k))) * dilation_h,
                    (n % out_w) * stride_w + (k // (c // warp_k) % kernel_w) * dilation_w,
                    k % (c // warp_k),
                    nn,
                    kk,
                ],
                name="data",
            )
    C = te.compute(
            [n_size // warp_i, f // warp_j, warp_i, warp_j],
            lambda i, j, ii, jj: te.sum(data[i, k_axis, ii, wk_axis].astype(out_dtype) * B[k_axis, j, wk_axis, jj].astype(out_dtype), axis=[k_axis, wk_axis]),
            "T_conv",
        )
    return A, B, C

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=128, help='n')
    parser.add_argument('--c', type=int, default=128, help='c')
    parser.add_argument('--h', type=int, default=56, help='h')
    parser.add_argument('--w', type=int, default=56, help='w')
    parser.add_argument('--f', type=int, default=128, help='f')
    parser.add_argument('--k', type=int, default=3, help='k')
    parser.add_argument('--s', type=int, default=2, help='s')
    parser.add_argument('--d', type=int, default=1, help='d')
    parser.add_argument('--p', type=int, default=1, help='p')
    args = parser.parse_args()
    N, C, H, W, F, K, S, D, P = args.n, args.c, args.h, args.w, args.f, args.k, args.s, args.d, args.p
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    total_flops = 2 * N * C * OH * OW * F * K * K
    
    if N == 1:
        arg1 = conv_nhwc_hwnc(N, F, H, W, C, K, K, S, D, P)

        args = arg1

        input_args = args[:-1]
        output_args = [args[-1]]
        node = IRNode([None for _ in input_args], args, "cutlass_conv2d_reshape_bias")
        output_nodes = [OutputNode(node)]
        policy = DefaultPolicy(output_nodes, arch)
        configs = policy.emit_config(20)
    elif C == 3:
        arg1 = conv_nhwc_hwnc(N, F, H, W, C, K, K, S, D, P)

        args = arg1

        input_args = args[:-1]
        output_args = [args[-1]]
        node = IRNode([None for _ in input_args], args, "cutlass_conv2d_reshape_bias")
        node.add_tag("tensorCoreConfig", [0, 1])
        node.add_tag("ladder_config", (False, False, 1))
        output_nodes = [OutputNode(node)]
        policy = LadderPolicy(output_nodes, arch)
        configs = policy.emit_config(20)
    else:
        arg1 = ladder_conv_nhwc_hwnc(N, F, H, W, C, K, K, S, D, P)
        # arg1 = ladder_conv_nhwc_hwnc(128, 64, 56, 56, 64, 3, 3, 1, 1, 1)
        # arg1 = ladder_conv_nhwc_hwnc(128, 128, 56, 56, 256, 1, 1, 1, 1, 0)
        # arg1 = conv_nhwc_hwnc(128, 64, 224, 224, 128, 7, 7, 2, 1, 3)

        args = arg1

        input_args = args[:-1]
        output_args = [args[-1]]
        node = IRNode([None for _ in input_args], args, "cutlass_conv2d_reshape_bias")
        node.add_tag("tensorCoreConfig", [2, 3])
        node.add_tag("ladder_config", (True, True, 1))
        
        # node.add_tag("tensorCoreConfig", [0, 1])
        # node.add_tag("ladder_config", (True, True, 1))
        output_nodes = [OutputNode(node)]
        policy = LadderPolicy(output_nodes, arch)
        configs = policy.emit_config(20)

    compile_results = []
    cgen = welder.CodeGenerator()
    for config in configs:
        try:
            cpresult = cgen.compile(output_nodes, config, "hip", 
                                    kernel_name="Fused")
        except:
            continue
        if cpresult is not None:
            compile_results.append(cpresult)
    welder.utils.compile_parallel(compile_results, arch, timeout=30)
    best = select_best(output_nodes, compile_results)
    best_latency = best.latency
    values = []
    print('code: ', best.code)
    print("-" * 80, flush=True)
    print("best latency: {}".format(best_latency))
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
