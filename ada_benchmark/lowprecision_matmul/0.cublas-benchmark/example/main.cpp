/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#include "sample_cublasLt_LtFp8Matmul.h"
#include "helpers.h"

// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> inference_server_set = {
    // gemm
    // std::make_tuple(16384, 16384, 16384, false, true),
    // std::make_tuple(1024, 1024, 1024, false, true),
    // std::make_tuple(1024, 8192, 1024, false, true),
    // std::make_tuple(4096, 4096, 4096, false, true),
    std::make_tuple(1, 14336, 57344, false, true),
    std::make_tuple(32, 14336, 57344, false, true),
    std::make_tuple(4096, 14336, 57344, false, true),
    std::make_tuple(1, 8192, 28672, false, true),
    std::make_tuple(32, 8192, 28672, false, true),
    std::make_tuple(4096, 8192, 28672, false, true),
};

int main() {
    for (const auto &problem : inference_server_set)
    {
            int m, n, k;
            bool a_t, b_t;
            std::tie(m, n, k, a_t, b_t) = problem;
        TestBench<__nv_fp8_e4m3, __nv_fp8_e4m3, float> props(m, n, k, 2.0f, 0.0f /* ignored */, 32ULL * 1024 * 1024);

        props.run([&props] {
            LtFp8Matmul(props.ltHandle,
                        props.m,
                        props.n,
                        props.k,
                        &props.alpha,
                        props.AscaleDev,
                        props.Adev,
                        props.k,
                        props.BscaleDev,
                        props.Bdev,
                        props.k,
                        props.CscaleDev,
                        props.Cdev,
                        props.m,
                        props.DscaleDev,
                        props.DamaxDev,
                        props.workspace,
                        props.workspaceSize);
        });
    }
    return 0;
}