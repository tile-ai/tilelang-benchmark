mkdir -p pdf
mkdir -p png

python plot_operator_figures_fp16_gemm.py
python plot_operator_figures_fp16_gemv.py
python plot_operator_figures_int8_gemm.py
python plot_operator_figures_int8_gemv.py
