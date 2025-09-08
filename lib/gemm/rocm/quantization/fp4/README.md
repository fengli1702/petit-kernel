[ RUN      ] NvFp4ToPetitFp4Test.TestLayout128x16Fp16
tid=64, wid=1, wtid=0, scale_val=0xb692
tid=65, wid=1, wtid=1, scale_val=0xdc8e
tid=66, wid=1, wtid=2, scale_val=0xf2dd
tid=67, wid=1, wtid=3, scale_val=0xa9ea
tid=68, wid=1, wtid=4, scale_val=0xbac6
tid=69, wid=1, wtid=5, scale_val=0xf2cf
tid=70, wid=1, wtid=6, scale_val=0xd999
tid=71, wid=1, wtid=7, scale_val=0x9e99
tid=0, wid=0, wtid=0, scale_val=0xc7e7
tid=1, wid=0, wtid=1, scale_val=0x8fc6
tid=2, wid=0, wtid=2, scale_val=0x86c2
tid=3, wid=0, wtid=3, scale_val=0xc8b7
tid=4, wid=0, wtid=4, scale_val=0xcbf2
tid=5, wid=0, wtid=5, scale_val=0x8cba
tid=6, wid=0, wtid=6, scale_val=0x68de
tid=7, wid=0, wtid=7, scale_val=0xf6aa
tid=8, wid=0, wtid=8, scale_val=0xd5eb
tid=9, wid=0, wtid=9, scale_val=0xa8ce
tid=10, wid=0, wtid=10, scale_val=0xcb95
tid=11, wid=0, wtid=11, scale_val=0xab9c
tid=12, wid=0, wtid=12, scale_val=0x92a7
tid=13, wid=0, wtid=13, scale_val=0x8ff5
tid=14, wid=0, wtid=14, scale_val=0xb5a3
tid=15, wid=0, wtid=15, scale_val=0xb697
tid=16, wid=0, wtid=16, scale_val=0xb4d5
tid=17, wid=0, wtid=17, scale_val=0x7ab4
tid=18, wid=0, wtid=18, scale_val=0xedd1
tid=19, wid=0, wtid=19, scale_val=0xf3c3
tid=20, wid=0, wtid=20, scale_val=0xa7f1
tid=21, wid=0, wtid=21, scale_val=0xd0e9
tid=22, wid=0, wtid=22, scale_val=0xe6b2
tid=23, wid=0, wtid=23, scale_val=0xaa81
tid=24, wid=0, wtid=24, scale_val=0xd4a4
tid=25, wid=0, wtid=25, scale_val=0x85ca
tid=26, wid=0, wtid=26, scale_val=0xcbe4
tid=27, wid=0, wtid=27, scale_val=0x7cd9
tid=28, wid=0, wtid=28, scale_val=0xb4a5
tid=29, wid=0, wtid=29, scale_val=0xc193
tid=30, wid=0, wtid=30, scale_val=0xc4cd
tid=31, wid=0, wtid=31, scale_val=0x74eb
tid=32, wid=0, wtid=32, scale_val=0xb692
tid=33, wid=0, wtid=33, scale_val=0xdc8e
tid=34, wid=0, wtid=34, scale_val=0xf2dd
tid=35, wid=0, wtid=35, scale_val=0xa9ea
tid=36, wid=0, wtid=36, scale_val=0xbac6
tid=37, wid=0, wtid=37, scale_val=0xf2cf
tid=38, wid=0, wtid=38, scale_val=0xd999
tid=39, wid=0, wtid=39, scale_val=0x9e99
tid=40, wid=0, wtid=40, scale_val=0xe4ac
tid=41, wid=0, wtid=41, scale_val=0xe3d7
tid=42, wid=0, wtid=42, scale_val=0xb081
tid=43, wid=0, wtid=43, scale_val=0xd4a8
tid=44, wid=0, wtid=44, scale_val=0xbadc
tid=45, wid=0, wtid=45, scale_val=0xa19f
tid=46, wid=0, wtid=46, scale_val=0xc188
tid=47, wid=0, wtid=47, scale_val=0x85a7
tid=48, wid=0, wtid=48, scale_val=0xb2be
tid=49, wid=0, wtid=49, scale_val=0xa18d
tid=50, wid=0, wtid=50, scale_val=0xe870
tid=51, wid=0, wtid=51, scale_val=0xb48c
tid=52, wid=0, wtid=52, scale_val=0xbbd9
tid=53, wid=0, wtid=53, scale_val=0xec98
tid=54, wid=0, wtid=54, scale_val=0x86c3
tid=55, wid=0, wtid=55, scale_val=0xf6b7
tid=56, wid=0, wtid=56, scale_val=0x8de0
tid=57, wid=0, wtid=57, scale_val=0xb39b
tid=58, wid=0, wtid=58, scale_val=0xbae7
tid=59, wid=0, wtid=59, scale_val=0x928e
tid=60, wid=0, wtid=60, scale_val=0xe8d4
tid=61, wid=0, wtid=61, scale_val=0xd082
tid=62, wid=0, wtid=62, scale_val=0x8a8f
tid=63, wid=0, wtid=63, scale_val=0xf1bd
/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.046875
  h_petit_output[i]
    Which is: 0.3125
Output and reference differ at index 64

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.046875
  h_petit_output[i]
    Which is: 0.3125
Output and reference differ at index 65

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.046875
  h_petit_output[i]
    Which is: 0.3125
Output and reference differ at index 66

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.0351562
  h_petit_output[i]
    Which is: -0.234375
Output and reference differ at index 67

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.0703125
  h_petit_output[i]
    Which is: -0.46875
Output and reference differ at index 69

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0703125
  h_petit_output[i]
    Which is: 0.46875
Output and reference differ at index 70

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0117188
  h_petit_output[i]
    Which is: 0.078125
Output and reference differ at index 71

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.00585938
  h_petit_output[i]
    Which is: -0.0390625
Output and reference differ at index 72

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0703125
  h_petit_output[i]
    Which is: 0.46875
Output and reference differ at index 73

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.00585938
  h_petit_output[i]
    Which is: 0.0390625
Output and reference differ at index 75

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0117188
  h_petit_output[i]
    Which is: 0.078125
Output and reference differ at index 76

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0117188
  h_petit_output[i]
    Which is: 0.078125
Output and reference differ at index 77

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0117188
  h_petit_output[i]
    Which is: 0.078125
Output and reference differ at index 78

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0703125
  h_petit_output[i]
    Which is: 0.46875
Output and reference differ at index 79

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.117188
  h_petit_output[i]
    Which is: 3.5
Output and reference differ at index 80

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.351562
  h_petit_output[i]
    Which is: -10.5
Output and reference differ at index 81

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0585938
  h_petit_output[i]
    Which is: 1.75
Output and reference differ at index 82

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.234375
  h_petit_output[i]
    Which is: 7
Output and reference differ at index 83

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.703125
  h_petit_output[i]
    Which is: -21
Output and reference differ at index 84

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.46875
  h_petit_output[i]
    Which is: -14
Output and reference differ at index 85

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.703125
  h_petit_output[i]
    Which is: 21
Output and reference differ at index 86

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.117188
  h_petit_output[i]
    Which is: 3.5
Output and reference differ at index 87

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.46875
  h_petit_output[i]
    Which is: 14
Output and reference differ at index 88

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.351562
  h_petit_output[i]
    Which is: -10.5
Output and reference differ at index 89

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.234375
  h_petit_output[i]
    Which is: 7
Output and reference differ at index 90

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.703125
  h_petit_output[i]
    Which is: -21
Output and reference differ at index 91

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.703125
  h_petit_output[i]
    Which is: 21
Output and reference differ at index 92

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.0585938
  h_petit_output[i]
    Which is: -1.75
Output and reference differ at index 93

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: -0.0585938
  h_petit_output[i]
    Which is: -1.75
Output and reference differ at index 94

/home/o_feng/work/petit-kernel/lib/gemm/rocm/quantization/fp4/quantization_utils_fp4_test.cc:89: Failure
Expected equality of these values:
  h_reference[i]
    Which is: 0.0585938
  h_petit_output[i]
    Which is: 1.75
Output and reference differ at index 95