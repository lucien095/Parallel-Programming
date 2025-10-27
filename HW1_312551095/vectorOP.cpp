#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  
  __pp_vec_float x; // float x;
  __pp_vec_int y; // int y;
  __pp_vec_int count_y; // int count;

  __pp_vec_float result;
  __pp_vec_float result_threshold = _pp_vset_float(9.999999f);

  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_mask mask_all, mask_y, mask_cnt_y_true, mask_should_clamped;

  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  if(N % VECTOR_WIDTH)
  {
    for(int i = N; i < N+VECTOR_WIDTH; ++i)
    {
      values[i] = 0.f;
      exponents[i] = 1;
    }
  }

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // All ones
    mask_all = _pp_init_ones();

    // Init exp mask val to zeros
    mask_y = _pp_init_ones(0);
    count_y = _pp_vset_int(0);

    _pp_vload_float(x, values + i, mask_all); //x = values[i];
    _pp_vload_int(y, exponents + i, mask_all);  //y = exponents[i];
    _pp_veq_int(mask_y, y, zero, mask_all); //if(y==0){
    _pp_vset_float(result, 1.f, mask_y); //output[i] = 1.f;
    mask_y = _pp_mask_not(mask_y); // } else{
    _pp_vmove_float(result, x, mask_y);  //result = x;
    _pp_vsub_int(count_y, y, one, mask_y);  //count = y - 1;

    _pp_vgt_int(mask_cnt_y_true, count_y, zero, mask_all);
    while(_pp_cntbits(mask_cnt_y_true))  //while(count > 0){
    {
      _pp_vmult_float(result, result, x, mask_cnt_y_true); //result *= x;
      _pp_vsub_int(count_y, count_y, one, mask_cnt_y_true);  //count--;
      _pp_vgt_int(mask_cnt_y_true, count_y, zero, mask_all); //}
    }

    mask_should_clamped = _pp_init_ones(0);
    _pp_vgt_float(mask_should_clamped, result, result_threshold, mask_all); //if (result > 9.999999f){
    _pp_vset_float(result, 9.999999f, mask_should_clamped);  //result = 9.999999f;}
    _pp_vstore_float(output + i, result, mask_all); //output[i] = result;
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float sum[VECTOR_WIDTH];
  __pp_vec_float sum_vector = _pp_vset_float(0.f);
  __pp_mask mask_all;

  mask_all = _pp_init_ones();
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    __pp_vec_float tmp;
    _pp_vload_float(tmp, values + i, mask_all);
    _pp_vadd_float(sum_vector, sum_vector, tmp, mask_all);
  }
  // run log(VECTOR_WIDTH) times
  for(int j = 1; j < VECTOR_WIDTH; j <<= 1)
  {
    _pp_hadd_float(sum_vector, sum_vector);
    _pp_interleave_float(sum_vector, sum_vector);
  }

  _pp_vstore_float(sum, sum_vector, mask_all);

  return sum[0];
}