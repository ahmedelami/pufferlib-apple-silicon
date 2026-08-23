#include "memory.h"

#define OBS_SIZE 1
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

#define Env Memory
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    DictItem* length = dict_get_unsafe(kwargs, "length");
    env->num_agents = 1;
    env->length = length == NULL ? 4 : (int)length->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
}
