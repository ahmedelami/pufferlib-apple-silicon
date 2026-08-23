#include "template.h"

#define OBS_SIZE 1
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T ByteTensor

#define Env Template
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    DictItem* size = dict_get_unsafe(kwargs, "size");
    env->num_agents = 1;
    env->size = size == NULL ? 5 : (int)size->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
}
