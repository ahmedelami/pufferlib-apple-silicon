#include "matsci.h"

#define OBS_SIZE 3
#define NUM_ATNS 3
#define ACT_SIZES {1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env Matsci
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    DictItem* item = dict_get_unsafe(kwargs, "num_agents");
    if (item == NULL) item = dict_get_unsafe(kwargs, "num_atoms");
    env->num_agents = item == NULL ? 2 : (int)item->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
}
