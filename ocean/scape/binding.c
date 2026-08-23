#include "scape.h"
#define OBS_SIZE 28
#define NUM_ATNS 2
#define ACT_SIZES {9, 5}
#define OBS_TENSOR_T FloatTensor

#define MY_VEC_VALIDATE scape_validate
#define Env Scape
#include "vecenv.h"

int scape_validate(Dict* vec_kwargs, Dict* kwargs) {
    (void)vec_kwargs;
    (void)kwargs;
    // Scape is an interactive rendering prototype: its observation function
    // is empty and c_step ignores policy actions. Do not advertise constant,
    // action-independent trajectories as a trainable vector environment.
    fprintf(stderr,
        "scape: RL vector API is incomplete upstream; training is unsupported\n");
    return 0;
}

void my_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->num_agents = 8;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
