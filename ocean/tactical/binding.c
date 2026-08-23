#include "tactical.h"

#define OBS_SIZE 10
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TENSOR_T ByteTensor

#define MY_VEC_VALIDATE tactical_validate
#define Env Tactical
#include "vecenv.h"

int tactical_validate(Dict* vec_kwargs, Dict* kwargs) {
    (void)vec_kwargs;
    (void)kwargs;
    // Tactical has always been a rendering/gameplay prototype: its upstream
    // compute_observations is empty and c_step does not consume actions. Fail
    // closed instead of advertising zero-state trajectories as trainable.
    fprintf(stderr,
        "tactical: RL vector API is incomplete upstream; training is unsupported\n");
    return 0;
}

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    Tactical* initialized = init_tactical();
    unsigned char* observations = initialized->observations;
    float* actions = initialized->actions;
    float* rewards = initialized->rewards;
    *env = *initialized;
    free(observations);
    free(actions);
    free(rewards);
    free(initialized);
    env->observations = NULL;
    env->actions = NULL;
    env->rewards = NULL;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
}
