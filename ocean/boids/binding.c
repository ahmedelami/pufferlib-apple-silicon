#include "boids.h"

#define OBS_SIZE 256
#define NUM_ATNS 2
#define ACT_SIZES {5, 5}
#define OBS_TENSOR_T FloatTensor

#define MY_VEC_VALIDATE boids_validate
#define Env Boids
#include "vecenv.h"

static double kwarg_or(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : item->value;
}

int boids_validate(Dict* vec_kwargs, Dict* kwargs) {
    int total_agents = (int)kwarg_or(vec_kwargs, "total_agents", 0);
    int num_buffers = (int)kwarg_or(vec_kwargs, "num_buffers", 0);
    int num_boids = (int)kwarg_or(kwargs, "num_boids", 64);
    if (num_boids != 64) {
        fprintf(stderr, "boids: static binding requires num_boids=64\n");
        return 0;
    }
    if (total_agents < 1 || num_buffers < 1
            || total_agents % num_boids != 0
            || total_agents % num_buffers != 0
            || (total_agents / num_buffers) % num_boids != 0) {
        fprintf(stderr,
            "boids: total_agents/buffers must divide into whole 64-boid environments\n");
        return 0;
    }
    return 1;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_boids = kwarg_or(kwargs, "num_boids", 64);
    env->num_agents = env->num_boids;
    env->report_interval = kwarg_or(kwargs, "report_interval", 1);
    env->margin_turn_factor = kwarg_or(kwargs, "margin_turn_factor", 1.0);
    env->centering_factor = kwarg_or(kwargs, "centering_factor", 0.0);
    env->avoid_factor = kwarg_or(kwargs, "avoid_factor", 0.0);
    env->matching_factor = kwarg_or(kwargs, "matching_factor", 0.0);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
