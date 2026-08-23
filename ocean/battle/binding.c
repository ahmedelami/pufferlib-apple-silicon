#include "battle.h"

#define OBS_SIZE 100
#define NUM_ATNS 3
#define ACT_SIZES {1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define MY_VEC_VALIDATE battle_validate
#define Env Battle
#include "vecenv.h"

static double kwarg_or(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : item->value;
}

int battle_validate(Dict* vec_kwargs, Dict* kwargs) {
    int total_agents = (int)kwarg_or(vec_kwargs, "total_agents", 0);
    int num_buffers = (int)kwarg_or(vec_kwargs, "num_buffers", 0);
    int num_agents = (int)kwarg_or(kwargs, "num_agents", 512);
    int num_armies = (int)kwarg_or(kwargs, "num_armies", 2);
    if (num_armies != 2) {
        fprintf(stderr, "battle: static binding requires num_armies=2\n");
        return 0;
    }
    if (num_agents < AGENT_OBS) {
        fprintf(stderr,
            "battle: num_agents is the policy-army size and must be at least %d\n",
            AGENT_OBS);
        return 0;
    }
    if (total_agents < 1 || num_buffers < 1
            || total_agents % num_agents != 0
            || total_agents % num_buffers != 0
            || (total_agents / num_buffers) % num_agents != 0) {
        fprintf(stderr,
            "battle: total_agents/buffers must divide into whole policy armies\n");
        return 0;
    }
    return 1;
}

void my_init(Env* env, Dict* kwargs) {
    int num_armies = (int)kwarg_or(kwargs, "num_armies", 2);
    env->width = kwarg_or(kwargs, "width", 1920);
    env->height = kwarg_or(kwargs, "height", 1080);
    env->size_x = kwarg_or(kwargs, "size_x", 1.0);
    env->size_y = kwarg_or(kwargs, "size_y", 1.0);
    env->size_z = kwarg_or(kwargs, "size_z", 1.0);
    // The static ABI exposes the trainable army only. An equally-sized second
    // army remains scripted, matching Battle's original training setup.
    env->num_agents = kwarg_or(kwargs, "num_agents", 512);
    env->num_armies = num_armies;
    env->num_sim_agents = env->num_agents * env->num_armies;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "oob_rate", log->oob_rate);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
