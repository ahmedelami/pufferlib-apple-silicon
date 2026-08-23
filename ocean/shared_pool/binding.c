#include "shared_pool.h"

#define OBS_SIZE 49
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T ByteTensor

#define MY_VEC_VALIDATE shared_pool_validate
#define Env CCpr
#include "vecenv.h"

static double kwarg_or(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : item->value;
}

int shared_pool_validate(Dict* vec_kwargs, Dict* kwargs) {
    int total_agents = (int)kwarg_or(vec_kwargs, "total_agents", 0);
    int num_buffers = (int)kwarg_or(vec_kwargs, "num_buffers", 0);
    int agents_per_env = (int)kwarg_or(kwargs, "num_agents", 8);
    int vision = (int)kwarg_or(kwargs, "vision", 3);
    if (vision != 3) {
        fprintf(stderr, "shared_pool: static binding requires vision=3\n");
        return 0;
    }
    if (agents_per_env < 1 || total_agents < 1 || num_buffers < 1
            || total_agents % agents_per_env != 0
            || total_agents % num_buffers != 0
            || (total_agents / num_buffers) % agents_per_env != 0) {
        fprintf(stderr,
            "shared_pool: total_agents/buffers must divide into whole environments\n");
        return 0;
    }
    return 1;
}

void my_init(Env* env, Dict* kwargs) {
    int vision = (int)kwarg_or(kwargs, "vision", 3);
    env->width = kwarg_or(kwargs, "width", 32);
    env->height = kwarg_or(kwargs, "height", 32);
    env->num_agents = kwarg_or(kwargs, "num_agents", 8);
    env->vision = vision;
    env->reward_food = kwarg_or(kwargs, "reward_food", 1.0);
    env->interactive_food_reward = kwarg_or(kwargs, "interactive_food_reward", 5.0);
    env->reward_move = kwarg_or(kwargs, "reward_move", -0.01);
    env->food_base_spawn_rate = kwarg_or(kwargs, "food_base_spawn_rate", 2e-3);
    init_ccpr(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "moves", log->moves);
    dict_set(out, "food_nb", log->food_nb);
    dict_set(out, "alive_steps", log->alive_steps);
}
