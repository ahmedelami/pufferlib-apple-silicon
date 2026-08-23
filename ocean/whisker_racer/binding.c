#include "whisker_racer.h"
#define OBS_SIZE 3
#define NUM_ATNS 1
#define ACT_SIZES {3}
#define OBS_TENSOR_T FloatTensor

#define Env WhiskerRacer
#include "vecenv.h"

static double kwarg_or(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : item->value;
}

void my_init(Env* env, Dict* kwargs) {
    int env_index = (int)env->rng;
    env->num_agents = 1;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->w_ang = dict_get(kwargs, "w_ang")->value;
    // The current two-whisker implementation derives its active angles from
    // w_ang in init(). Keep the legacy four-angle keys optional so the shipped
    // config remains constructible and older explicit configs still load.
    env->llw_ang = kwarg_or(kwargs, "llw_ang", -2.0 * env->w_ang);
    env->flw_ang = kwarg_or(kwargs, "flw_ang", -env->w_ang);
    env->frw_ang = kwarg_or(kwargs, "frw_ang", env->w_ang);
    env->rrw_ang = kwarg_or(kwargs, "rrw_ang", 2.0 * env->w_ang);
    env->max_whisker_length = dict_get(kwargs, "max_whisker_length")->value;
    env->turn_pi_frac = dict_get(kwargs, "turn_pi_frac")->value;
    env->maxv = dict_get(kwargs, "maxv")->value;
    env->render = kwarg_or(kwargs, "render", 0);
    env->continuous = dict_get(kwargs, "continuous")->value;
    env->reward_yellow = dict_get(kwargs, "reward_yellow")->value;
    env->reward_green = dict_get(kwargs, "reward_green")->value;
    env->gamma = dict_get(kwargs, "gamma")->value;
    env->track_width = dict_get(kwargs, "track_width")->value;
    env->num_radial_sectors = dict_get(kwargs, "num_radial_sectors")->value;
    env->num_points = dict_get(kwargs, "num_points")->value;
    env->bezier_resolution = dict_get(kwargs, "bezier_resolution")->value;
    env->corner_thresh = dict_get(kwargs, "corner_thresh")->value;
    env->ftmp1 = dict_get(kwargs, "ftmp1")->value;
    env->ftmp2 = dict_get(kwargs, "ftmp2")->value;
    env->ftmp3 = dict_get(kwargs, "ftmp3")->value;
    env->ftmp4 = dict_get(kwargs, "ftmp4")->value;
    env->mode7 = dict_get(kwargs, "mode7")->value;
    env->render_many = dict_get(kwargs, "render_many")->value;
    env->rng = dict_get(kwargs, "rng")->value;
    env->method = dict_get(kwargs, "method")->value;
    env->i = kwarg_or(kwargs, "i", env_index);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
