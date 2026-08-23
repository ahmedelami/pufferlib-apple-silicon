#include "convert_circle.h"

#define OBS_SIZE 28
#define NUM_ATNS 2
#define ACT_SIZES {9, 5}
#define OBS_TENSOR_T FloatTensor

#define MY_VEC_VALIDATE convert_circle_validate
#define Env ConvertCircle
#include "vecenv.h"

static double kwarg_or(Dict *kwargs, const char *key, double fallback) {
  DictItem *item = dict_get_unsafe(kwargs, key);
  return item == NULL ? fallback : item->value;
}

int convert_circle_validate(Dict *vec_kwargs, Dict *kwargs) {
  int total_agents = (int)kwarg_or(vec_kwargs, "total_agents", 0);
  int num_buffers = (int)kwarg_or(vec_kwargs, "num_buffers", 0);
  int agents_per_env = (int)kwarg_or(kwargs, "num_agents", 1024);
  int num_resources = (int)kwarg_or(kwargs, "num_resources", 8);
  if (num_resources != 8) {
    fprintf(stderr, "convert_circle: static binding requires num_resources=8\n");
    return 0;
  }
  if (agents_per_env < 1 || total_agents < 1 || num_buffers < 1
          || total_agents % agents_per_env != 0
          || total_agents % num_buffers != 0
          || (total_agents / num_buffers) % agents_per_env != 0) {
    fprintf(stderr,
            "convert_circle: total_agents/buffers must divide into whole environments\n");
    return 0;
  }
  return 1;
}

void my_init(Env *env, Dict *kwargs) {
  int num_resources = (int)kwarg_or(kwargs, "num_resources", 8);
  env->width = kwarg_or(kwargs, "width", 1920);
  env->height = kwarg_or(kwargs, "height", 1080);
  env->num_agents = kwarg_or(kwargs, "num_agents", 1024);
  env->num_factories = kwarg_or(kwargs, "num_factories", 32);
  env->num_resources = num_resources;
  env->equidistant = kwarg_or(kwargs, "equidistant", 0);
  env->radius = kwarg_or(kwargs, "radius", 30);
  init(env);
}

void my_log(Log *log, Dict *out) {
  dict_set(out, "perf", log->perf);
  dict_set(out, "score", log->score);
  dict_set(out, "episode_return", log->episode_return);
  dict_set(out, "episode_length", log->episode_length);
}
