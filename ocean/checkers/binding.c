#include "checkers.h"

#define OBS_SIZE 64
#define NUM_ATNS 1
#define ACT_SIZES {512}
#define OBS_TENSOR_T ByteTensor

#define MY_VEC_VALIDATE checkers_validate
#define Env Checkers
#include "vecenv.h"

int checkers_validate(Dict *vec_kwargs, Dict *kwargs) {
  DictItem *size = dict_get_unsafe(kwargs, "size");
  if (size == NULL || (int)size->value != 8) {
    fprintf(stderr, "checkers: static binding requires size=8\n");
    return 0;
  }

  DictItem *total_agents_item = dict_get_unsafe(vec_kwargs, "total_agents");
  DictItem *num_buffers_item = dict_get_unsafe(vec_kwargs, "num_buffers");
  int total_agents = total_agents_item == NULL ? 0 : (int)total_agents_item->value;
  int num_buffers = num_buffers_item == NULL ? 0 : (int)num_buffers_item->value;
  if (total_agents < 1 || num_buffers < 1 || total_agents % num_buffers != 0) {
    fprintf(stderr,
            "checkers: total_agents must divide evenly across positive buffers\n");
    return 0;
  }
  return 1;
}

void my_init(Env *env, Dict *kwargs) {
  env->num_agents = 1;
  env->size = dict_get(kwargs, "size")->value;
}

void my_log(Log *log, Dict *out) {
  dict_set(out, "perf", log->perf);
  dict_set(out, "score", log->score);
  dict_set(out, "episode_return", log->episode_return);
  dict_set(out, "episode_length", log->episode_length);
  dict_set(out, "winrate", log->winrate);
}
