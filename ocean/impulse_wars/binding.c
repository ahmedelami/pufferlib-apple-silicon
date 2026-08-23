#include <stdio.h>

#include "env.h"

#define OBS_SIZE 1000
#define NUM_ATNS 5
#define ACT_SIZES {9, 17, 2, 2, 2}
#define OBS_TENSOR_T ByteTensor

// The 4.0 Python wrapper translated the five discrete policy heads into the
// seven continuous controls consumed by stepEnv. Keep that contract in C now
// that vecenv calls the environment directly.
#undef c_step
void puffer_iw_step(iwEnv* env);
#define c_step puffer_iw_step

#undef c_close
void puffer_iw_close(iwEnv* env);
#define c_close puffer_iw_close

#define MY_VEC_VALIDATE puffer_iw_validate
#define Env iwEnv
#include "vecenv.h"

static double kwarg_or(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : item->value;
}

int puffer_iw_validate(Dict* vec_kwargs, Dict* kwargs) {
    int total_agents = (int)kwarg_or(vec_kwargs, "total_agents", 0);
    int num_buffers = (int)kwarg_or(vec_kwargs, "num_buffers", 0);
    int num_drones = (int)kwarg_or(kwargs, "num_drones", 2);
    int num_agents = (int)kwarg_or(kwargs, "num_agents", 1);
    int continuous = (int)kwarg_or(kwargs, "continuous", 0);
    int enable_teams = (int)kwarg_or(kwargs, "enable_teams", 0);
    int map_idx = (int)kwarg_or(kwargs, "map_idx", -1);

    if (num_drones != 2) {
        fprintf(stderr, "impulse_wars: static binding requires num_drones=2\n");
        return 0;
    }
    if (num_agents < 1 || num_agents > num_drones) {
        fprintf(stderr, "impulse_wars: num_agents must be in [1, num_drones]\n");
        return 0;
    }
    if (continuous) {
        fprintf(stderr, "impulse_wars: static binding exposes the discrete action API\n");
        return 0;
    }
    if (enable_teams) {
        fprintf(stderr, "impulse_wars: teams require more than two drones\n");
        return 0;
    }
    if (map_idx < -1 || map_idx >= NUM_MAPS) {
        fprintf(stderr, "impulse_wars: map_idx must be -1 or in [0, %d]\n", NUM_MAPS - 1);
        return 0;
    }
    if (total_agents < 1 || num_buffers < 1
            || total_agents % num_agents != 0
            || total_agents % num_buffers != 0
            || (total_agents / num_buffers) % num_agents != 0) {
        fprintf(stderr,
            "impulse_wars: total_agents/buffers must divide into whole environments\n");
        return 0;
    }
    return 1;
}

void puffer_iw_step(iwEnv* env) {
    float* policy_actions = env->actions;
    float continuous_actions[_MAX_DRONES * 7] = {0};

    for (uint8_t i = 0; i < env->numAgents; i++) {
        const int src = i * NUM_ATNS;
        const int dst = i * 7;
        int move = (int)policy_actions[src];
        int aim = (int)policy_actions[src + 1];

        if (move >= 0 && move < 8) {
            continuous_actions[dst] = discMoveToContMoveMap[0][move];
            continuous_actions[dst + 1] = discMoveToContMoveMap[1][move];
        }
        if (aim >= 0 && aim < 16) {
            continuous_actions[dst + 2] = discAimToContAimMap[0][aim];
            continuous_actions[dst + 3] = discAimToContAimMap[1][aim];
        }
        continuous_actions[dst + 4] = policy_actions[src + 2];
        continuous_actions[dst + 5] = policy_actions[src + 3];
        continuous_actions[dst + 6] = policy_actions[src + 4];
    }

    env->actions = continuous_actions;
    stepEnv(env);
    env->actions = policy_actions;
}

void puffer_iw_close(iwEnv* env) {
    destroyEnv(env);
    fastFree(env->truncations);
    env->truncations = NULL;
}

void my_init(Env* env, Dict* kwargs) {
    int num_drones = (int)kwarg_or(kwargs, "num_drones", 2);
    int num_agents = (int)kwarg_or(kwargs, "num_agents", 1);
    int enable_teams = (int)kwarg_or(kwargs, "enable_teams", 0);
    int map_idx = (int)kwarg_or(kwargs, "map_idx", -1);

    uint64_t seed = (uint64_t)kwarg_or(kwargs, "seed", env->rng);
    initEnv(
        env,
        (uint8_t)num_drones,
        (uint8_t)num_agents,
        (int8_t)map_idx,
        seed,
        (bool)enable_teams,
        (bool)kwarg_or(kwargs, "sitting_duck", 0),
        (bool)kwarg_or(kwargs, "is_training", 1),
        false
    );
    setRewards(
        env,
        (float)kwarg_or(kwargs, "reward_win", WIN_REWARD),
        (float)kwarg_or(kwargs, "reward_self_kill", SELF_KILL_PUNISHMENT),
        (float)kwarg_or(kwargs, "reward_enemy_death", ENEMY_DEATH_REWARD),
        (float)kwarg_or(kwargs, "reward_enemy_kill", ENEMY_KILL_REWARD),
        0.0f,
        0.0f,
        (float)kwarg_or(kwargs, "reward_death", -0.25f),
        (float)kwarg_or(kwargs, "reward_energy_emptied", ENERGY_EMPTY_PUNISHMENT),
        (float)kwarg_or(kwargs, "reward_weapon_pickup", WEAPON_PICKUP_REWARD),
        (float)kwarg_or(kwargs, "reward_shield_break", SHIELD_BREAK_REWARD),
        (float)kwarg_or(kwargs, "reward_shot_hit_coef", SHOT_HIT_REWARD_COEF),
        (float)kwarg_or(kwargs, "reward_explosion_hit_coef", EXPLOSION_HIT_REWARD_COEF)
    );

    static bool maps_initialized = false;
    if (!maps_initialized) {
        initMaps(env);
        maps_initialized = true;
    }
}

static const char* IW_STAT_NAMES[] = {
    "returns",
    "distance_traveled",
    "abs_distance_traveled",
    "brake_time",
    "total_bursts",
    "bursts_hit",
    "energy_emptied",
    "shields_broken",
    "own_shield_broken",
    "self_kills",
    "kills",
    "unknown_kills",
    "wins",
    "total_shots_fired",
    "total_shots_hit",
    "total_shots_taken",
    "total_own_shots_taken",
    "total_picked_up",
    "total_shot_distances",
};

#define IW_NUM_STATS ((int)(sizeof(IW_STAT_NAMES) / sizeof(IW_STAT_NAMES[0])))

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_length", log->length);
    dict_set(out, "ties", log->ties);
    dict_set(out, "perf", log->stats[0].wins);
    dict_set(out, "score", log->stats[0].wins);

    static char keys[_MAX_DRONES][IW_NUM_STATS][64];
    if (keys[0][0][0] == '\0') {
        for (int i = 0; i < _MAX_DRONES; i++) {
            for (int j = 0; j < IW_NUM_STATS; j++) {
                snprintf(keys[i][j], sizeof(keys[i][j]),
                    "drone_%d_%s", i, IW_STAT_NAMES[j]);
            }
        }
    }

    for (int i = 0; i < _MAX_DRONES; i++) {
        droneStats* stats = &log->stats[i];
        float values[IW_NUM_STATS] = {
            stats->returns,
            stats->distanceTraveled,
            stats->absDistanceTraveled,
            stats->brakeTime,
            stats->totalBursts,
            stats->burstsHit,
            stats->energyEmptied,
            stats->shieldsBroken,
            stats->ownShieldBroken,
            stats->selfKills,
            stats->kills,
            stats->unknownKills,
            stats->wins,
            stats->totalShotsFired,
            stats->totalShotsHit,
            stats->totalShotsTaken,
            stats->totalOwnShotsTaken,
            stats->totalWeaponsPickedUp,
            stats->totalShotDistances,
        };
        for (int j = 0; j < IW_NUM_STATS; j++) {
            dict_set(out, keys[i][j], values[j]);
        }
    }
}
