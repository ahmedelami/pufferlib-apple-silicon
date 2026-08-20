// Double pendulum swing-up and balance task with discrete cart forces.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define DP_OBS_SIZE 8
#define DP_ACTIONS 3
#define DP_MAX_STEPS 600
#define DP_X_THRESHOLD 5.0f
#define DP_WIDTH 800
#define DP_HEIGHT 420
#define DP_SCALE 65.0f

// DeepMind Control Suite-style dense reward constants.
#define DP_CENTER_MARGIN  2.0f
#define DP_ANG_VEL_MARGIN 5.0f
#define DP_VELOCITY_FLOOR 0.60f
#define DP_LN_10          2.302585093f
#define DP_BALANCE_ANGLE_SCALE   0.30f
#define DP_BALANCE_ANG_VEL_SCALE 1.0f
#define DP_BALANCE_CART_SCALE    2.0f

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float x_threshold_termination;
    float max_steps_termination;
    float hold_time;
    float n;
} Log;

typedef struct DoublePendulum {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;

    float x;
    float x_dot;
    float theta1;
    float theta1_dot;
    float theta2;
    float theta2_dot;
    int tick;
    float episode_return;
    int upright_steps;
    int max_upright_steps;

    float cart_mass;
    float link1_mass;
    float link2_mass;
    float link1_length;
    float link2_length;
    float gravity;
    float force_mag;
    float dt;
    int substeps;         // physics substeps per control step (RK4)
    float balance_bonus_weight;  // blend between deepmind term and balance_quality
} DoublePendulum;

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_YELLOW = (Color){245, 197, 66, 255};

static inline float dp_randf(DoublePendulum* env, float lo, float hi) {
    float t = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return lo + t * (hi - lo);
}

static inline float wrap_pi(float x) {
    while (x > M_PI) x -= 2.0f * M_PI;
    while (x < -M_PI) x += 2.0f * M_PI;
    return x;
}

void compute_observations(DoublePendulum* env) {
    env->observations[0] = env->x / DP_X_THRESHOLD;
    env->observations[1] = env->x_dot / 5.0f;
    env->observations[2] = sinf(env->theta1);
    env->observations[3] = cosf(env->theta1);
    env->observations[4] = env->theta1_dot / 8.0f;
    env->observations[5] = sinf(env->theta2);
    env->observations[6] = cosf(env->theta2);
    env->observations[7] = env->theta2_dot / 8.0f;
}

void add_log(DoublePendulum* env, bool x_done, bool timeout) {
    float normalized = env->episode_return / (float)DP_MAX_STEPS;
    env->log.perf += fminf(fmaxf(normalized, 0.0f), 1.0f);
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.x_threshold_termination += x_done ? 1.0f : 0.0f;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.hold_time += (float)env->max_upright_steps;
    env->log.n += 1.0f;
}

void init(DoublePendulum* env) {
    env->num_agents = 1;
}

void c_reset(DoublePendulum* env) {
    env->x = dp_randf(env, -0.04f, 0.04f);
    env->x_dot = dp_randf(env, -0.04f, 0.04f);
    env->theta1 = M_PI + dp_randf(env, -0.08f, 0.08f);
    env->theta1_dot = dp_randf(env, -0.04f, 0.04f);
    env->theta2 = M_PI + dp_randf(env, -0.08f, 0.08f);
    env->theta2_dot = dp_randf(env, -0.04f, 0.04f);
    env->tick = 0;
    env->episode_return = 0.0f;
    env->upright_steps = 0;
    env->max_upright_steps = 0;
    compute_observations(env);
}

static void solve_3x3(float A[3][3], float b[3], float x[3]) {
    for (int i = 0; i < 3; i++) {
        int pivot = i;
        float best = fabsf(A[i][i]);
        for (int r = i + 1; r < 3; r++) {
            float v = fabsf(A[r][i]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (pivot != i) {
            for (int c = i; c < 3; c++) {
                float tmp = A[i][c];
                A[i][c] = A[pivot][c];
                A[pivot][c] = tmp;
            }
            float tmp = b[i];
            b[i] = b[pivot];
            b[pivot] = tmp;
        }

        float inv = 1.0f / A[i][i];
        for (int c = i; c < 3; c++) A[i][c] *= inv;
        b[i] *= inv;
        for (int r = 0; r < 3; r++) {
            if (r == i) continue;
            float f = A[r][i];
            for (int c = i; c < 3; c++) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    x[0] = b[0];
    x[1] = b[1];
    x[2] = b[2];
}

// Acceleration is a pure function of the angles, angular velocities and force
// (it does not depend on cart position/velocity), so RK4 can evaluate it at
// trial states.  qdd = [xdd, th1dd, th2dd].
static void dp_accel(DoublePendulum* env, const float th[2], const float w[2],
                     float force, float qdd[3]) {
    float m0 = env->cart_mass;
    float m1 = env->link1_mass;
    float m2 = env->link2_mass;
    float l1 = env->link1_length;
    float l2 = env->link2_length;
    float t1 = th[0];
    float t2 = th[1];
    float w1 = w[0];
    float w2 = w[1];
    float c1 = cosf(t1);
    float c2 = cosf(t2);
    float s1 = sinf(t1);
    float s2 = sinf(t2);
    float c12 = cosf(t1 - t2);
    float s12 = sinf(t1 - t2);

    float A[3][3] = {
        {m0 + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2},
        {(m1 + m2) * l1 * c1, (m1 + m2) * l1 * l1, m2 * l1 * l2 * c12},
        {m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2 * l2},
    };
    float b[3] = {
        force + (m1 + m2) * l1 * s1 * w1 * w1 + m2 * l2 * s2 * w2 * w2,
        (m1 + m2) * env->gravity * l1 * s1 - m2 * l1 * l2 * s12 * w2 * w2,
        m2 * env->gravity * l2 * s2 + m2 * l1 * l2 * s12 * w1 * w1,
    };
    solve_3x3(A, b, qdd);
}

// q = [x, theta1, theta2], v = [x_dot, theta1_dot, theta2_dot].
static void dp_state_deriv(DoublePendulum* env, const float q[3], const float v[3],
                           float force, float dq[3], float dv[3]) {
    for (int i = 0; i < 3; i++) dq[i] = v[i];
    float th[2] = {q[1], q[2]};
    float w[2] = {v[1], v[2]};
    dp_accel(env, th, w, force, dv);
}

static void dp_rk4_step(DoublePendulum* env, float force, float h,
                        float q[3], float v[3]) {
    float k1q[3], k1v[3], k2q[3], k2v[3], k3q[3], k3v[3], k4q[3], k4v[3];
    float tq[3], tv[3];
    dp_state_deriv(env, q, v, force, k1q, k1v);
    for (int i = 0; i < 3; i++) { tq[i] = q[i] + 0.5f*h*k1q[i]; tv[i] = v[i] + 0.5f*h*k1v[i]; }
    dp_state_deriv(env, tq, tv, force, k2q, k2v);
    for (int i = 0; i < 3; i++) { tq[i] = q[i] + 0.5f*h*k2q[i]; tv[i] = v[i] + 0.5f*h*k2v[i]; }
    dp_state_deriv(env, tq, tv, force, k3q, k3v);
    for (int i = 0; i < 3; i++) { tq[i] = q[i] + h*k3q[i]; tv[i] = v[i] + h*k3v[i]; }
    dp_state_deriv(env, tq, tv, force, k4q, k4v);
    for (int i = 0; i < 3; i++) {
        q[i] += (h/6.0f) * (k1q[i] + 2.0f*k2q[i] + 2.0f*k3q[i] + k4q[i]);
        v[i] += (h/6.0f) * (k1v[i] + 2.0f*k2v[i] + 2.0f*k3v[i] + k4v[i]);
    }
}

void integrate_physics(DoublePendulum* env, float force) {
    float q[3] = {env->x, env->theta1, env->theta2};
    float v[3] = {env->x_dot, env->theta1_dot, env->theta2_dot};
    int substeps = env->substeps > 0 ? env->substeps : 1;
    float h = env->dt / (float)substeps;
    for (int s = 0; s < substeps; s++) {
        dp_rk4_step(env, force, h, q, v);
        v[0] = fminf(fmaxf(v[0], -20.0f), 20.0f);
        v[1] = fminf(fmaxf(v[1], -30.0f), 30.0f);
        v[2] = fminf(fmaxf(v[2], -30.0f), 30.0f);
    }
    env->x = q[0];
    env->x_dot = v[0];
    env->theta1 = wrap_pi(q[1]);
    env->theta1_dot = v[1];
    env->theta2 = wrap_pi(q[2]);
    env->theta2_dot = v[2];
}

float upright_reward(DoublePendulum* env, float force) {
    float tip_y = env->link1_length * cosf(env->theta1)
        + env->link2_length * cosf(env->theta2);
    float max_y = env->link1_length + env->link2_length;
    float height = 0.5f * (tip_y / max_y + 1.0f);

    bool stable = height > 0.9f
        && fabsf(env->theta1_dot) < 1.5f
        && fabsf(env->theta2_dot) < 1.5f
        && fabsf(env->x_dot) < 1.0f;
    if (stable) env->upright_steps += 1;
    else env->upright_steps = 0;
    if (env->upright_steps > env->max_upright_steps) {
        env->max_upright_steps = env->upright_steps;
    }

    // Reward: (1-w) * [h * centered * small_control * small_velocity] + w * balance_quality
    float h = fminf(fmaxf(height, 0.0f), 1.0f);
    float sw1 = env->theta1_dot / DP_ANG_VEL_MARGIN;
    float sw2 = env->theta2_dot / DP_ANG_VEL_MARGIN;
    float min_vel_tol = fminf(expf(-DP_LN_10 * sw1 * sw1),
                              expf(-DP_LN_10 * sw2 * sw2));
    float small_velocity = DP_VELOCITY_FLOOR
        + (1.0f - DP_VELOCITY_FLOOR) * min_vel_tol;
    float scaled_x = env->x / DP_CENTER_MARGIN;
    float centered = 0.5f * (1.0f + expf(-DP_LN_10 * scaled_x * scaled_x));
    float a = env->force_mag > 0.0f ? force / env->force_mag : 0.0f;
    float small_control = 0.2f * (4.0f + fmaxf(0.0f, 1.0f - a * a));
    float deepmind = h * centered * small_control * small_velocity;
    float angle_mse = 0.5f * (env->theta1 * env->theta1
        + env->theta2 * env->theta2);
    float ang_vel_mse = 0.5f * (env->theta1_dot * env->theta1_dot
        + env->theta2_dot * env->theta2_dot);
    float balance_x = env->x / DP_BALANCE_CART_SCALE;
    float balance_quality = expf(
        -angle_mse / (DP_BALANCE_ANGLE_SCALE * DP_BALANCE_ANGLE_SCALE)
        -ang_vel_mse / (DP_BALANCE_ANG_VEL_SCALE * DP_BALANCE_ANG_VEL_SCALE)
        -balance_x * balance_x);
    float w = env->balance_bonus_weight;
    return (1.0f - w) * deepmind + w * balance_quality;
}

void c_step(DoublePendulum* env) {
    float a = env->actions[0];
    if (!isfinite(a)) a = 1.0f;
    int action = (int)a;
    if ((unsigned)action >= DP_ACTIONS) action = 1;
    float force = 0.0f;
    if (action == 0) force = -env->force_mag;
    else if (action == 2) force = env->force_mag;

    integrate_physics(env, force);
    env->tick += 1;

    bool invalid = !isfinite(env->x) || !isfinite(env->x_dot)
        || !isfinite(env->theta1) || !isfinite(env->theta1_dot)
        || !isfinite(env->theta2) || !isfinite(env->theta2_dot);
    bool x_done = env->x < -DP_X_THRESHOLD || env->x > DP_X_THRESHOLD;
    bool timeout = env->tick >= DP_MAX_STEPS;
    bool done = invalid || x_done || timeout;
    env->rewards[0] = upright_reward(env, force);
    env->episode_return += env->rewards[0];
    env->terminals[0] = (invalid || x_done) ? 1.0f : 0.0f;

    if (done) {
        add_log(env, invalid || x_done, timeout);
        c_reset(env);
        return;
    }
    compute_observations(env);
}

void c_render(DoublePendulum* env) {
    if (!IsWindowReady()) {
        InitWindow(DP_WIDTH, DP_HEIGHT, "PufferLib Double Pendulum");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (IsKeyPressed(KEY_TAB)) ToggleFullscreen();
    if (!isfinite(env->x) || !isfinite(env->theta1) || !isfinite(env->theta2)) return;

    float rail_y = DP_HEIGHT * 0.72f;
    float cart_x = DP_WIDTH / 2.0f + env->x * DP_SCALE;
    cart_x = fminf(fmaxf(cart_x, 32.0f), DP_WIDTH - 32.0f);
    float cart_y = rail_y - 16.0f;
    float l1 = env->link1_length * 2.0f * DP_SCALE;
    float l2 = env->link2_length * 2.0f * DP_SCALE;
    Vector2 p0 = {cart_x, cart_y};
    Vector2 p1 = {cart_x + sinf(env->theta1) * l1, cart_y - cosf(env->theta1) * l1};
    Vector2 p2 = {p1.x + sinf(env->theta2) * l2, p1.y - cosf(env->theta2) * l2};

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawLine(0, (int)rail_y, DP_WIDTH, (int)rail_y, PUFF_CYAN);
    DrawRectangle((int)(cart_x - 28), (int)(cart_y - 12), 56, 24, PUFF_CYAN);
    DrawLineEx(p0, p1, 7.0f, PUFF_RED);
    DrawLineEx(p1, p2, 6.0f, PUFF_YELLOW);
    DrawCircleV(p0, 8.0f, PUFF_WHITE);
    DrawCircleV(p1, 8.0f, PUFF_WHITE);
    DrawCircleV(p2, 10.0f, PUFF_WHITE);
    DrawText(TextFormat("steps %d  return %.1f  hold %d/%d",
        env->tick, env->episode_return, env->upright_steps, env->max_upright_steps),
        20, 20, 20, PUFF_WHITE);
    DrawText(TextFormat("x %.2f  theta1 %.1f  theta2 %.1f",
        env->x, env->theta1 * 180.0f / M_PI, env->theta2 * 180.0f / M_PI),
        20, 48, 20, PUFF_WHITE);
    EndDrawing();
}

void c_close(DoublePendulum* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
