#ifndef PUFFERLIB_MATSCI_DYNAMICS_H
#define PUFFERLIB_MATSCI_DYNAMICS_H

#include <math.h>

#define MATSCI_BOX_LO (-10.0)
#define MATSCI_BOX_HI (10.0)
#define MATSCI_BOX_LENGTH (MATSCI_BOX_HI - MATSCI_BOX_LO)
#define MATSCI_TIMESTEP (0.5)

typedef struct {
    double x, y, z;
} MatsciPosition;

static inline double matsci_wrap_periodic(double value) {
    double wrapped = fmod(value - MATSCI_BOX_LO, MATSCI_BOX_LENGTH);
    if (wrapped < 0.0) wrapped += MATSCI_BOX_LENGTH;
    return wrapped + MATSCI_BOX_LO;
}

static inline MatsciPosition matsci_integrate_ballistic(
    MatsciPosition position,
    float velocity_x,
    float velocity_y,
    float velocity_z
) {
    position.x = matsci_wrap_periodic(
        position.x + MATSCI_TIMESTEP * (double)velocity_x);
    position.y = matsci_wrap_periodic(
        position.y + MATSCI_TIMESTEP * (double)velocity_y);
    position.z = matsci_wrap_periodic(
        position.z + MATSCI_TIMESTEP * (double)velocity_z);
    return position;
}

#endif
