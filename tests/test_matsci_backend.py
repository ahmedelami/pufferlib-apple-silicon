import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_matsci_backend_selection_is_explicit():
    source = (ROOT / 'ocean' / 'matsci' / 'matsci.h').read_text()
    assert '__has_include' not in source
    assert '#ifndef PUFFERLIB_USE_LAMMPS' in source
    assert '#if PUFFERLIB_USE_LAMMPS' in source


def test_native_ballistic_dynamics_match_periodic_zero_pair_model():
    compiler = shutil.which('clang') or shutil.which('cc')
    if compiler is None:
        pytest.skip('a C compiler is required for the Matsci dynamics test')

    program = r'''
        #include <math.h>
        #include "ocean/matsci/dynamics.h"

        static int close_enough(double a, double b) {
            return fabs(a - b) < 1e-12;
        }

        int main(void) {
            if (!close_enough(matsci_wrap_periodic(-10.0), -10.0)) return 1;
            if (!close_enough(matsci_wrap_periodic(10.0), -10.0)) return 2;
            if (!close_enough(matsci_wrap_periodic(31.0), -9.0)) return 3;
            if (!close_enough(matsci_wrap_periodic(-31.0), 9.0)) return 4;

            MatsciPosition p = {9.0, -9.0, 0.0};
            p = matsci_integrate_ballistic(p, 4.0f, -4.0f, 82.0f);
            if (!close_enough(p.x, -9.0)) return 5;
            if (!close_enough(p.y, 9.0)) return 6;
            if (!close_enough(p.z, 1.0)) return 7;

            for (int i = 0; i < 10000; i++) {
                p = matsci_integrate_ballistic(
                    p, 123.25f, -99.5f, 1000.125f);
                if (p.x < MATSCI_BOX_LO || p.x >= MATSCI_BOX_HI) return 8;
                if (p.y < MATSCI_BOX_LO || p.y >= MATSCI_BOX_HI) return 9;
                if (p.z < MATSCI_BOX_LO || p.z >= MATSCI_BOX_HI) return 10;
            }
            return 0;
        }
    '''
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        source = directory / 'matsci_dynamics_test.c'
        executable = directory / 'matsci_dynamics_test'
        source.write_text(program)
        subprocess.run(
            [
                compiler,
                '-std=c11',
                '-Wall',
                '-Wextra',
                '-Werror',
                '-I',
                str(ROOT),
                str(source),
                '-lm',
                '-o',
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True)
